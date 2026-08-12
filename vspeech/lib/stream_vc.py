"""The reusable core of streaming VC (ADR-0053).

Converts statefully by concatenating a fixed-length block with a rolling left context.
It reuses the internals of the existing `change_voice` (HuBERT features / f0 / infer /
int16 conversion) as-is and leaves the utterance-path `change_voice` untouched. Block
boundaries are joined with a crossfade (amplitude-preserving sum=1 with SOLA on,
equal-power with SOLA off), and SOLA aligns the phase before mixing
(`_emit_with_crossfade`).

The pure helpers (next_context / crossfade_weights / overlap_add / sola_offset) are
written in terms of `len(seq)` so they stay agnostic about the sequence type, and this
module imports fine on a CPU machine without the rvc extra (the heavy imports happen
inside StreamingVc's methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: used by StreamingVc's annotations below. The pure helpers
    # (next_context / crossfade_weights / overlap_add)
    # don't need these, so keeping the imports under TYPE_CHECKING (rather
    # than module-level) still lets this module import on a CPU machine
    # without onnxruntime/the rvc extra.
    import numpy as np
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession

    from vspeech.config import RvcConfig
    from vspeech.lib.cuda_util import Device
    from vspeech.lib.rvc import HubertSession


def next_context(seq, context_len: int):
    """The last `context_len` elements of `seq` (the next tick's left context).

    When `context_len == 0`, `seq[-0:]` would return the whole thing, so return an
    explicit empty slice. Written in terms of `len(seq)` and slicing only, so it makes no
    assumption about the sequence type. When `context_len >= len(seq)` the whole sequence
    is returned (a clamp -- a defensive guard that passes as much left context as is
    available; the StreamingVc caller pre-fills the context to full length).
    """
    if context_len <= 0:
        return seq[:0]
    return seq[max(0, len(seq) - context_len) :]


def crossfade_weights(
    n: int, *, correlated: bool
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Crossfade weights `(fade_in, fade_out)` of length n. The fade law is chosen by
    whether adjacent renders are correlated (i.e. whether SOLA is on). `correlated` is a
    required keyword-only argument.

    - `correlated=True` (SOLA on): cell-centred sin²/cos² (the `(1-cos)/2` family), which
      is **amplitude-preserving** (`fade_in + fade_out == 1`). SOLA aligns the phase and
      thereby deliberately correlates adjacent renders (correlation at the alignment
      point rho ~ 0.82), and mixing correlated signals with sum=1 gives unity gain. Using
      equal power (sum of squares = 1) here over-adds at the seam (+1.14dB on the SOLA-on
      re-measurement, ADR-0053). w-okada's VCClient also pairs SOLA with sum-to-1.
    - `correlated=False` (SOLA off, `sola_search_len == 0`): sin/cos, which is **equal
      power** (`fade_in**2 + fade_out**2 == 1`). With SOLA off adjacent renders are
      uncorrelated (rho ~ 0), and mixing uncorrelated signals with sum=1 puts an about
      -1.25dB notch in the crossfade band (a faint tremolo at the block rate). Equal
      power keeps the summed power flat even for uncorrelated addition, and as the
      equal-power law it matches the pre-SOLA output down to float32 rounding (~1 ULP,
      max|delta| ~ 1.19e-7). It is not a strict bit match -- folding the weights first as
      `theta=0.5*pi*x; sin(theta)` differs by ~1 ULP from pre-SOLA's literal
      `sin(0.5*pi*x)` -- but that is far below int16 quantization and inaudible (the
      computation is harmless, so leave it).

    `n <= 0` returns empty arrays.
    """
    import numpy as np

    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    theta = 0.5 * np.pi * ((np.arange(n, dtype=np.float32) + 0.5) / n)
    if correlated:
        fade_in = (np.sin(theta) ** 2).astype(np.float32)
        fade_out = (np.cos(theta) ** 2).astype(np.float32)
    else:
        fade_in = np.sin(theta).astype(np.float32)
        fade_out = np.cos(theta).astype(np.float32)
    return fade_in, fade_out


def overlap_add(prev_tail, head, fade_in, fade_out):
    """Fade `prev_tail` out, fade `head` in, and add them.

    The one line of a crossfade overlap-add. Being elementwise it makes no assumption
    about the sequence type (callers pass equal lengths in the same domain).
    """
    return prev_tail * fade_out + head * fade_in


# RMS floor of the tail below which SOLA gives up (as a ratio of int16 full scale,
# 32768). 1e-4 is about -80dBFS. The previous absolute 1e-9 test only fired on "perfect
# digital silence"; at a realistic noise floor (measured RMS 0.000298 * 32768 ~ 9.8 int16
# units) it correlated noise against noise and let argmax pick an effectively random lag.
# In a region where there is no phase to align to at all, the right answer is not to
# search and to return the nominal (unshifted) position.
_SOLA_MIN_RMS = 32768.0 * 1e-4

# A tiny bias that pulls toward the centre (the nominal lag) when the correlation surface
# is nearly flat. Normalized correlation lives in [-1, 1], so a penalty of at most 1e-3
# normalized by the search half-width cannot affect a real peak (measured 0.89 vs 0.02)
# and only matters on a genuinely tied surface (e.g. a constant-DC region where the
# normalized correlation is 1.0 at every lag). Without it argmax always picks index 0 =
# the earliest point of the search window = the largest negative shift.
_SOLA_CENTER_BIAS = 1e-3


def sola_offset(prev_tail, region):
    """Return the start position (index) within `region` that correlates best with
    `prev_tail`.

    The RVC decoder is stateless, so it returns a phase-shifted waveform on every tick
    even for the same input span (measured: correlation -0.02 at lag 0 versus 0.89 at the
    best lag). Mixing at a fixed position adds the same waveform to itself a few ms
    apart, which forms a comb filter and produces that "electric fan" sound. Align the
    phase before mixing.

    The return value is an index from the start of `region`
    (0 <= i <= len(region)-len(prev_tail)). What corresponds to "no shift" (nominal) is
    **the centre `(len(region)-n)//2`, not index 0** -- the caller cuts region starting
    one search half-width earlier. So "giving up the search" returns the centre, not 0:

    - `prev_tail`'s RMS is below `_SOLA_MIN_RMS` (effectively silent) -> the centre.
      There is no phase to align to, and the argmax of noise against noise is an
      arbitrary lag.
    - The correlation surface is nearly flat (tied) -> `_SOLA_CENTER_BIAS` lets the
      centre win.

    Only when `region` is shorter than `prev_tail` is not a single window available, so
    that returns 0.
    """
    import numpy as np

    n = len(prev_tail)
    if n == 0 or len(region) < n:
        return 0
    center = (len(region) - n) // 2  # the index corresponding to no shift (nominal)
    tail_rms = float(
        np.sqrt(np.mean(np.square(np.asarray(prev_tail, dtype=np.float64))))
    )
    if tail_rms < _SOLA_MIN_RMS:
        return center
    tail_norm = float(np.linalg.norm(prev_tail))
    win = np.lib.stride_tricks.sliding_window_view(region, n)
    num = win @ prev_tail
    den = np.linalg.norm(win, axis=1) * tail_norm + 1e-9
    score = num / den
    # A tiny penalty proportional to the distance from the centre. Divided by the search
    # half-width, so it peaks at _SOLA_CENTER_BIAS regardless of window length or sample
    # rate. O(len(region)) and cheap.
    lags = np.arange(score.shape[0])
    score = score - _SOLA_CENTER_BIAS * np.abs(lags - center) / max(center, 1)
    return int(np.argmax(score))


class StreamingVc:
    """Stateful VC over fixed blocks plus a rolling left context (ADR-0053).

    On every tick it assembles `[context | block]` (16kHz), runs it through the existing
    `change_voice` internals (HuBERT features -> f0 -> infer -> int16), keeps only the
    block's worth of output and updates the context. Fixing block_len / context_len fixes
    the input shape, so a single warmup suffices (no re-autotune afterwards).

    The heavy dependencies (numpy, the rvc internals) are imported here for the first
    time. `rvc_config`'s f0_extractor_type must match the `f0_session` that is passed in.
    """

    def __init__(
        self,
        rvc_config: RvcConfig,
        device: Device,
        hubert_model: HubertSession,
        session: InferenceSession,
        f0_session: InferenceSession | None,
        target_sample_rate: int,
        f0_enabled: bool,
        emb_output_layer: int,
        use_final_proj: bool,
        block_len: int,
        context_len: int,
        crossfade_len: int = 0,
        sola_search_len: int = 0,
        lookahead_len: int = 0,
    ) -> None:
        import numpy as np

        from vspeech.lib.rvc import _is_model_half

        self.rvc_config = rvc_config
        # The device value travels unchanged all the way to onnxruntime: since ADR-0081
        # the conversion path binds `OrtValue`s instead of framework tensors, so there is
        # no framework device object to convert to.
        self.device = device
        self.hubert_model = hubert_model
        self.session = session
        self.f0_session = f0_session
        self.target_sample_rate = target_sample_rate
        self.f0_enabled = f0_enabled
        self.emb_output_layer = emb_output_layer
        self.use_final_proj = use_final_proj
        self.block_len = block_len
        self.context_len = context_len
        self._is_half = _is_model_half(session)
        self._sid = np.zeros(1, dtype=np.int64)
        self._context = np.zeros(context_len, dtype=np.float32)

        self.crossfade_len = crossfade_len
        # SOLA search half-width (in 16kHz input samples). 0 disables SOLA = the previous
        # fixed-position behaviour.
        self.sola_search_len = sola_search_len
        # How many input samples earlier than the tail anchor to read the emit from.
        # Buying right context this way costs exactly this much extra latency; the caller
        # is expected to extend context_len by the same amount so the left context at the
        # emit start does not shrink (ADR-0072).
        self.lookahead_len = lookahead_len
        # The output-domain lengths of the crossfade (hop / crossfade / SOLA search
        # half-width) are derived from the real-time clock
        # (`* target_sample_rate / 16000`). Deriving them as a ratio of the render length
        # out.shape[0] comes out short by HuBERT's receptive field (about 320 input
        # samples) and starves the sink. Only the read position keeps its derivation from
        # out.shape[0] (anchored at the tail) so that the truncated tail is avoided. The
        # lengths are constant across ticks -> computed on the first emit and cached.
        self._xfade_cache: (
            tuple[int, int, int, int, NDArray[np.float32], NDArray[np.float32]] | None
        ) = None
        self._output_tail = (
            None  # zeros(out_xf) is created lazily on the first crossfade
        )
        # How many samples before the start of the input block the content of the latest
        # emit begins (at the output rate). The emit comes out late by the crossfade and
        # by HuBERT's receptive-field truncation, so anything that overlays the output in
        # time alignment (the VAD gate's mask, ADR-0059) corrects with this. The value is
        # derived from the **nominal** read position and is therefore constant across
        # ticks -- SOLA's lag is deliberately not folded in (see the corresponding note in
        # `_emit_with_crossfade`).
        self.emit_delay_samples = 0
        if crossfade_len > 0 and context_len < crossfade_len:
            raise ValueError(
                "context_len must be >= crossfade_len for context-overlap crossfade"
            )
        if crossfade_len > 0 and crossfade_len >= block_len:
            raise ValueError("crossfade_len must be < block_len")

    def warmup(self, n: int = 3) -> None:
        """Build the ONNX graph / CUDA kernels up front using blocks of zeros.

        block_len is fixed, so pushing the shape through is enough -- the real values do
        not matter and nothing stalls afterwards. The context is reset to zeros after
        warmup.
        """
        import numpy as np

        zeros = np.zeros(self.block_len, dtype=np.float32)
        for _ in range(n):
            self.process_block(zeros)
        self._reset_context()

    def _reset_context(self) -> None:
        import numpy as np

        self._context = np.zeros(self.context_len, dtype=np.float32)
        # The crossfade tail is rolling state too. Reset it so a stale tail left over from
        # warmup cannot leak into the seam of the first real block (the next emit
        # re-initializes it to zeros -> fade in from silence).
        self._output_tail = None
        self._xfade_cache = None

    def process_block(self, block: NDArray[np.float32]) -> NDArray[np.int16]:
        """Convert block_len samples of 16kHz float32 [-1,1] and return an int16 block."""
        import numpy as np

        from vspeech.lib.rvc import _align_pitch_to_feats
        from vspeech.lib.rvc import _extract_hubert_feats
        from vspeech.lib.rvc import _select_pitch
        from vspeech.lib.rvc import _to_int16
        from vspeech.lib.rvc import infer

        block_f = np.ascontiguousarray(block, dtype=np.float32)
        # fixed length L = context_len + block_len
        seq = np.concatenate([self._context, block_f])

        feats = _extract_hubert_feats(
            hubert_model=self.hubert_model,
            audio_pad=seq,
            device=self.device,
            emb_output_layer=self.emb_output_layer,
            use_final_proj=self.use_final_proj,
        )

        p_len = seq.shape[0] // self.rvc_config.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        pitch, pitchf = _select_pitch(
            audio_pad=seq,
            rvc_config=self.rvc_config,
            f0_enabled=self.f0_enabled,
            p_len=p_len,
            f0_session=self.f0_session,
        )

        feats_len = feats.shape[1]
        pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
        p_len_array = np.array([feats_len], dtype=np.int64)

        out = _to_int16(
            infer(
                is_half=self._is_half,
                session=self.session,
                device=self.device,
                feats=feats,
                pitch_length=p_len_array,
                pitch=pitch,
                pitchf=pitchf,
                sid=self._sid,
            )[0]
        )

        # A view into `seq`, which is rebuilt from scratch every tick and never mutated,
        # so no copy is needed to keep the context stable.
        self._context = next_context(seq, self.context_len)
        if self.crossfade_len > 0:
            return self._emit_with_crossfade(out)
        return self._emit_no_crossfade(out)

    def _emit_no_crossfade(self, out: NDArray[np.int16]) -> NDArray[np.int16]:
        """The emit path when crossfade is disabled (exactly one hop, anchored at the
        tail minus the lookahead).

        The length comes from the real-time clock. Deriving it as a ratio of the render
        length comes out short by HuBERT's receptive field (about 320 input samples) and
        starves the sink. The read position stays anchored at the tail, offset earlier by
        `lookahead_len` (symmetric with `_emit_with_crossfade`), so the truncated tail is
        avoided and `lookahead_len == 0` reproduces the pre-lookahead read position sample
        for sample.
        """
        out_hop = round(self.block_len * self.target_sample_rate / 16000)
        out_look = round(self.lookahead_len * self.target_sample_rate / 16000)
        if out.shape[0] < out_hop + out_look:
            # The render is shorter than one hop plus the lookahead = a broken config
            # whose context_ms is too short (the crossfade path raises ValueError here).
            # In this branch the emit length already falls short of a hop and the rate
            # lock is broken, so the delay we report is meaningless too (ctx_out becomes
            # the whole context and the gate's mask clamps everywhere to the previous
            # block's value). The principled fix would be to fail loud like the crossfade
            # path, but that changes the behaviour of existing crossfade_ms=0 configs, so
            # it is left as separate work.
            self.emit_delay_samples = self._emit_delay(0)
            return out
        start = out.shape[0] - out_hop - out_look
        self.emit_delay_samples = self._emit_delay(start)
        return out[start : start + out_hop]

    def _emit_delay(self, start: int) -> int:
        """How many samples before the start of the input block the emit starts (at the
        output rate).

        The decoder render is aligned to the start of the analysis window
        `[context | block]` (only the tail gets truncated), so an index into the render
        is directly a position within the window. The block starts at sample
        `context_len` of the window = `context_len * rate / 16000` at the output rate.
        The difference from the read start `start` is therefore "how much earlier the
        emit sounds".
        """
        ctx_out = round(self.context_len * self.target_sample_rate / 16000)
        return ctx_out - start

    def _emit_with_crossfade(self, out: NDArray[np.int16]) -> NDArray[np.int16]:
        """Align the phase with SOLA, overlap-add, and return exactly one real-time hop.

        The output-domain lengths (hop / crossfade / SOLA search half-width) are derived
        from the **real-time clock** (`block_len * target_sample_rate / 16000` and so
        on). Deriving them as a ratio of the render length out.shape[0] is wrong: HuBERT's
        receptive field truncates a fixed amount off the tail (about 320 input samples),
        which shortens the hop by that much and permanently starves the sink (measured
        3.03% = 30.3ms/s). With the shape fixed the lengths are constant across ticks, so
        they are computed once and cached.

        The **read position, on the other hand, is still derived backwards from out_total
        (anchored at the tail)**, which naturally avoids the truncated tail.

        The previous tick kept the last out_xf samples of its emit in `_output_tail`, and
        this tick blends it (amplitude-preserving, sum=1) with the re-rendered span
        covering the same input time to form the start of this emit (a true overlap-add
        at the seam). But the RVC decoder is stateless, so the same input span picks up a
        phase shift of a few ms on every tick (measured: correlation -0.02 at lag 0 versus
        0.89 at the best lag). Mixing at a fixed position would form a comb filter, so
        before mixing `sola_offset` searches a +/-out_sola window for the read position
        `start` that correlates best with `_output_tail` (SOLA = Synchronous
        OverLap-Add). The fade law switches on whether SOLA is on
        (`crossfade_weights(..., correlated=self.sola_search_len > 0)`). With SOLA on
        adjacent renders are deliberately correlated (correlation at the alignment point
        rho = 0.82), so amplitude preservation (sum=1, sin²/cos²) gives unity gain (equal
        power over-adds by +1.14dB, while sum-to-1 sits at -0.76dB, near unity; w-okada's
        VCClient also pairs SOLA with sum-to-1). With SOLA off (`sola_search_len == 0`)
        adjacent renders are uncorrelated (rho ~ 0), so equal power (sum of squares = 1,
        sin/cos) applies. Using sum=1 on uncorrelated signals puts an about -1.25dB notch
        in the band (a tremolo at the block rate). Equal power matches the pre-SOLA output
        down to float32 rounding (~1 ULP; not a strict bit match; ADR-0053).

        Index invariants:

        - The emit length is always exactly out_hop =
          `block_len * target_sample_rate / 16000` (the input hop mapped onto the sink's
          sample rate as a real-time length; no drift). The lag changes only *where we
          read from*, never *how much we emit*.
        - The largest index touched is
          `start + out_hop + out_xf <= (nominal + out_sola) + out_hop + out_xf
          == out_total`, so we never read past the render.
        - `nominal - out_sola >= 0` (the search window never runs off the front of the
          output) is guaranteed by clamping out_sola to
          `out_sola <= (out_total - out_hop - out_xf) // 2`.
        - When `out_sola == 0` (i.e. `sola_search_len == 0`), `start == nominal ==
          out_total - out_hop - out_xf`, so the read position matches pre-SOLA and the
          fade law also falls back to equal power (`correlated=False`); the samples
          emitted therefore match pre-SOLA down to float32 rounding (~1 ULP -- not a
          strict bit match because the weights are folded first, but below int16
          quantization and inaudible). The invariant holds precisely because both the
          read position and the weights match.
        - When `sola_offset` gives up the search (the tail is effectively silent, or the
          correlation surface is flat) it returns the **centre** `out_sola` of the region,
          so `start == nominal`. That is, it falls back to "no shift" (not to index 0 =
          the largest negative shift).
        - `lookahead_len` shifts `nominal` earlier by exactly `out_look`, so the emitted
          content is that much older and every emitted sample gains that much right
          context. It never changes the emit length, so the rate lock is untouched; the
          cost is exactly `out_look` of extra latency. `lookahead_len == 0` reproduces the
          pre-lookahead read position sample for sample.

        The algorithmic delay is out_sola samples (from moving the read position one
        search half-width earlier) plus HuBERT's receptive-field truncation of the tail
        (about 320 input samples = 20ms). Neither affects the emit length.
        """
        import numpy as np

        out_total = out.shape[0]
        if self._xfade_cache is None:
            r = self.target_sample_rate
            # Derive the lengths from the real-time clock. Deriving them as a ratio of
            # out_total shortens the hop by however much HuBERT's receptive field
            # truncates off the tail (about 320 input samples) and permanently starves
            # the output device (measured 3.03% = 30.3ms/s). The read position is still
            # derived backwards from out_total (anchored at the tail), which naturally
            # avoids the truncated tail.
            out_hop = round(self.block_len * r / 16000)
            out_xf = round(self.crossfade_len * r / 16000)
            out_sola = round(self.sola_search_len * r / 16000)
            out_look = round(self.lookahead_len * r / 16000)
            if out_total < out_hop:
                raise ValueError(
                    f"decoder output ({out_total}) < one hop ({out_hop}): "
                    "context_ms が短すぎる(HuBERT の受容野ぶん実効長が縮む)。"
                    "context_ms を増やすこと。"
                )
            # Keep the crossfade band at or below the hop and at or below the context
            # span (out_total-out_hop)
            out_xf = min(out_xf, out_hop, out_total - out_hop)
            # guarantee nominal - out_sola >= 0 (the search window never runs off the
            # front of the output)
            out_sola = max(0, min(out_sola, (out_total - out_hop - out_xf) // 2))
            # nominal - out_sola >= 0 must still hold with the lookahead subtracted. The
            # caller extends context_len by the lookahead, so out_total grows by the same
            # amount and this can only trip on a hand-built geometry -- fail loud rather
            # than clamp, or the measured lookahead would silently differ from the
            # configured one.
            if out_total - out_hop - out_xf - 2 * out_sola - out_look < 0:
                raise ValueError(
                    f"lookahead ({out_look}) が描画長に対して大きすぎる "
                    f"(out_total={out_total} hop={out_hop} xf={out_xf} "
                    f"sola={out_sola}): context_ms を増やすこと "
                    "(crossfade_ms・sola_search_ms が大きいほど必要な context_ms も"
                    "増える)。"
                )
            # The fade law follows whether SOLA is on: with SOLA the adjacent renders are
            # correlated (sum=1), without it they are uncorrelated (equal power). At
            # sola_search_len==0 this matches pre-SOLA to within ~1 ULP.
            fade_in, fade_out = crossfade_weights(
                out_xf, correlated=self.sola_search_len > 0
            )
            self._xfade_cache = (out_hop, out_xf, out_sola, out_look, fade_in, fade_out)
        out_hop, out_xf, out_sola, out_look, fade_in, fade_out = self._xfade_cache
        out_f = out.astype(np.float32)
        if self._output_tail is None:
            self._output_tail = np.zeros(out_xf, dtype=np.float32)
        # The read start. With out_sola=0 this is identical to before
        # (= out_total-out_hop-out_xf).
        nominal = out_total - out_hop - out_xf - out_sola - out_look
        if out_sola > 0:
            region = out_f[nominal - out_sola : nominal + out_sola + out_xf]
            start = (nominal - out_sola) + sola_offset(self._output_tail, region)
        else:
            start = nominal
        # The published delay excludes SOLA's lag (it is derived from `nominal`, not from
        # `start`). The lag moves by +/-out_sola every tick, and folding that into the
        # time axis would re-anchor the time axis of whoever overlays the output with it
        # (the VAD gate's mask) on every tick, making the gain jump at the emit seam
        # (measured up to 0.06 on hardware; the structural bound is 2*out_sola/window =
        # 0.31 = a click). The residual content offset is at most out_sola (5ms by
        # default), comfortably below the mask's 32ms resolution. SOLA is a fine
        # adjustment of the output's time reference, not a delay of the content.
        self.emit_delay_samples = self._emit_delay(nominal)
        head = out_f[start : start + out_xf]
        blended = overlap_add(self._output_tail, head, fade_in, fade_out)
        middle = out_f[start + out_xf : start + out_hop]
        emit_f = np.concatenate([blended, middle])
        self._output_tail = out_f[start + out_hop : start + out_hop + out_xf].copy()
        return np.clip(np.rint(emit_f), -32768.0, 32767.0).astype(np.int16)
