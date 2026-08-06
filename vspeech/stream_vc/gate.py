"""The per-window VAD noise gate of streaming VC (ADR-0059 / ADR-0053 / ADR-0019).

The streaming path keeps running through silence, so without a gate **the room's noise
floor goes straight through RVC and comes out amplified**, continuously. On top of that,
at a phrase onset the tiny input just before phonation (measured RMS 0.002) is
synthesized into audible sound (measured at **+43dB** over the batch path for the same
audio and the same model). This depends on the contents of the analysis window: the
content encoder encodes the same sound differently depending on the left context (it was
measured not to be the f0 path; exactly which property of the window drives it has not
been isolated -- see ADR-0059). The model cannot be changed, so we **gate at the same
32ms window granularity as the batch path** to remove the audible component (end-to-end
measurement on real recordings: -25.4dB / -16.9dB on the onset breath, while real onsets
and steady state are preserved).

This module is model-independent pure logic holding only the decision and its
application; the Silero VAD itself is reused read-only from `vspeech/lib/vad.py` (shared
with the utterance path `[vc]`). It can therefore be unit tested on CPU with no model.

Design points:

- **Decide on the input block, apply to the output block.** Inference is never skipped
  even when the gate is closed. `StreamingVc` is a stateful conversion with a rolling
  left context and a crossfade tail, so skipping a block punches a hole in the context
  and breaks the seam when speech resumes. Only the emitted audio is attenuated.
- **Both the decision and the application are at 32ms window granularity** (the same
  idea as `speech_gate_mask` / `apply_vad_gate` in the utterance path's `lib/vad.py`).
  Taking the max window probability at block granularity (160ms) would open the entire
  block because of a single onset window and pass the pre-phonation breath before it at
  full gain.
- **The hangover never dilates forward.** The batch-side `speech_gate_mask` dilates
  symmetrically, but dilating forward in streaming opens the breath right before an
  onset (measured by applying the mask alone to real recordings: -26dB at 0ms forward,
  regressing to -9dB when 32ms is added). Only the backward direction is needed to
  protect word endings and inter-word gaps, so `hangover_ms` is used as a **backward
  dilation**.
- **Correct for the emit delay when overlaying.** The emit's content starts earlier than
  the input block (50ms by default, from the crossfade plus HuBERT's receptive-field
  truncation). Without the correction the mask lands on misaligned audio and the
  measured suppression drops from -26dB to -8dB. The delay is published by
  `StreamingVc.emit_delay_samples` (derived from the nominal position, hence constant
  across ticks).
- **The gain is linearly interpolated between window centres.** Stepping the gain at a
  boundary is itself a click, so it is handed over across 32ms (continuing from the tail
  of the previous block's mask).

As in `vspeech/lib/stream_vc.py`, numpy is imported inside the methods (to keep
importing this module cheap).
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import VAD_WINDOW_SAMPLES
from vspeech.lib.vad import VadCarry

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# The length of one Silero window (ms). The hangover is converted into a window count at
# this granularity.
_WINDOW_MS = VAD_WINDOW_SAMPLES * 1000.0 / VAD_SAMPLE_RATE


class StreamingVadGate:
    """A 32ms-window gate mask plus application with emit-delay correction.

    `window_gains()` returns this block's per-window gains and `apply()` maps them onto
    the emit's sample grid (with the delay correction) and multiplies. The state is just
    two things: how many windows have passed since speech was last seen, and the previous
    block's mask.
    """

    def __init__(self, threshold: float, hangover_ms: float, min_gain: float) -> None:
        self.threshold = threshold
        self.min_gain = min_gain
        # Silero's recurrent state. Rebuilding it per block cold-starts the RNN every
        # time and wrecks even the probabilities of clearly voiced windows (see VadCarry
        # in lib/vad.py). The runner passes it to speech_probs.
        self.vad_carry = VadCarry()
        self._hangover_windows = max(0, round(hangover_ms / _WINDOW_MS))
        # Windows since the last speech. Saturated at the budget so it stops growing
        # monotonically. The initial value is the "closed" state: at window granularity a
        # speech window opens the gate by itself, so there is no need to start open and
        # leak silence.
        self._since_speech = self._hangover_windows + 1
        self._prev_gains: NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Return to the closed state (hangover empty, no previous-block mask, and a fresh
        VAD state).

        Called by the runner on a transition so that, after real time has jumped from a
        pause/resume or a capture reopen, a stale hangover budget, a stale mask, or a VAD
        recurrent state grown on pre-jump audio cannot leak through and oddly open or
        attenuate the block right after. Fault state is not held here at all: the
        fail-open warning is thinned by a LogThrottle the runner owns (ADR-0062), so a
        pause cannot reset the log-thinning episode either.

        Keeping `vad_carry` was measured and **rejected** (see ADR-0059's Alternatives).
        Pausing mid-speech and resuming into silence lets the stale "in speech" state
        misjudge the first window as speech. A single misjudged window resets
        `_since_speech` to 0 and rearms the full hangover budget, so the leak does not
        stop at one window (measured: 8 leaks out of 104 cases, up to 320ms). That is
        precisely the "amplified tiny input becomes audible" this ADR exists to remove,
        so it cannot be traded for accuracy.
        """
        self._since_speech = self._hangover_windows + 1
        self._prev_gains = None
        self.vad_carry = VadCarry()

    def window_gains(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return this block's per-window gains from the window probabilities (backward
        dilation only).

        A speech window gets 1.0. A silent window gets 1.0 while it is within
        `hangover_ms` of the last speech and `min_gain` beyond that. The budget carries
        across block boundaries (decisions arrive per block, but speech does not care
        about the boundaries).
        """
        import numpy as np

        # [Open, deferred 2026-08-06] A single misjudged window rearms the WHOLE hangover
        # budget, so one false positive on a pre-phonation breath holds the gate wide open
        # (gain > 0.5) for 398ms at the rig's vad_hangover_ms=500 (300ms -> 366ms,
        # 100ms -> 174ms, 0ms -> 78ms; measured on this pure logic). That is the leading
        # suspect for the onset residual still audible on hardware after ADR-0059, and it
        # dwarfs the ramp floor noted in `apply`. Deferred rather than guessed at: which of
        # the two dominates has to be measured (do the windows before phonation actually
        # cross vad_threshold?), and the ear A/B that would otherwise decide it was
        # inconclusive on this rig. Raising vad_threshold is not the answer -- it costs
        # real onsets; the candidate fix is requiring N consecutive speech windows to open
        # (attack hysteresis), which is an ADR-level change.
        gains = np.empty(probs.shape[0], dtype=np.float64)
        for i in range(probs.shape[0]):
            if probs[i] >= self.threshold:
                self._since_speech = 0
            else:
                self._since_speech = min(
                    self._since_speech + 1, self._hangover_windows + 1
                )
            gains[i] = (
                1.0 if self._since_speech <= self._hangover_windows else self.min_gain
            )
        return gains

    def apply(
        self,
        out_i16: NDArray[np.int16],
        gains: NDArray[np.float64],
        delay_samples: int,
        sample_rate: int,
    ) -> NDArray[np.int16]:
        """Map the window gains onto the emit's sample grid and multiply (with delay
        correction).

        The sound carried by emit sample j sits at position `j - delay_samples` (at the
        output rate) relative to the start of the input block, so the mask is shifted by
        the same amount when overlaid. The first `delay_samples` of the emit correspond to
        the **previous** block's input, so the previous block's mask (`_prev_gains`) is
        concatenated on the left before interpolating -- which simultaneously guarantees
        gain continuity across the block boundary (no step = no click). That continuity
        only holds while `delay_samples` is constant across ticks, which is why
        `StreamingVc` publishes the **nominal** delay, excluding SOLA's lag (ADR-0059).

        All-1.0 gains (with the previous block also all 1.0) take an identity fast path
        that returns the input object as-is: with continuous speech, or with the feature
        off by default, the output is bit-identical to the ungated one (the single block
        right after startup or a reset is the exception, since it opens from the closed
        state).
        """
        import numpy as np

        n = int(out_i16.shape[0])
        if n == 0 or gains.shape[0] == 0:
            return out_i16
        # Output samples per window. Window centres are laid out on this grid and linearly
        # interpolated.
        step = VAD_WINDOW_SAMPLES * sample_rate / VAD_SAMPLE_RATE
        prev = self._prev_gains
        self._prev_gains = gains
        if prev is None:
            # No previous information (right after startup or a reset). The head of the
            # emit is audio rendered from before the real-time jump, or from a zeros
            # context, so start from the closed state (min_gain) -- matching
            # `_since_speech`'s initial value. Seed **a hop's worth of windows, not one**:
            # with a single element its centre lands a whole hop earlier (-144ms by
            # default), so the ramp is handed over across 160ms instead of 32ms and the
            # head never fully closes (measured -4.6dB). Count the windows the same way
            # the real mask does (ceil): round would give one window fewer when the block
            # length is not a multiple of the window length (block_ms=80), shifting the
            # last seed centre earlier and leaving the head not fully closed.
            prev = np.full(max(1, ceil(n / step)), self.min_gain, dtype=np.float64)
        if float(gains.min()) == 1.0 and float(prev.min()) == 1.0:
            return out_i16
        n_prev = int(prev.shape[0])
        # The previous block's origin is **one emit length (= hop) earlier**, not
        # "window count x window length". speech_probs zero-pads to ceil(block_len/512)
        # windows, so when block_len is not a multiple of 512 the windows total more than
        # the block length (e.g. 96ms at block_ms=80). Shifting by n_prev*step would move
        # the mask earlier by that difference (16ms at the 80ms setting).
        centers = np.concatenate(
            [
                (np.arange(n_prev, dtype=np.float64) + 0.5) * step - n,
                (np.arange(gains.shape[0], dtype=np.float64) + 0.5) * step,
            ]
        )
        all_gains = np.concatenate([prev, gains])
        # `prev` only holds one block, so in configurations where `delay_samples` exceeds
        # the hop (e.g. block_ms=80 with crossfade_ms=70) the head of the emit falls left
        # of the first window centre and is clamped to `prev[0]`. Being continuous it does
        # not click, but the mask over that span carries no information from two blocks
        # back.
        # [Open, deferred 2026-08-06] Interpolating between centres ties the opening ramp
        # to the 32ms window spacing, which puts a floor under onset suppression: measured
        # at the rig's settings the gain is already 0.50 (-6dB) at the moment the first
        # speech window *starts*, and only reaches -26dB 14.4ms before it. So the 16ms of
        # breath immediately before phonation cannot be held below -6dB no matter what the
        # VAD decides. A click needs only a few ms of ramp, so an asymmetric attack (short
        # opening ramp, 32ms closing ramp unchanged) would lower that floor -- an ADR-level
        # change, deferred until measurement says this rather than the hangover rearm in
        # `window_gains` is what dominates the residual.
        gain = np.interp(
            np.arange(n, dtype=np.float64) - delay_samples, centers, all_gains
        )
        out_f = out_i16.astype(np.float32) * gain
        return np.clip(np.rint(out_f), -32768.0, 32767.0).astype(np.int16)
