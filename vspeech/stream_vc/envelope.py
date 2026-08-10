"""Input envelope following for streaming VC (ADR-0057, ADR-0065).

Normalizes the input block's relative loudness envelope against a rolling EMA of the
mean input RMS and applies it to the output block as a duck gain
(clip(shape^strength, min_gain, max_gain)). Same ducking idea as the batch
apply_input_envelope (worker/vc.py), but this streaming version replaces the reference
"mean over the whole utterance" with a rolling EMA (only one block is available at a
time).

The shape is laid on the emit's **absolute sample grid**, carrying the previous block's
shape across the seam and correcting for the emit delay -- the same construction as the
VAD gate's mask (gate.py, ADR-0059). ADR-0057 v1 did neither: it interpolated on a
per-block normalized 0..1 axis, so the gain stepped at every block boundary (measured up
to the full rail-to-rail 0.5 = +7dB in a single sample at the tuned settings = a click at
the block rate) and the shape sat 50ms late on the audio. See ADR-0065.

Pure decide-and-apply logic; numpy is imported inside the method (so it can be unit
tested on CPU with no model and without pulling in torch/sounddevice -- the same shape
as gate.py).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Sample rate of the input block (16k, the same as capture.py's CAPTURE_RATE). Importing
# capture.py would pull in sounddevice, so it is kept as a constant here (to keep this
# module unit-testable on CPU).
_INPUT_RATE = 16000


class StreamingEnvelope:
    """Input envelope following against a rolling-EMA reference (duck, ADR-0057/0065).

    The state is the reference level `_ema_level` (a scalar) plus the shape history
    (`_history`), which the seam handover needs. `apply()` multiplies the output block by
    the current input block's relative loudness envelope and updates both for the next
    block.
    """

    def __init__(
        self,
        strength: float,
        min_gain: float,
        max_gain: float,
        window_ms: float,
        ema_ms: float,
        block_ms: float,
    ) -> None:
        self.strength = strength
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.window_ms = window_ms
        # Per-block EMA coefficient for time constant ema_ms:
        # alpha = 1 - exp(-block_ms/ema_ms).
        self._alpha = 1.0 - math.exp(-block_ms / ema_ms) if ema_ms > 0 else 1.0
        # initialized from the block mean on the first apply
        self._ema_level: float | None = None
        # The shapes of the most recent blocks, oldest first, as (shape, emit_len) pairs.
        # One block suffices while the emit delay stays below one emit length; lookahead
        # (ADR-0070) pushes it past that, and then the head of the emit carries audio
        # shaped two or more blocks ago. The length is derived per call from the delay.
        self._history: list[tuple[NDArray[np.float64], int]] = []

    def reset(self) -> None:
        """Return the reference level and the seam handover to uninitialized (called by
        the runner on pause/resume and on a capture reopen).

        So that a stale reference level does not oddly duck the next block after
        real time has jumped, force the next apply to cold start again (initializing from
        the block mean). The shape history is dropped for the same reason: it describes
        audio from before the jump, and the head of the next emit is rendered from a
        zeros context, so handing over from it would shape the wrong audio.
        """
        self._ema_level = None
        self._history = []

    def apply(
        self,
        out_i16: NDArray[np.int16],
        in_block: NDArray[np.float32],
        delay_samples: int,
    ) -> NDArray[np.int16]:
        """Duck the output block out_i16 by the relative loudness envelope of the input
        block in_block (16k float32), against the rolling EMA reference.

        The reference is the **past** EMA (history). On a cold start, or right after
        reset, it is initialized from the current block's mean (so the first block is not
        ducked unnaturally). The reference is updated before returning, so the next block
        uses an EMA that already includes this one.

        `delay_samples` is `StreamingVc.emit_delay_samples`: how many samples before the
        start of the input block the emit's content begins (at the output rate). The sound
        carried by emit sample j sits at position `j - delay_samples` relative to the
        start of the input block, so the shape is laid on that shifted grid -- identical
        to the VAD gate's mask overlay (gate.py, ADR-0059).

        **Known characteristic (ADR-0057, tuned by on-hardware ear checks):** during long
        silence the reference EMA drifts toward the input's noise floor (decaying with
        envelope_ema_ms). The phrase onset right after that is judged loud on every frame
        against the low reference and is barely ducked -- i.e. this block alone gets
        little shaping. Inter-word dips and decay tails within continuous speech are
        shaped correctly because the reference sits at speech level. The phrase onset is
        the VAD gate's job. Lengthening ema_ms keeps the reference at speech level across
        silence and makes onset shaping more effective.
        """
        import numpy as np

        out_len = int(out_i16.shape[0])
        if out_len == 0 or in_block.shape[0] == 0 or self.strength <= 0.0:
            return out_i16
        # Per-frame RMS of the input (the absolute scale is irrelevant: it cancels in the
        # reference normalization).
        frame_len = max(1, round(self.window_ms * _INPUT_RATE / 1000.0))
        n_frames = max(1, in_block.shape[0] // frame_len)
        bounds = np.linspace(0, in_block.shape[0], n_frames + 1).astype(np.int64)
        frame_rms = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            seg = in_block[bounds[i] : bounds[i + 1]].astype(np.float64)
            if seg.shape[0]:
                frame_rms[i] = np.sqrt(np.mean(seg**2))
        block_mean = float(frame_rms.mean())
        if self._ema_level is None:
            self._ema_level = block_mean
        ref = self._ema_level
        self._ema_level = self._alpha * block_mean + (1.0 - self._alpha) * ref
        # Output samples per input frame, and half a frame -- the margin the seam
        # continuity needs (see the bounds note below).
        half_frame = out_len / n_frames / 2.0
        # How many blocks of history the delay reaches back into. Sized off the length of
        # the most recently *stored* block, not this call's `out_len`: `out_len` is the
        # block just decided, not yet in `_history`, and what determines how far back the
        # already-stored blocks reach is their own recorded length, not the new one's (a
        # block can be shorter than the norm, e.g. a short buffer in a test, without
        # shrinking how much real history is required to be read back). Falls back to
        # `out_len` itself before any block has been stored (right after construction or a
        # reset). Constant across ticks in practice (the delay is nominal and the block
        # length does not change tick to tick), and exactly 1 for the geometry that existed
        # before lookahead -- which is why this is a no-op there.
        ref_len = self._history[-1][1] if self._history else out_len
        need = max(1, math.ceil((delay_samples + half_frame) / ref_len))
        history = self._history[-need:]
        if len(history) < need:
            # Startup, or right after a reset. The head of the emit is rendered from a
            # zeros context or from before a real-time jump, so hand over from **unity** --
            # the same "the first block is not ducked" cold start as `_ema_level`. Seed a
            # whole emit's worth of frames (not one): with a single element its centre
            # would land a whole emit earlier and stretch the ramp over two blocks.
            seed = np.ones(n_frames, dtype=np.float64)
            history = [(seed, out_len)] * (need - len(history)) + history
        # effectively digital silence (e.g. pure silence right after init) -> pass through
        if ref < 1e-8:
            # This block went out at unity, so hand unity over: leaving the older shape in
            # place would make the next block step off a value that was never applied.
            self._history.append((np.ones(n_frames, dtype=np.float64), out_len))
            del self._history[:-need]
            return out_i16
        # The relative shape (relative to the reference, not mean~1), linearly
        # interpolated onto the emit's sample grid.
        shape_now = frame_rms / ref
        self._history.append((shape_now, out_len))
        del self._history[:-need]
        # Frame centres on the emit's absolute sample grid. Each history block sits its
        # own emit length earlier, accumulated, which is what makes the gain continuous
        # across the seam: with the delay correction the seam falls in the interior of the
        # shape, where both blocks interpolate the *same* two frame centres with the same
        # values (ADR-0065). The emit length is carried per block rather than assumed
        # equal to `out_len` so a length change cannot silently shift an origin.
        centers_parts: list[NDArray[np.float64]] = []
        shape_parts: list[NDArray[np.float64]] = []
        offset = 0.0
        for past_shape, past_len in reversed(history):
            offset -= past_len
            k = past_shape.shape[0]
            centers_parts.append(
                (np.arange(k, dtype=np.float64) + 0.5) / k * past_len + offset
            )
            shape_parts.append(past_shape)
        centers_parts.reverse()
        shape_parts.reverse()
        centers_parts.append(
            (np.arange(n_frames, dtype=np.float64) + 0.5) / n_frames * out_len
        )
        shape_parts.append(shape_now)
        centers = np.concatenate(centers_parts)
        # Two bounds on the delay, both of which the validated geometry clears; outside
        # them the handover degrades gradually rather than breaking.
        # - Too large: the history now grows with the delay, so the head no longer clamps
        #   to the oldest frame. What remains bounded is the reference EMA: shaping audio
        #   from `need` blocks ago against a reference that has since moved on gets less
        #   apt as the lookahead grows. envelope_ema_ms (2000ms by default) is far longer
        #   than any usable lookahead, so this stays a second-order effect.
        # - Too small: exact continuity needs the seam to fall in the *interior* of the
        #   shape, i.e. `delay_samples >= half a frame` (out_len / n_frames / 2), where
        #   both blocks interpolate the same two centres. Below that the block's tail
        #   clamps to its last frame while the next block's head is already interpolating,
        #   leaving a partial step. Default geometry: 50ms of delay against a 13ms half
        #   frame (25ms window, 160ms block); even crossfade_ms=0, whose delay is only
        #   HuBERT's ~20ms truncation, still clears it. Raising envelope_window_ms towards
        #   block_ms shrinks the margin (n_frames falls, the half frame grows).
        shape = np.interp(
            np.arange(out_len, dtype=np.float64) - delay_samples,
            centers,
            np.concatenate(shape_parts),
        )
        gain = np.clip(np.power(shape, self.strength), self.min_gain, self.max_gain)
        out_f = out_i16.astype(np.float32)
        return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)
