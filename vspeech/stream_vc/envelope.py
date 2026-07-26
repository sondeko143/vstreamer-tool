"""Input envelope following for streaming VC (ADR-0057).

Normalizes the input block's relative loudness envelope against a rolling EMA of the
mean input RMS and applies it to the output block as a duck gain
(clip(shape^strength, min_gain, max_gain)). Same ducking idea as the batch
apply_input_envelope (worker/vc.py), but this streaming version replaces the reference
"mean over the whole utterance" with a rolling EMA (only one block is available at a
time).

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
    """Input envelope following against a rolling-EMA reference (duck, ADR-0057).

    The only state is the reference level `_ema_level` (a scalar). `apply()` multiplies
    the output block by the current input block's relative loudness envelope and updates
    the reference EMA for the next block.
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

    def reset(self) -> None:
        """Return the reference level to uninitialized (called by the runner on
        pause/resume and on a capture reopen).

        So that a stale reference level does not oddly duck the next block after
        real time has jumped, force the next apply to cold start again (initializing from
        the block mean).
        """
        self._ema_level = None

    def apply(
        self, out_i16: NDArray[np.int16], in_block: NDArray[np.float32]
    ) -> NDArray[np.int16]:
        """Duck the output block out_i16 by the relative loudness envelope of the input
        block in_block (16k float32), against the rolling EMA reference.

        The reference is the **past** EMA (history). On a cold start, or right after
        reset, it is initialized from the current block's mean (so the first block is not
        ducked unnaturally). The reference is updated before returning, so the next block
        uses an EMA that already includes this one.

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
        # effectively digital silence (e.g. pure silence right after init) -> pass through
        if ref < 1e-8:
            return out_i16
        # Linearly interpolate the relative shape (relative to the reference, not mean~1)
        # onto the output sample grid.
        # [Unhandled, separate concern] This mapping assumes the input block and the emit
        # occupy the same time (a normalized 0..1 axis), but the real emit comes out later
        # than the input (about 50ms measured, from crossfade + SOLA + HuBERT's receptive
        # field). ADR-0059 corrected for that on the gate side via
        # StreamingVc.emit_delay_samples; this side was left alone because it is within
        # ADR-0057's scope. The impact is limited to a time offset in the shaping, which
        # is bounded by envelope_min/max_gain (0.1/1.0 by default, 0.6/0.9 on the real
        # rig). Fixing it means taking delay_samples and shifting dst_x.
        src_x = (np.arange(n_frames) + 0.5) / n_frames
        dst_x = (np.arange(out_len) + 0.5) / out_len
        shape = np.interp(dst_x, src_x, frame_rms / ref)
        gain = np.clip(np.power(shape, self.strength), self.min_gain, self.max_gain)
        out_f = out_i16.astype(np.float32)
        return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)
