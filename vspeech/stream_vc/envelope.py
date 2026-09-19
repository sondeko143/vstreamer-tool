"""Input envelope shaping for streaming VC (ADR-0057, ADR-0065, ADR-0092, ADR-0093).

Normalizes the input block's relative loudness envelope against a rolling EMA of the
mean input RMS and applies it to the output block as
`clip(shape^strength, min_gain, max_gain)`. Same idea as the batch
apply_input_envelope (worker/vc.py), but this streaming version replaces the reference
"mean over the whole utterance" with a rolling EMA (only one block is available at a
time) that follows speech only (ADR-0093).

The gain law is log-linear -- `gain_dB = strength * shape_dB` -- so `strength` is a
slope and its **sign is the direction** (ADR-0092):

- `strength > 0` **duck**: the input's quiet parts pull the output down with them, which
  is envelope *following*. `max_gain` 1.0 keeps it duck-only (ADR-0018).
- `strength == 0` identity (the caller may still hold an instance).
- `strength < 0` **lift**: the quiet parts are raised towards the reference and the loud
  parts pushed down, i.e. compression. Needs `max_gain > 1` to raise anything, and that
  bound doubles as the noise-amplification guard.

What makes the lift direction sound like the input rather than like an inverted mix is
that the RVC output tracks the input closely at this timescale but not at all across
sustained level changes -- measured on real speech, `out_dB = 0.955 * in_dB` at 25ms
frames (corr +0.805), against a slope of 0.065 over a settled 36dB gain staircase. The
EMA reference removes exactly the scale the model already normalizes away.

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

# Lower bound on the relative shape, as a ratio to the reference level. 1e-6 is -120dB,
# below int16's whole dynamic range (~96dB), so this can only bind on a frame that is
# already digital silence. See where it is applied for why the bound exists at all.
_SHAPE_FLOOR = 1e-6


class StreamingEnvelope:
    """Input envelope shaping against a rolling-EMA reference (ADR-0057/0065/0092/0093).

    The state is the reference level `_ema_level` (a scalar) plus the shape history
    (`_history`), which the seam handover needs. `apply()` multiplies the output block by
    the current input block's relative loudness envelope and updates both for the next
    block. The direction (duck or lift) is the sign of `strength`; see the module
    docstring.
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
        # (ADR-0072) pushes it past that, and then the head of the emit carries audio
        # shaped two or more blocks ago. The length is derived per call from the delay.
        self._history: list[tuple[NDArray[np.float64], int]] = []

    def reset(self) -> None:
        """Return the reference level and the seam handover to uninitialized (called by
        the runner on pause/resume and on a capture reopen).

        So that a stale reference level does not oddly shape the next block after
        real time has jumped, force a cold start again -- initializing from the block mean
        of the next apply that is allowed to move the reference at all (ADR-0093), so a
        resume into silence cannot pin the fresh reference to the noise floor. Until then
        the blocks pass through unshaped. The shape history is dropped too: it describes
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
        update_reference: bool,
    ) -> NDArray[np.int16]:
        """Shape the output block out_i16 by the relative loudness envelope of the input
        block in_block (16k float32), against the rolling EMA reference.

        The reference is the **past** EMA (history). On a cold start, or right after
        reset, it is initialized from the current block's mean (so the first block is not
        shaped unnaturally). The reference is updated before returning, so the next block
        uses an EMA that already includes this one.

        `delay_samples` is `StreamingVc.emit_delay_samples`: how many samples before the
        start of the input block the emit's content begins (at the output rate). The sound
        carried by emit sample j sits at position `j - delay_samples` relative to the
        start of the input block, so the shape is laid on that shifted grid -- identical
        to the VAD gate's mask overlay (gate.py, ADR-0059).

        `update_reference` says whether this block may move the reference level, and both
        the cold start and the EMA update obey it (ADR-0093). The caller decides the
        policy -- `runner.reference_may_follow` reads it off the VAD gate's verdict -- so
        this class never has to know what a VAD is. It has no default on purpose, the
        same discipline as `delay_samples` (ADR-0065): a default would let a caller drop
        the argument and silently go back to a reference that follows silence.

        **Known characteristic (ADR-0057), reduced but not removed by ADR-0093:** the
        reference still lags speech level at the start of a phrase, because it has to
        climb there with `envelope_ema_ms`. The first frames are therefore judged louder
        against it than they are. Measured on a real recording at 13% speech duty, this
        leaves 12.3% of speech frames on a rail in the lift direction (it was 38.6% with
        the ungated 4000ms reference). Inter-word dips and decay tails inside continuous
        speech are shaped correctly, because there the reference does sit at speech
        level (3.0% railed). The phrase onset itself is the VAD gate's job.
        """
        import numpy as np

        out_len = int(out_i16.shape[0])
        if out_len == 0 or in_block.shape[0] == 0 or self.strength == 0.0:
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
        # The cold start is deferred to the first block the caller lets set the reference.
        # Seeding it from silence pins it to the noise floor, and the first phrase after
        # that reads as tens of dB above the reference -- which the lift direction turns
        # into a blanket attenuation of the whole phrase rather than shaping within it.
        ref: float | None
        if update_reference:
            if self._ema_level is None:
                self._ema_level = block_mean
            ref = self._ema_level
            self._ema_level = self._alpha * block_mean + (1.0 - self._alpha) * ref
        else:
            ref = self._ema_level
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
        # No reference to shape against yet (the caller has not let one be established
        # since construction or `reset()`), or one that is effectively digital silence
        # (e.g. pure silence right after init) -> pass through.
        if ref is None or ref < 1e-8:
            # This block went out at unity, so hand unity over: leaving the older shape in
            # place would make the next block step off a value that was never applied.
            self._history.append((np.ones(n_frames, dtype=np.float64), out_len))
            del self._history[:-need]
            return out_i16
        # The relative shape (relative to the reference, not mean~1), linearly
        # interpolated onto the emit's sample grid.
        #
        # Floored, because a digitally silent frame makes `shape ** strength` raise
        # `0 ** negative` = inf in the lift direction, and numpy warns while doing it --
        # once per block for as long as the input is silent. `_SHAPE_FLOOR` sits below
        # int16's whole dynamic range, so the floor can only ever bind on a frame that
        # is already silent, and the duck direction is unaffected (both 0 and the floor
        # clamp to `min_gain`).
        shape_now = np.maximum(frame_rms / ref, _SHAPE_FLOOR)
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
