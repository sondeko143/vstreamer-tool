import warnings

import numpy as np

from vspeech.stream_vc.envelope import StreamingEnvelope

# The emit delay published by StreamingVc at the validated geometry (500/160/25/5 at a
# 40kHz model) = 50ms = 2000 output samples. The shape has to be laid on the emit shifted
# by this, exactly as the VAD gate lays its mask (ADR-0059).
_DELAY = 2000
_AMP = 30000


def _env(**kw):
    base = dict(
        strength=1.0,
        min_gain=0.1,
        max_gain=1.0,
        window_ms=25.0,
        ema_ms=2000.0,
        block_ms=160.0,
    )
    base.update(kw)
    return StreamingEnvelope(**base)


def _block(level, n=2560):  # 160ms @ 16k
    return np.full(n, level, dtype=np.float32)


def _out(n=7680):  # 160ms @ 48k, full-scale-ish
    return np.full(n, 10000, dtype=np.int16)


def _gain_curves(env, blocks, out_len=6400, delay=_DELAY):
    """Per-block gain curves.

    The "output" is a constant, so the returned samples ARE the applied gain and the curve
    can be read sample by sample (including right at the block seam).
    """
    const = np.full(out_len, _AMP, dtype=np.int16)
    return [env.apply(const, b, delay, True).astype(np.float64) / _AMP for b in blocks]


def test_first_block_is_near_unity():
    # cold start: ref := block mean, flat block -> shape 1 -> gain 1 -> unchanged.
    env = _env()
    out = _out()
    got = env.apply(out.copy(), _block(0.2), 0, True)
    assert np.allclose(got, out, atol=1)


def test_quiet_block_after_loud_is_ducked():
    env = _env()
    for _ in range(20):  # establish ema at the loud level
        env.apply(_out(), _block(0.3), 0, True)
    got = env.apply(_out(), _block(0.03), 0, True)  # a decay-tail block, 10x quieter
    # Read the tail, not the head: the head hands the gain over from the previous block
    # (it is a ramp now, not a step -- see test_gain_is_continuous_across_the_block_seam).
    assert got[len(got) // 2 :].max() < 10000 * 0.5  # ducked well below the loud level


def test_steady_level_stays_near_unity():
    env = _env()
    for _ in range(10):
        env.apply(_out(), _block(0.3), 0, True)
    got = env.apply(_out(), _block(0.3), 0, True)
    assert np.allclose(got, _out(), atol=20)  # duck-only, steady -> ~unity


def test_within_block_attack_ramp_is_ducked_at_the_quiet_lead_in():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3), 0, True)  # ref at speech level
    # a block that is quiet in its first half, loud in its second (an onset)
    onset = np.concatenate([_block(0.02, 1280), _block(0.3, 1280)])
    got = env.apply(_out(), onset, 0, True)
    assert got[0] < got[-1]  # gain rises across the block = attack ramp recovered


def test_min_gain_clamps_the_duck():
    env = _env(min_gain=0.25)
    for _ in range(20):
        env.apply(_out(), _block(0.3), 0, True)
    got = env.apply(_out(), _block(0.0001), 0, True)  # near-silent block
    assert got.max() >= 10000 * 0.25 - 2  # not ducked below min_gain


def test_reset_clears_the_ema_and_the_previous_block_shape():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3), 0, True)
    env.reset()
    # Cold start again: ref := block mean AND the handover seeds at unity, so a flat block
    # comes out unchanged instead of ramping down out of the stale loud reference.
    got = env.apply(_out().copy(), _block(0.03), 0, True)
    assert np.allclose(got, _out(), atol=1)


def test_empty_and_silent_passthrough():
    env = _env()
    out = _out()
    assert np.array_equal(
        env.apply(out.copy(), np.zeros(0, dtype=np.float32), 0, True), out
    )
    zero_env = _env(strength=0.0)
    assert np.array_equal(zero_env.apply(out.copy(), _block(0.3), 0, True), out)


# --- cross-block continuity and the emit-delay correction -------------------


def test_gain_is_continuous_across_the_block_seam():
    """The gain curve must hand over from the previous block, with no step at the seam.

    Regression: the shape used to be interpolated on a per-block normalized axis with no
    carry, so the first sample of a block jumped straight from the previous block's last
    gain to this block's first frame. Measured at the real rig's settings (min 0.4 /
    max 0.9, ema 4000): 33 of 124 seams stepped by more than 0.05 and the worst was the
    full rail-to-rail 0.5 = **+7dB in one sample**, i.e. a click at the block rate. The
    VAD gate carries `_prev_gains` for exactly this reason (gate.py).
    """
    env = _env(min_gain=0.4, max_gain=0.9, ema_ms=4000.0)
    levels = [0.30, 0.30, 0.002, 0.002, 0.35, 0.02, 0.30, 0.002, 0.28, 0.30]
    curves = _gain_curves(env, [_block(v) for v in levels])
    # The largest gain change between two adjacent samples *inside* a block. The handover
    # ramp is itself interior to the following block, so this is the honest yardstick:
    # the seam must be no more special than any other sample.
    interior = max(float(np.abs(np.diff(c)).max()) for c in curves)
    seams = [abs(curves[k + 1][0] - curves[k][-1]) for k in range(len(curves) - 1)]
    # 1e-4 covers the int16 quantization of the constant probe (1 LSB / _AMP).
    assert max(seams) <= interior + 1e-4, (
        f"seam step {max(seams):.5f} exceeds the largest interior step {interior:.5f}"
    )


def test_the_shape_is_laid_on_the_emit_shifted_by_the_emit_delay():
    """The emit lags the input by delay_samples (emit sample j carries the audio from
    block-relative time j - delay_samples), so an input feature lands that much **later**
    on the emit's grid -- the same arithmetic as the VAD gate's mask."""
    onset = np.concatenate([_block(0.002, 1280), _block(0.30, 1280)])
    blocks = [_block(0.30), _block(0.30), onset]

    def rising_edge(delay: int) -> int:
        """Where the duck of the quiet lead-in recovers (the trough's trailing edge).

        Not the *first* 0.5 crossing: the block opens high because it hands the gain over
        from the loud previous block, so index 0 already sits above the level.
        """
        curve = _gain_curves(_env(), blocks, delay=delay)[-1]
        trough = int(np.argmin(curve))
        return trough + int(np.nonzero(curve[trough:] >= 0.5)[0][0])

    assert rising_edge(1600) - rising_edge(0) == 1600


def test_a_dropped_carry_does_not_shift_the_shape_of_the_next_block():
    """A block whose reference is digital silence passes through, and the block after it
    still hands over (from unity) rather than cold-starting clamped."""
    env = _env(min_gain=0.4, max_gain=0.9)
    curves = _gain_curves(env, [_block(0.0), _block(0.0), _block(0.30)])
    assert np.allclose(curves[0], 1.0, atol=1e-4)  # silence -> pass through
    seam = abs(curves[2][0] - curves[1][-1])
    interior = float(np.abs(np.diff(curves[2])).max())
    assert seam <= interior + 1e-4


def test_shape_reaches_two_blocks_back_when_the_delay_exceeds_one_emit():
    """With a delay past one emit length, the head carries the shape from two blocks back.

    With only one block of history the head falls left of the oldest frame centre and
    clamps to the previous block's first value. Lookahead puts the geometry in exactly
    that region.
    """
    env = _env(strength=1.0, min_gain=0.0, max_gain=1.0)
    out_len = 6400
    delay = 9000  # past one emit length (6400)
    loud = _block(0.2, n=2560)
    quiet = _block(0.002, n=2560)
    ones = np.full(out_len, 10000, dtype=np.int16)
    env.apply(ones.copy(), loud, delay, True)
    env.apply(ones.copy(), quiet, delay, True)
    got = env.apply(ones.copy(), quiet, delay, True)
    g = got.astype(np.float64) / 10000.0
    # the head is audio from two blocks back (loud), so it is not ducked
    assert g[0] > 0.5
    # the tail has come down to the quiet level
    assert g[-1] < 0.2


def test_gain_is_continuous_across_the_seam_with_a_long_delay():
    """With a delay past one emit length, the gain still does not step at a block boundary
    (extends ADR-0065's guarantee across the history generalisation)."""
    env = _env(strength=1.0, min_gain=0.0, max_gain=1.0)
    out_len, delay = 6400, 9000
    ones = np.full(out_len, 10000, dtype=np.int16)
    curve = [
        env.apply(ones.copy(), _block(level, n=2560), delay, True).astype(np.float64)
        / 10000.0
        for level in (0.02, 0.02, 0.3, 0.3, 0.05, 0.02)
    ]
    full = np.concatenate(curve)
    assert float(np.abs(np.diff(full)).max()) < 0.02


# --- the lift direction (negative strength) ---------------------------------


def test_negative_strength_lifts_a_block_quieter_than_the_reference():
    # gain = shape ** strength, so a negative strength turns shape < 1 into gain > 1:
    # the quiet parts are raised towards the reference instead of ducked away from it.
    env = _env(strength=-0.3, min_gain=0.3, max_gain=2.0)
    for _ in range(20):  # establish the reference at the loud level
        env.apply(_out(), _block(0.3), 0, True)
    got = env.apply(_out(), _block(0.03), 0, True)  # 10x quieter than the reference
    # The tail, not the head: the head is the handover ramp from the previous block.
    assert got[len(got) // 2 :].min() > 10000


def test_negative_strength_attenuates_a_block_louder_than_the_reference():
    env = _env(strength=-0.3, min_gain=0.3, max_gain=2.0)
    for _ in range(20):  # establish the reference at the quiet level
        env.apply(_out(), _block(0.03), 0, True)
    got = env.apply(_out(), _block(0.3), 0, True)  # 10x louder than the reference
    assert got[len(got) // 2 :].max() < 10000


def test_a_digitally_silent_frame_does_not_make_the_lift_gain_infinite():
    # shape ** strength is 0 ** negative = inf for a frame of digital silence, and numpy
    # raises a divide-by-zero RuntimeWarning doing it. The clip hides the value, but the
    # warning would fire once per block for as long as the input is silent.
    env = _env(strength=-0.3, min_gain=0.3, max_gain=2.0)
    for _ in range(20):
        env.apply(_out(), _block(0.3), 0, True)
    half_silent = np.concatenate([_block(0.3, 1280), _block(0.0, 1280)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = env.apply(_out(), half_silent, 0, True)
    assert got[-1] == 20000  # the silent tail rails at max_gain, not at inf


# --- the reference only follows speech ---------------------------------------


def test_a_block_the_caller_marks_non_speech_does_not_move_the_reference():
    # Silence between phrases would otherwise drag the reference down towards the noise
    # floor, and the next phrase then reads as far louder than the reference -- which in
    # the lift direction attenuates the whole phrase instead of shaping it.
    env = _env(strength=-0.3, min_gain=0.3, max_gain=2.0)
    for _ in range(20):
        env.apply(_out(), _block(0.3), 0, True)
    for _ in range(25):  # 4s of silence, with the VAD gate closed
        env.apply(_out(), _block(0.003), 0, False)
    got = env.apply(_out(), _block(0.3), 0, True)  # the next phrase, at the same level
    # Still read as "at the reference" -> gain ~1, i.e. neither lifted nor attenuated.
    assert abs(int(got[len(got) // 2 :].mean()) - 10000) < 200


def test_a_cold_start_in_silence_does_not_pin_the_reference_to_the_noise_floor():
    # Starting (or resuming) mid-silence must not seed the reference from the noise
    # floor: the first phrase after it would then read as far above the reference and be
    # attenuated as a whole rather than shaped.
    env = _env(strength=-0.3, min_gain=0.3, max_gain=2.0)
    for _ in range(10):  # silence before the first phrase, with the gate closed
        env.apply(_out(), _block(0.003), 0, False)
    got = env.apply(_out(), _block(0.3), 0, True)  # the first phrase
    assert abs(int(got[len(got) // 2 :].mean()) - 10000) < 200
