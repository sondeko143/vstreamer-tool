import numpy as np

from vspeech.stream_vc.envelope import StreamingEnvelope


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


def test_first_block_is_near_unity():
    # cold start: ref := block mean, flat block -> shape 1 -> gain 1 -> unchanged.
    env = _env()
    out = _out()
    got = env.apply(out.copy(), _block(0.2))
    assert np.allclose(got, out, atol=1)


def test_quiet_block_after_loud_is_ducked():
    env = _env()
    for _ in range(20):  # establish ema at the loud level
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.03))  # a decay-tail block, 10x quieter
    assert got.max() < 10000 * 0.5  # ducked well below the loud level


def test_steady_level_stays_near_unity():
    env = _env()
    for _ in range(10):
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.3))
    assert np.allclose(got, _out(), atol=20)  # duck-only, steady -> ~unity


def test_within_block_attack_ramp_is_ducked_at_the_quiet_lead_in():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3))  # ref at speech level
    # a block that is quiet in its first half, loud in its second (an onset)
    onset = np.concatenate([_block(0.02, 1280), _block(0.3, 1280)])
    got = env.apply(_out(), onset)
    assert got[0] < got[-1]  # gain rises across the block = attack ramp recovered


def test_min_gain_clamps_the_duck():
    env = _env(min_gain=0.25)
    for _ in range(20):
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.0001))  # near-silent block
    assert got.max() >= 10000 * 0.25 - 2  # not ducked below min_gain


def test_reset_clears_ema():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3))
    env.reset()
    got = env.apply(
        _out().copy(), _block(0.03)
    )  # cold start again -> ~unity, not ducked
    assert np.allclose(got, _out(), atol=1)


def test_empty_and_silent_passthrough():
    env = _env()
    out = _out()
    assert np.array_equal(env.apply(out.copy(), np.zeros(0, dtype=np.float32)), out)
    zero_env = _env(strength=0.0)
    assert np.array_equal(zero_env.apply(out.copy(), _block(0.3)), out)
