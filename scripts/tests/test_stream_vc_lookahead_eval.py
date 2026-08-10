import numpy as np

from scripts.stream_vc_lookahead_eval import align_offset
from scripts.stream_vc_lookahead_eval import frame_energy
from scripts.stream_vc_lookahead_eval import spectral_distance


def _speechlike(n: int, seed: int = 0) -> np.ndarray:
    """A signal with a speech-like envelope, so the coarse search has something to lock
    onto."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n) * 0.2
    env = np.abs(np.sin(np.linspace(0.0, 12.0 * np.pi, n)))
    return (base * env).astype(np.float32)


def test_frame_energy_is_per_hop_rms():
    x = np.concatenate([np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)])
    got = frame_energy(x, hop=4)
    assert got.shape == (2,)
    assert got[0] == 0.0
    assert abs(got[1] - 1.0) < 1e-12


def test_frame_energy_of_a_short_signal_is_empty():
    assert frame_energy(np.ones(3, dtype=np.float32), hop=4).shape == (0,)


def test_align_offset_recovers_a_known_shift():
    ref = _speechlike(60000)
    shift = 3111
    test = np.concatenate([np.zeros(shift, dtype=np.float32), ref])
    assert align_offset(ref, test, hint=3000) == shift


def test_align_offset_recovers_a_shift_far_from_the_hint():
    ref = _speechlike(60000, seed=2)
    shift = 9000
    test = np.concatenate([np.zeros(shift, dtype=np.float32), ref])
    # the coarse stage finds it even when the hint is a whole block off
    assert align_offset(ref, test, hint=2400) == shift


def test_spectral_distance_is_zero_for_identical_input():
    rng = np.random.default_rng(1)
    lm = rng.standard_normal((80, 300)) * 2.0 - 3.0
    assert spectral_distance(lm, lm) == (0.0, 0.0)


def test_spectral_distance_equals_a_uniform_offset():
    lm = np.zeros((80, 100))
    mean, p95 = spectral_distance(lm, lm + 3.0)
    assert abs(mean - 3.0) < 1e-9
    assert abs(p95 - 3.0) < 1e-9


def test_spectral_distance_ignores_frames_below_the_floor():
    lm = np.full((80, 10), -100.0)
    lm[:, 0] = 0.0  # the only frame with energy
    test = lm.copy()
    test[:, 1:] = 50.0  # wreck the silent frames only
    mean, _ = spectral_distance(lm, test, floor_db=-40.0)
    assert mean == 0.0
