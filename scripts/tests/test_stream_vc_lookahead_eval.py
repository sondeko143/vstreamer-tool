import wave

import numpy as np

from scripts.stream_vc_lookahead_eval import align_offset
from scripts.stream_vc_lookahead_eval import frame_energy
from scripts.stream_vc_lookahead_eval import spectral_distance
from scripts.stream_vc_lookahead_eval import write_wav


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


def test_spectral_distance_distinguishes_mean_from_p95():
    """Per-frame distances that vary widely, so a bug that returned `max` instead of the
    95th percentile, or indexed the wrong element, would be caught (the two existing
    numeric tests above both use a constant per-frame distance, so mean == p95 there and
    neither bug would show up).

    One mel bin, so the L2 distance over mels collapses to the plain per-frame
    difference, and 21 frames put the 95th percentile (linear interpolation, numpy's
    default) exactly on the 20th smallest value with no interpolation -- so the expected
    figure can be read off `diffs` by hand, independent of `np.percentile`.
    """
    n_frames = 21
    diffs = np.arange(n_frames, dtype=np.float64)  # per-frame distance, 0..20
    ref = np.zeros((1, n_frames))
    test = ref + diffs[np.newaxis, :]
    mean, p95 = spectral_distance(ref, test)
    assert mean != p95
    assert abs(mean - 10.0) < 1e-9  # arithmetic mean of 0..20
    assert abs(p95 - 19.0) < 1e-9  # the 20th smallest value (0-indexed: diffs[19])


def test_write_wav_round_trips_samples(tmp_path):
    samples = np.array([0, 1, -1, 32767, -32768, 12345], dtype=np.int16)
    rate = 22050
    path = tmp_path / "out.wav"
    write_wav(path, samples, rate)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == rate
        assert w.getnframes() == samples.shape[0]
        frames = w.readframes(w.getnframes())
    got = np.frombuffer(frames, dtype=np.int16)
    np.testing.assert_array_equal(got, samples)
