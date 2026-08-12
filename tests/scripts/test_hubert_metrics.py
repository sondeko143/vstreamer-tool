"""Tests for the pure functions of scripts/hubert_metrics (no assets, no GPU)."""

import warnings

import numpy as np
import pytest

from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr


def test_feature_cosine_identical_is_one():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert feature_cosine(a, a) == pytest.approx(1.0)


def test_feature_cosine_orthogonal_is_zero():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert feature_cosine(a, b) == pytest.approx(0.0, abs=1e-9)


def test_feature_cosine_opposite_is_minus_one():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    assert feature_cosine(a, -a) == pytest.approx(-1.0)


def test_feature_cosine_rejects_shape_mismatch():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        feature_cosine(a, b)


def test_feature_cosine_both_zero_frames_match():
    z = np.zeros((3, 4), dtype=np.float32)
    assert feature_cosine(z, z) == pytest.approx(1.0)


def test_feature_cosine_one_side_all_zero_is_not_a_match():
    """One side being all zeros is a mismatch. Excluding zero-norm frames wholesale would
    return 1.0."""
    a = np.ones((3, 4), dtype=np.float32)
    z = np.zeros((3, 4), dtype=np.float32)
    assert feature_cosine(a, z) == pytest.approx(0.0)


def test_feature_max_abs_diff():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.5, 2.0]], dtype=np.float32)
    assert feature_max_abs_diff(a, b) == pytest.approx(0.5)


def test_waveform_correlation_identical_and_inverted():
    x = np.sin(np.linspace(0.0, 10.0, 512)).astype(np.float32)
    assert waveform_correlation(x, x) == pytest.approx(1.0)
    assert waveform_correlation(x, -x) == pytest.approx(-1.0)


def test_waveform_snr_identical_is_inf_and_emits_no_warning():
    """An exact match is infinite SNR. No division happens, so numpy emits no warning."""
    x = np.sin(np.linspace(0.0, 10.0, 4096))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            assert waveform_snr(x, x) == float("inf")


def test_waveform_snr_decreases_with_noise():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 10.0, 4096))
    small = x + 1e-4 * rng.standard_normal(x.size)
    large = x + 1e-2 * rng.standard_normal(x.size)
    assert waveform_snr(x, small) > waveform_snr(x, large)
    assert np.isfinite(waveform_snr(x, large))


def test_waveform_snr_all_silent_is_inf():
    z = np.zeros(2048, dtype=np.float32)
    assert waveform_snr(z, z) == float("inf")


def test_waveform_snr_silent_reference_with_corrupted_test_is_minus_inf():
    """A silent reference with energy on the test side must never be reported as
    "perfect"."""
    ref = np.zeros(2048)
    test = np.zeros(2048)
    test[1024:] = 0.1
    assert waveform_snr(ref, test) == float("-inf")


def test_waveform_snr_catches_corruption_at_any_offset():
    """With no framing, corruption is always detected, even in a trailing partial frame.

    The overall SNR discards no tail, so even when the length is not a multiple of
    frame_len, a case where only the trailing partial frame is corrupt is detected rather
    than misreported as inf (perfect).
    """
    ref = np.sin(np.linspace(0.0, 40.0, 2548))
    test = ref.copy()
    test[2048:] += 5.0  # corrupt only the tail past the last frame_len multiple
    result = waveform_snr(ref, test)
    assert np.isfinite(result), f"tail corruption was hidden: {result}"
    assert result < 40.0  # must not pass the gate


def test_waveform_snr_rejects_non_finite_input():
    """NaN / inf is corruption. Fail explicitly rather than misreporting `inf`
    (perfect)."""
    x = np.sin(np.linspace(0.0, 10.0, 2048))
    with pytest.raises(ValueError, match="finite"):
        waveform_snr(x, np.full_like(x, np.nan))
    with pytest.raises(ValueError, match="finite"):
        waveform_snr(np.full_like(x, np.inf), x)


def test_waveform_snr_rejects_energy_overflow():
    """When the energy sum overflows float64, fail instead of returning inf."""
    x = np.full(16, 1e200)
    test = x.copy()
    test[0] = 0.0
    with pytest.raises(ValueError, match="overflow"):
        waveform_snr(x, test)


def test_fp16_thresholds_are_looser_than_fp32_but_still_tight():
    """The fp16 gate is looser than fp32, but not meaninglessly loose.

    `1e-1` / `0.999` are **hard limits that do not move**. If 10x the measured value
    exceeds them, the fp16 export is broken, so suspect the export rather than the
    threshold.
    """
    from scripts.hubert_metrics import COSINE_MIN
    from scripts.hubert_metrics import COSINE_MIN_FP16
    from scripts.hubert_metrics import MAX_ABS_MAX
    from scripts.hubert_metrics import MAX_ABS_MAX_FP16

    assert MAX_ABS_MAX_FP16 > MAX_ABS_MAX
    assert MAX_ABS_MAX_FP16 <= 1e-1
    assert COSINE_MIN_FP16 <= COSINE_MIN
    assert COSINE_MIN_FP16 >= 0.999
