import numpy as np
import pytest

from vspeech.worker.vc import _input_as_float32_16k
from vspeech.worker.vc import apply_input_envelope
from vspeech.worker.vc import check_cuda_provider


def test_check_cuda_provider_passes_when_cuda_present():
    # Should not raise.
    check_cuda_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_check_cuda_provider_raises_when_only_cpu():
    with pytest.raises(RuntimeError) as excinfo:
        check_cuda_provider(["CPUExecutionProvider"])
    # Message should be actionable about the missing GPU runtime.
    assert "CUDAExecutionProvider" in str(excinfo.value)


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def test_envelope_strength_zero_returns_output_unchanged():
    rng = np.random.default_rng(0)
    out = rng.integers(-3000, 3000, 8000).astype(np.int16)
    inp = _pcm(rng.integers(-3000, 3000, 4000))
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 0.0, 0.1, 4.0)
    np.testing.assert_array_equal(res, out)


def test_silent_input_returns_output_unchanged():
    out = (np.tile([1, -1], 4000) * 8000).astype(np.int16)
    inp = _pcm(np.zeros(4000))
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    np.testing.assert_array_equal(res, out)


def test_output_length_is_preserved():
    inp = _pcm(np.tile([1, -1], 2000) * 5000)
    out = (np.tile([1, -1], 5000) * 3000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    assert res.shape[0] == out.shape[0]


def test_constant_input_preserves_output_level():
    # Flat-amplitude input and flat-amplitude output -> gain ~1 everywhere.
    inp = _pcm(np.tile([1, -1], 2000) * 5000)
    out = (np.tile([1, -1], 4000) * 3000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    np.testing.assert_allclose(res.astype(np.float32), out.astype(np.float32), atol=1.0)


def test_ramp_input_gain_follows_input_dynamics():
    # Input amplitude grows over the clip; flat output -> louder toward the end.
    n = 8000
    env = np.linspace(0.05, 1.0, n)
    carrier = np.tile([1.0, -1.0], n // 2)
    inp = _pcm(env * carrier * 20000)
    out = (np.tile([1, -1], n // 2) * 8000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.01, 10.0)
    first = np.abs(res[:1000].astype(np.float32)).mean()
    last = np.abs(res[-1000:].astype(np.float32)).mean()
    assert last > first * 3


def test_gain_transition_is_smooth_not_stepped():
    # Quiet first half, loud second half: the boundary must ramp, not jump.
    n = 8000
    amp = np.where(np.arange(n) < n // 2, 2000.0, 20000.0)
    carrier = np.tile([1.0, -1.0], n // 2)
    inp = _pcm(amp * carrier)
    out = (np.tile([1, -1], n // 2) * 8000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.05, 10.0)
    gain = np.abs(res.astype(np.float32)) / (np.abs(out.astype(np.float32)) + 1e-9)
    assert np.abs(np.diff(gain)).max() < 0.5


def test_quiet_output_region_is_not_amplified():
    # Regression: the gain must NOT be boosted where the RVC output is quiet.
    # A steady input leaves the gain ~1, so a near-silent (noise-floor) output
    # region stays quiet -- no "noise in the silent sections". (The earlier
    # divide-by-output form saturated the gain to max_gain here, pumping the
    # noise floor up ~4x.)
    n = 8000
    rng = np.random.default_rng(0)
    out = np.empty(n, dtype=np.float64)
    out[: n // 2] = np.sin(2 * np.pi * 200 * np.arange(n // 2) / 16000) * 8000
    out[n // 2 :] = rng.normal(0, 60, n // 2)  # quiet noise floor
    out = out.astype(np.int16)
    inp = _pcm(np.sin(2 * np.pi * 180 * np.arange(n) / 16000) * 10000)  # steady

    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)

    noise_in = np.abs(out[n // 2 :].astype(np.float64)).mean()
    noise_out = np.abs(res[n // 2 :].astype(np.float64)).mean()
    assert noise_out < noise_in * 1.5


def test_does_not_compress_output_dynamics():
    # Regression: with a steady input the gain must not fight the output's own
    # envelope (that anti-correlation is what made max_gain>1 sound like a
    # compressor). The earlier divide-by-output form gave a strongly negative
    # correlation; the shape-only form keeps it near zero.
    n = 8000
    out_env = np.linspace(1.0, 0.2, n)  # output loud -> quiet
    out = (out_env * np.tile([1.0, -1.0], n // 2) * 9000).astype(np.int16)
    inp = _pcm(np.sin(2 * np.pi * 180 * np.arange(n) / 16000) * 10000)  # steady

    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)

    gain = np.abs(res.astype(np.float64)) / (np.abs(out.astype(np.float64)) + 1e-9)
    corr = np.corrcoef(gain, np.abs(out.astype(np.float64)))[0, 1]
    assert corr > -0.2


def test_loud_output_is_not_clipped_with_default_config():
    # Regression: the RVC output is already a full-level int16 signal, so the
    # gain must not boost it past int16 range. With the default max_gain the
    # gain stays <= 1 (a duck), so a near-full-scale output region cannot be
    # amplified into hard clipping ("音割れ" on loud parts).
    from vspeech.config import VcConfig

    n = 8000
    t = np.arange(n)
    out = np.empty(n)
    out[: n // 2] = np.sin(2 * np.pi * 200 * t[: n // 2] / 16000) * 30000  # loud
    out[n // 2 :] = np.sin(2 * np.pi * 200 * t[n // 2 :] / 16000) * 6000
    out = out.astype(np.int16)
    inp_env = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 0.3)])
    inp = _pcm(inp_env * np.sin(2 * np.pi * 180 * t / 16000) * 20000)

    cfg = VcConfig()
    res = apply_input_envelope(
        out,
        inp,
        2,
        16000,
        cfg.volume_adjust_window_ms,
        cfg.envelope_strength,
        cfg.min_gain,
        cfg.max_gain,
    )
    assert int(np.sum(np.abs(res.astype(np.int32)) >= 32767)) == 0
    # still follows input: loud half louder than the ducked quiet half.
    assert np.abs(res[: n // 2]).mean() > np.abs(res[n // 2 :]).mean()


def test_vc_config_envelope_defaults():
    from vspeech.config import VcConfig

    cfg = VcConfig()
    assert cfg.adjust_output_vol_to_input_voice is True
    assert cfg.envelope_strength == 1.0
    assert cfg.min_gain == 0.1
    assert cfg.max_gain == 1.0
    assert cfg.volume_adjust_window_ms == 25.0


def test_input_as_float32_16k_scales_int16_without_resample():
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    res = _input_as_float32_16k(samples.tobytes(), 2, 16000)
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, [0.0, 0.5, -0.5, 32767 / 32768], atol=1e-6)
