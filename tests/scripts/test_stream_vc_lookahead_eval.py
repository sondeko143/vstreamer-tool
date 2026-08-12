import wave

import numpy as np

from scripts.stream_vc_lookahead_eval import _mel_filterbank
from scripts.stream_vc_lookahead_eval import align_offset
from scripts.stream_vc_lookahead_eval import frame_energy
from scripts.stream_vc_lookahead_eval import log_mel
from scripts.stream_vc_lookahead_eval import right_context_ms
from scripts.stream_vc_lookahead_eval import spectral_distance
from scripts.stream_vc_lookahead_eval import warmup_skip_samples
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


def test_log_mel_shape_and_finite_on_a_short_input():
    """A 200-sample input is well under one FFT frame (1024), but the reflect-padding
    (512 samples each side) still yields exactly one frame, and every value is finite."""
    x = (0.3 * np.sin(2 * np.pi * 220.0 * np.arange(200) / 16000) * 20000).astype(
        np.int16
    )
    out = log_mel(x, 16000)
    assert out.shape == (80, 1)
    assert np.all(np.isfinite(out))


def test_log_mel_handles_a_length_one_input():
    """The shortest possible input (a single sample) does not raise or divide by zero
    (guards the padding/framing arithmetic at its own boundary)."""
    out = log_mel(np.array([12345], dtype=np.int16), 16000)
    assert out.shape == (80, 1)
    assert np.all(np.isfinite(out))


def test_log_mel_of_silence_is_a_constant_floor():
    """All-zero input -> every mel bin of every frame reads the 1e-10 floor added inside
    log_mel (10*log10(1e-10) = -100 dB), not NaN/-inf from log(0)."""
    out = log_mel(np.zeros(4000, dtype=np.int16), 16000)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 10.0 * np.log10(1e-10))


def test_mel_filterbank_has_a_dead_channel_only_at_48k():
    """Documents the one structural gap from torchaudio's filterbank noted in
    log_mel's docstring: at this n_fft=1024/n_mels=80 shape, floor-to-bin rounding
    leaves exactly one all-zero channel at 48kHz (the operative rate for this file's
    real callers) and none at 16kHz. torchaudio's filterbank has no dead channel here;
    this one is harmless for the ranking use in log_mel (a zero channel contributes the
    same term to both sides of every spectral_distance comparison), but it is real."""
    fb_16k = _mel_filterbank(1024, 16000, 80)
    fb_48k = _mel_filterbank(1024, 48000, 80)
    assert np.count_nonzero(~fb_16k.any(axis=1)) == 0
    assert np.count_nonzero(~fb_48k.any(axis=1)) == 1


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


def test_load_wav_16k_reads_what_write_wav_produced(tmp_path):
    """The eval writes wavs and the RTF harness reads them; pin that they agree.

    This broke in practice: torchaudio 2.11 (the terminal release pinned by ADR-0069)
    routes `load` through torchcodec, which is not a dependency, so the reader raised
    ImportError on a real run. Reading is stdlib now, and this keeps it that way.
    """
    from scripts.stream_vc_rtf import load_wav_16k

    samples = (np.sin(np.linspace(0.0, 40.0 * np.pi, 16000)) * 12000).astype(np.int16)
    path = tmp_path / "roundtrip.wav"
    write_wav(path, samples, 16000)
    got = load_wav_16k(path)
    assert got.dtype == np.float32
    assert got.shape == samples.shape
    np.testing.assert_allclose(got, samples.astype(np.float32) / 32768.0)


def test_load_wav_16k_rejects_non_16bit_wav(tmp_path):
    """An 8-bit wav must fail loudly rather than be misread as int16 garbage."""
    from scripts.stream_vc_rtf import load_wav_16k

    path = tmp_path / "eight_bit.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(bytes(range(256)))
    try:
        load_wav_16k(path)
    except ValueError as e:
        assert "sampwidth=1" in str(e)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected ValueError for an 8-bit wav")


def test_warmup_skip_samples_matches_the_context_plus_block_span():
    # 500 + 0 + 160 = 660ms at 16kHz = 10560 samples exactly, no rounding involved.
    assert (
        warmup_skip_samples(
            context_ms=500.0, lookahead_ms=0.0, block_ms=160.0, rate=16000
        )
        == 10560
    )


def test_warmup_skip_samples_grows_with_the_lookahead():
    """The context buffer `run_streaming` builds is context_ms + lookahead_ms long
    (ADR-0072), so the warm-up span to skip must grow by exactly the lookahead -- a
    fixed skip would leave partially-cold output in the comparison for every
    lookahead_ms > 0."""
    base = warmup_skip_samples(
        context_ms=500.0, lookahead_ms=0.0, block_ms=160.0, rate=16000
    )
    with_lookahead = warmup_skip_samples(
        context_ms=500.0, lookahead_ms=160.0, block_ms=160.0, rate=16000
    )
    assert with_lookahead - base == round(160.0 / 1000.0 * 16000)  # 2560 samples


def test_right_context_ms_subtracts_the_hubert_truncation():
    # rate=1000 makes delay_samples and delay_ms the same number, so the input can be
    # written directly in ms with nothing to round.
    assert right_context_ms(delay_samples=1000, rate=1000) == 1000.0 - 20.0


def test_right_context_ms_reproduces_the_validated_default_geometry():
    """At the validated default geometry (crossfade_ms=25, sola_search_ms=5), the
    analytic relationship traced from `_emit_delay` / `_emit_with_crossfade`
    (vspeech/lib/stream_vc.py) gives
    delay_ms == _HUBERT_TRUNCATION_MS(20) + crossfade_ms(25) + sola_search_ms(5) +
    lookahead_ms, so the old hardcoded `30.0 + lookahead_ms` was correct only for this
    one geometry: crossfade_ms/sola_search_ms are user-settable, ge=0, so a different
    config made it silently wrong.

    This pins the arithmetic `right_context_ms` implements against that traced
    relationship, with delay_ms constructed by hand rather than by calling
    `right_context_ms` again. It does not run the real StreamingVc -- that needs a live
    model and GPU, which are out of reach here.
    """
    rate = 1000  # samples == ms at this rate, so delay_ms needs no rounding
    delay_ms_at_lookahead_0 = 20.0 + 25.0 + 5.0 + 0.0
    delay_ms_at_lookahead_160 = 20.0 + 25.0 + 5.0 + 160.0
    assert right_context_ms(int(delay_ms_at_lookahead_0), rate) == 30.0
    assert right_context_ms(int(delay_ms_at_lookahead_160), rate) == 190.0
