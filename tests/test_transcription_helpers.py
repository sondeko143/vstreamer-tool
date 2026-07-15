from types import SimpleNamespace

import numpy as np
import pytest

from vspeech.config import SampleFormat
from vspeech.config import WhisperConfig
from vspeech.shared_context import SoundInput
from vspeech.worker.transcription import join_transcribed_segments
from vspeech.worker.transcription import pcm_to_waveform


def _sound(
    data: bytes,
    channels: int = 1,
    rate: int = 16000,
    sample_format: SampleFormat = SampleFormat.INT16,
) -> SoundInput:
    return SoundInput(data=data, rate=rate, format=sample_format, channels=channels)


def test_pcm_to_waveform_int16_mono_is_normalized_to_unit_range():
    pcm = np.array([0, 16384, -32768], dtype=np.int16).tobytes()
    waveform = pcm_to_waveform(_sound(pcm))
    assert waveform.dtype == np.float32
    assert np.allclose(waveform, [0.0, 0.5, -1.0])


def test_pcm_to_waveform_stereo_is_averaged_to_mono():
    # frames: (16384, 0) -> 0.25, (-32768, 0) -> -0.5
    pcm = np.array([16384, 0, -32768, 0], dtype=np.int16).tobytes()
    waveform = pcm_to_waveform(_sound(pcm, channels=2))
    assert np.allclose(waveform, [0.25, -0.5])


def test_pcm_to_waveform_16k_is_not_resampled():
    pcm = np.arange(-100, 100, dtype=np.int16).tobytes()
    waveform = pcm_to_waveform(_sound(pcm))
    assert len(waveform) == 200


def test_pcm_to_waveform_resamples_non_16k_to_16k():
    # 48 kHz mono int16, 1 second of audio -> ~16000 samples at 16 kHz.
    src_rate = 48000
    pcm = np.zeros(src_rate, dtype=np.int16)
    pcm[:: src_rate // 100] = 8000  # sparse impulses so it is not pure silence
    waveform = pcm_to_waveform(_sound(pcm.tobytes(), rate=src_rate))
    assert waveform.dtype == np.float32
    assert abs(len(waveform) - 16000) <= 10


def test_pcm_to_waveform_float32_is_decoded_without_scaling():
    samples = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    waveform = pcm_to_waveform(
        _sound(samples.tobytes(), sample_format=SampleFormat.FLOAT32)
    )
    assert waveform.dtype == np.float32
    assert np.allclose(waveform, [0.0, 0.5, -1.0])


def test_pcm_to_waveform_uint8_is_debiased_to_unit_range():
    # unsigned 8-bit: 128 == silence; must subtract the 128 bias, not read signed.
    pcm = np.array([128, 0, 255, 192], dtype=np.uint8).tobytes()
    waveform = pcm_to_waveform(_sound(pcm, sample_format=SampleFormat.UINT8))
    assert np.allclose(waveform, [0.0, -1.0, 127 / 128, 0.5])


def test_pcm_to_waveform_int8_is_normalized_to_unit_range():
    pcm = np.array([0, 64, -128, 127], dtype=np.int8).tobytes()
    waveform = pcm_to_waveform(_sound(pcm, sample_format=SampleFormat.INT8))
    assert np.allclose(waveform, [0.0, 0.5, -1.0, 127 / 128])


def test_pcm_to_waveform_int24_is_normalized_to_unit_range():
    # 3-byte little-endian signed: 0x400000 -> +0.5, 0x800000 -> -1.0
    pcm = bytes([0x00, 0x00, 0x40]) + bytes([0x00, 0x00, 0x80])
    waveform = pcm_to_waveform(_sound(pcm, sample_format=SampleFormat.INT24))
    assert np.allclose(waveform, [0.5, -1.0])


def test_pcm_to_waveform_unsupported_format_raises_valueerror():
    # The worker relies on ValueError (not garbage) to skip a bad utterance.
    with pytest.raises(ValueError):
        pcm_to_waveform(_sound(b"\x00\x00", sample_format=SampleFormat.INVALID))


def test_pcm_to_waveform_resamples_stereo_and_float32_to_16k():
    src_rate = 48000
    stereo = np.zeros((src_rate, 2), dtype=np.int16)
    stereo[:: src_rate // 100, :] = 8000
    w_stereo = pcm_to_waveform(_sound(stereo.tobytes(), channels=2, rate=src_rate))
    assert abs(len(w_stereo) - 16000) <= 10

    f32 = np.zeros(src_rate, dtype=np.float32)
    f32[:: src_rate // 100] = 0.5
    w_f32 = pcm_to_waveform(
        _sound(f32.tobytes(), rate=src_rate, sample_format=SampleFormat.FLOAT32)
    )
    assert w_f32.dtype == np.float32
    assert abs(len(w_f32) - 16000) <= 10


def test_pcm_to_waveform_resample_preserves_signal_energy():
    # A 440 Hz tone (well under the 8 kHz Nyquist) must survive resampling,
    # not collapse to silence -- length alone would not catch that.
    src_rate = 48000
    t = np.arange(src_rate) / src_rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    waveform = pcm_to_waveform(_sound(tone.tobytes(), rate=src_rate))
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    assert rms > 0.2  # ~0.354 for a 0.5-amplitude sine


def test_pcm_to_waveform_empty_at_non_16k_returns_empty():
    # Empty buffer at a non-16k rate must not crash the PyAV resampler.
    waveform = pcm_to_waveform(_sound(b"", rate=48000))
    assert len(waveform) == 0


def _segment(text: str, no_speech_prob: float, avg_logprob: float, temperature: float):
    return SimpleNamespace(
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        temperature=temperature,
    )


def test_join_transcribed_segments_keeps_confident_segments():
    config = WhisperConfig(no_speech_prob_threshold=0.6, logprob_threshold=-1.0)
    segments = [
        _segment("こんにちは", no_speech_prob=0.1, avg_logprob=-0.2, temperature=0.0),
        _segment("世界", no_speech_prob=0.2, avg_logprob=-0.3, temperature=0.5),
    ]
    assert join_transcribed_segments(segments, config) == "こんにちは世界"


def test_join_transcribed_segments_drops_high_no_speech_prob():
    config = WhisperConfig(no_speech_prob_threshold=0.6, logprob_threshold=-1.0)
    segments = [
        _segment("ノイズ", no_speech_prob=0.9, avg_logprob=-0.2, temperature=0.0),
        _segment("本文", no_speech_prob=0.1, avg_logprob=-0.2, temperature=0.0),
    ]
    assert join_transcribed_segments(segments, config) == "本文"


def test_join_transcribed_segments_drops_low_logprob_and_hot_temperature():
    config = WhisperConfig(no_speech_prob_threshold=0.6, logprob_threshold=-1.0)
    segments = [
        _segment("低確信", no_speech_prob=0.1, avg_logprob=-1.5, temperature=0.0),
        _segment("高温", no_speech_prob=0.1, avg_logprob=-0.2, temperature=1.0),
        _segment("採用", no_speech_prob=0.1, avg_logprob=-0.2, temperature=0.0),
    ]
    assert join_transcribed_segments(segments, config) == "採用"


async def test_recording_log_write_failure_degrades(monkeypatch, tmp_path):
    from io import BytesIO

    from vspeech.worker import transcription as tx

    def _boom(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(tx.Path, "mkdir", _boom)
    tx._rec_log_warned.clear()
    # 例外を投げずに返る（プロセスを止めない）
    await tx.log_transcribed(tmp_path, BytesIO(b"data"), "text")
    assert str(tmp_path) in tx._rec_log_warned
