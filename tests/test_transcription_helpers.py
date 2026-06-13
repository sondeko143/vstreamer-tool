from types import SimpleNamespace

import numpy as np

from vspeech.config import SampleFormat
from vspeech.config import WhisperConfig
from vspeech.shared_context import SoundInput
from vspeech.worker.transcription import join_transcribed_segments
from vspeech.worker.transcription import pcm_to_waveform


def _sound(data: bytes, channels: int = 1) -> SoundInput:
    return SoundInput(
        data=data, rate=16000, format=SampleFormat.INT16, channels=channels
    )


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
