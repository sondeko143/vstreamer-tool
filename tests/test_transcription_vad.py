from pathlib import Path


def test_transcription_config_vad_defaults_are_off_and_sane():
    from vspeech.config import TranscriptionConfig

    config = TranscriptionConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1


def test_create_session_none_when_gate_disabled():
    from vspeech.config import TranscriptionConfig
    from vspeech.worker.transcription import create_transcription_vad_session

    assert create_transcription_vad_session(TranscriptionConfig()) is None
