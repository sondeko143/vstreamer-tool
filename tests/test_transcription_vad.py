from pathlib import Path


def test_transcription_config_vad_defaults_are_off_and_sane():
    from vspeech.config import TranscriptionConfig

    config = TranscriptionConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1
