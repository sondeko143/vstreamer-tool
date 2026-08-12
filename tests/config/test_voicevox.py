from pathlib import Path

from vspeech.config import VoicevoxConfig


def test_voicevox_config_new_fields_defaults():
    cfg = VoicevoxConfig()
    assert cfg.model_dir == Path("./voicevox/models/vvms")
    assert cfg.onnxruntime_path is None


def test_voicevox_config_model_validate_roundtrip():
    cfg = VoicevoxConfig.model_validate(
        {
            "speaker_id": 3,
            "openjtalk_dir": "d",
            "model_dir": "m",
            "onnxruntime_path": "o.dll",
        }
    )
    assert cfg.speaker_id == 3
    assert cfg.openjtalk_dir == Path("d")
    assert cfg.model_dir == Path("m")
    assert cfg.onnxruntime_path == Path("o.dll")
