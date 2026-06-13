from pathlib import Path

from vspeech.config import VoicevoxConfig


def test_voicevox_config_new_fields_defaults():
    cfg = VoicevoxConfig()
    assert cfg.model_dir == Path("./voicevox/models/vvms")
    assert cfg.onnxruntime_path is None


def test_voicevox_config_parse_obj_roundtrip():
    cfg = VoicevoxConfig.parse_obj(
        {
            "speaker_id": 3,
            "openjtalk_dir": "d",
            "model_dir": "m",
            "onnxruntime_path": "o.dll",
        }
    )
    assert cfg.speaker_id == 3
    assert cfg.model_dir == Path("m")
    assert cfg.onnxruntime_path == Path("o.dll")
