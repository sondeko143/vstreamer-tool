import json

from vspeech.config import Config


def test_secret_roundtrip_via_model_dump_json():
    cfg = Config.model_validate({"ami": {"appkey": "topsecret"}})
    assert cfg.ami.appkey.get_secret_value() == "topsecret"
    dumped = cfg.model_dump_json()
    # json_encoders により秘密値は平文で出力される（GUI→本体の受け渡しに必須）
    assert "topsecret" in dumped
    reloaded = Config.model_validate(json.loads(dumped))
    assert reloaded.ami.appkey.get_secret_value() == "topsecret"
