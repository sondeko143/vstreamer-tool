import json

from vspeech.config import Config


def test_secret_roundtrip_via_model_dump_json():
    cfg = Config.model_validate(
        {
            "ami": {"appkey": "topsecret"},
            "gcp": {"service_account_info": {"private_key": "gcpsecret"}},
        }
    )
    assert cfg.ami.appkey.get_secret_value() == "topsecret"
    assert cfg.gcp.service_account_info["private_key"].get_secret_value() == "gcpsecret"

    dumped = cfg.model_dump_json()
    # field_serializer(when_used="json") により秘密値は平文で出力される（GUI→本体の受け渡しに必須）
    data = json.loads(dumped)
    assert data["ami"]["appkey"] == "topsecret"
    assert data["gcp"]["service_account_info"]["private_key"] == "gcpsecret"

    reloaded = Config.model_validate(data)
    assert reloaded.ami.appkey.get_secret_value() == "topsecret"
    secret = reloaded.gcp.service_account_info["private_key"]
    assert secret.get_secret_value() == "gcpsecret"
