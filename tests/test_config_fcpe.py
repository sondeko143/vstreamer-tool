from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig


def test_fcpe_extractor_and_model_file():
    assert F0ExtractorType("fcpe") is F0ExtractorType.fcpe
    c = RvcConfig.model_validate(
        {"f0_extractor_type": "fcpe", "fcpe_model_file": "x.onnx"}
    )
    assert c.f0_extractor_type is F0ExtractorType.fcpe
    assert str(c.fcpe_model_file) == "x.onnx"
