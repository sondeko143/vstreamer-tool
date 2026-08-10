import pytest
from pydantic import ValidationError

from vspeech.config import Config
from vspeech.config import RvcConfig


def test_f0_filter_radius_defaults_to_one():
    assert RvcConfig().f0_filter_radius == 1


def test_f0_filter_radius_rejects_out_of_range():
    with pytest.raises(ValidationError):
        RvcConfig(f0_filter_radius=-1)
    with pytest.raises(ValidationError):
        RvcConfig(f0_filter_radius=8)


def test_stream_vc_rvc_table_gets_the_same_default():
    # [stream_vc.rvc] is the same RvcConfig model, so the knob reaches the streaming
    # path without a second field. A realistic table (not an absent one) is used here
    # because that is the case the before-validator handles.
    config = Config.model_validate({"stream_vc": {"rvc": {"model_file": "x.onnx"}}})
    assert config.stream_vc.rvc.f0_filter_radius == 1


def test_stream_vc_rvc_table_honours_an_explicit_value():
    config = Config.model_validate(
        {"stream_vc": {"rvc": {"model_file": "x.onnx", "f0_filter_radius": 0}}}
    )
    assert config.stream_vc.rvc.f0_filter_radius == 0
