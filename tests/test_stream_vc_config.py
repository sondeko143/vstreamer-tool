import io

from vspeech.config import Config
from vspeech.config import RvcConfig
from vspeech.config import StreamVcConfig
from vspeech.config import TransportType


def test_stream_vc_defaults():
    c = StreamVcConfig()
    assert c.enable is False
    assert c.block_ms == 80.0
    assert c.context_ms == 100.0
    assert c.crossfade_ms == 10.0
    assert c.transport_type == TransportType.in_process
    assert c.max_queued_blocks == 8
    # nested rvc is an independent RvcConfig (ADR-0054/0046): default_factory
    # gives each instance its own copy, not a shared mutable default.
    assert isinstance(c.rvc, RvcConfig)
    assert StreamVcConfig().rvc is not StreamVcConfig().rvc


def test_config_has_stream_vc_section():
    c = Config()
    assert c.stream_vc.enable is False


def test_stream_vc_parses_from_toml():
    toml_text = b"""
[stream_vc]
enable = true
block_ms = 120
context_ms = 200
crossfade_ms = 12
transport_type = "in_process"

[stream_vc.rvc]
model_file = "/models/voice.onnx"
f0_extractor_type = "fcpe"
"""
    f = io.BytesIO(toml_text)
    f.name = "config.toml"
    c = Config.read_config_from_file(f)
    assert c.stream_vc.enable is True
    assert c.stream_vc.block_ms == 120.0
    assert c.stream_vc.crossfade_ms == 12.0
    assert c.stream_vc.rvc.model_file.as_posix() == "/models/voice.onnx"
    assert c.stream_vc.rvc.f0_extractor_type.value == "fcpe"


def test_stream_vc_survives_export_to_toml_round_trip():
    import toml as toml_lib

    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.block_ms = 160.0
    dumped = c.export_to_toml()
    reloaded = toml_lib.loads(dumped)
    assert reloaded["stream_vc"]["enable"] is True
    assert reloaded["stream_vc"]["block_ms"] == 160.0
    assert reloaded["stream_vc"]["transport_type"] == "in_process"


def test_stream_vc_rejects_out_of_range():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StreamVcConfig(block_ms=0)  # gt=0
    with pytest.raises(ValidationError):
        StreamVcConfig(max_queued_blocks=0)  # gt=0
    with pytest.raises(ValidationError):
        StreamVcConfig(context_ms=-1)  # ge=0
