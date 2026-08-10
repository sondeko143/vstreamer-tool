import io

from vspeech.config import Config
from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig
from vspeech.config import StreamVcConfig
from vspeech.config import TransportType


def test_stream_vc_defaults():
    c = StreamVcConfig()
    assert c.enable is False
    # 160/500/25, which was clean in the on-hardware ear check (RTX 4060 Laptop / fcpe /
    # a real voice).
    assert c.block_ms == 160.0
    assert c.context_ms == 500.0
    assert c.crossfade_ms == 25.0
    assert c.transport_type == TransportType.in_process
    assert c.max_queued_blocks == 8
    # nested rvc is an independent RvcConfig (ADR-0054/0046): default_factory
    # gives each instance its own copy, not a shared mutable default.
    assert isinstance(c.rvc, RvcConfig)
    assert StreamVcConfig().rvc is not StreamVcConfig().rvc
    # Streaming's default f0 extractor is fcpe, not [rvc]'s rmvpe, so that a minimal
    # [stream_vc] comes up in the configuration that passed the on-hardware ear check
    # (ADR-0053).
    assert c.rvc.f0_extractor_type is F0ExtractorType.fcpe
    # the utterance path is unchanged
    assert RvcConfig().f0_extractor_type is F0ExtractorType.rmvpe


def test_stream_vc_rvc_f0_default_applies_to_explicit_table():
    """Even with a [stream_vc.rvc] table written out, omitting f0_extractor_type gives
    fcpe.

    default_factory only fires when the table is missing altogether. In a realistic config
    carrying model_file and friends, pydantic validates the table and an omitted f0 falls
    back to RvcConfig's default (rmvpe) -- this is the regression test pinning that the
    before-validator corrects it to fcpe.
    """
    # Passed as a dict (TOML/JSON are dicts internally too). Passing a dict to the nested
    # rvc type-checks under model_validate, and the before-validator runs as usual.
    # f0 omitted -> fcpe.
    absent = StreamVcConfig.model_validate(
        {"rvc": {"model_file": "/models/voice.onnx"}}
    )
    assert absent.rvc.f0_extractor_type is F0ExtractorType.fcpe
    # An explicit rmvpe is honoured (the correction only applies when unspecified).
    explicit_rmvpe = StreamVcConfig.model_validate(
        {"rvc": {"model_file": "/models/voice.onnx", "f0_extractor_type": "rmvpe"}}
    )
    assert explicit_rmvpe.rvc.f0_extractor_type is F0ExtractorType.rmvpe
    # An explicit fcpe is left as-is too.
    explicit_fcpe = StreamVcConfig.model_validate(
        {"rvc": {"model_file": "/models/voice.onnx", "f0_extractor_type": "fcpe"}}
    )
    assert explicit_fcpe.rvc.f0_extractor_type is F0ExtractorType.fcpe


def test_stream_vc_rvc_f0_from_toml_table_without_f0_is_fcpe():
    """Omitting f0_extractor_type in TOML's [stream_vc.rvc] also gives fcpe."""
    toml_text = b"""
[stream_vc]
enable = true

[stream_vc.rvc]
model_file = "/models/voice.onnx"
"""
    f = io.BytesIO(toml_text)
    f.name = "config.toml"
    c = Config.read_config_from_file(f)
    assert c.stream_vc.rvc.f0_extractor_type is F0ExtractorType.fcpe


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


def test_stream_vc_latency_defaults_to_low():
    """The default equals the value that used to be hardcoded, so existing configs do
    not change behaviour (ADR-0071)."""
    c = StreamVcConfig()
    assert c.input_latency == "low"
    assert c.output_latency == "low"


def test_stream_vc_latency_accepts_high_and_explicit_seconds():
    """PortAudio takes either of its two device defaults or an arbitrary
    suggestedLatency in seconds; all three have to survive validation."""
    c = StreamVcConfig.model_validate({"input_latency": "high", "output_latency": 0.02})
    assert c.input_latency == "high"
    assert c.output_latency == 0.02


def test_stream_vc_latency_sides_are_independent():
    """Input and output are different devices (different machines once the role is
    split), so raising one must not move the other -- the reason ADR-0071 rejected a
    single shared field."""
    c = StreamVcConfig.model_validate({"output_latency": "high"})
    assert c.input_latency == "low"
    assert c.output_latency == "high"


def test_stream_vc_latency_parses_from_toml():
    toml_text = b"""
[stream_vc]
input_latency = "high"
output_latency = 0.05
"""
    f = io.BytesIO(toml_text)
    f.name = "config.toml"
    c = Config.read_config_from_file(f)
    assert c.stream_vc.input_latency == "high"
    assert c.stream_vc.output_latency == 0.05


def test_stream_vc_latency_rejects_unknown_string_and_non_positive():
    """Bad values fail at config load, which ADR-0068 already routes into the same
    per-problem report preflight uses -- hence no dedicated preflight check."""
    import pytest
    from pydantic import ValidationError

    # A typo must not silently fall through to a float coercion.
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"input_latency": "lowest"})
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"output_latency": 0.0})
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"input_latency": -0.01})


def test_stream_vc_latency_survives_export_to_toml_round_trip():
    import toml as toml_lib

    c = Config()
    c.stream_vc.output_latency = 0.05
    reloaded = toml_lib.loads(c.export_to_toml())
    assert reloaded["stream_vc"]["input_latency"] == "low"
    assert reloaded["stream_vc"]["output_latency"] == 0.05


def test_lookahead_defaults_to_the_measured_value_and_rejects_negative():
    """40ms is the on-hardware choice (ADR-0072): 65% of the p95 improvement for 25% of
    the latency. Pinned so the default cannot drift without the ADR moving with it."""
    import pytest
    from pydantic import ValidationError

    assert StreamVcConfig().lookahead_ms == 40.0
    with pytest.raises(ValidationError):
        StreamVcConfig(lookahead_ms=-1.0)


def test_lookahead_zero_is_still_reachable_for_the_pre_lookahead_geometry():
    """Turning it off must stay possible: the default now costs 40ms of latency, and a
    latency-critical deployment has to be able to buy that back."""
    assert StreamVcConfig(lookahead_ms=0.0).lookahead_ms == 0.0
