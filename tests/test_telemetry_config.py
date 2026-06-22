from vspeech.config import Config
from vspeech.config import TelemetryConfig


def test_defaults():
    c = TelemetryConfig()
    assert c.enable is True
    assert c.max_samples == 5000
    assert c.log_raw_e2e is True
    assert c.skew_warn_threshold == 120.0


def test_config_has_telemetry():
    assert Config().telemetry.enable is True
