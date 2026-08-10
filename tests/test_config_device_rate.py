"""The four device-rate overrides (ADR-0071)."""

import pytest
from pydantic import ValidationError

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import StreamVcConfig


def test_defaults_are_none_so_existing_configs_still_load() -> None:
    assert RecordingConfig().input_device_rate is None
    assert PlaybackConfig().output_device_rate is None
    assert StreamVcConfig().input_device_rate is None
    assert StreamVcConfig().output_device_rate is None


def test_explicit_rates_are_accepted() -> None:
    assert RecordingConfig(input_device_rate=48000).input_device_rate == 48000
    assert PlaybackConfig(output_device_rate=44100).output_device_rate == 44100
    sv = StreamVcConfig(input_device_rate=48000, output_device_rate=48000)
    assert (sv.input_device_rate, sv.output_device_rate) == (48000, 48000)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_rates_are_rejected(bad: int) -> None:
    with pytest.raises(ValidationError):
        RecordingConfig(input_device_rate=bad)
    with pytest.raises(ValidationError):
        PlaybackConfig(output_device_rate=bad)
    with pytest.raises(ValidationError):
        StreamVcConfig(input_device_rate=bad)
    with pytest.raises(ValidationError):
        StreamVcConfig(output_device_rate=bad)
