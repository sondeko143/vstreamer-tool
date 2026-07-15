import pytest

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.exceptions import DeviceNotFoundError
from vspeech.lib.audio import DeviceInfo


def _device(index: int = 3):
    return DeviceInfo(
        host_api=0,
        max_input_channels=2,
        max_output_channels=2,
        name="Line 4",
        index=index,
    )


def test_input_resolver_returns_found_device(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: _device(3))
    cfg = RecordingConfig(input_device_name="Line 4")
    assert audio.resolve_input_device(cfg).index == 3


def test_input_resolver_raises_named_error_when_missing(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = RecordingConfig(input_host_api_name="MME", input_device_name="Ghost")
    with pytest.raises(DeviceNotFoundError) as ei:
        audio.resolve_input_device(cfg)
    assert "input_device_name" in str(ei.value)
    assert "Ghost" in str(ei.value)


def test_output_resolver_raises_named_error_when_missing(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = PlaybackConfig(output_device_name="Ghost")
    with pytest.raises(DeviceNotFoundError) as ei:
        audio.resolve_output_device(cfg)
    assert "output_device_name" in str(ei.value)
