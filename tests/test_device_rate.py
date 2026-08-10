"""Resolving the true device rate (ADR-0071).

sounddevice is stubbed: these are pure decisions over the device table, and the real
table differs per machine.
"""

import pytest

from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import resolve_device_rate

WASAPI = 2
MME = 0

# (hostapi index -> name) and the device rows resolve_device_rate reads.
_HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
]
_DEVICES = [
    # MME truncates the name to 31 chars and lies about the rate.
    {
        "index": 0,
        "name": "Speakers (Realtek(R) Audio)",
        "hostapi": MME,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "index": 1,
        "name": "Microphone Array (Realtek(R) Au",
        "hostapi": MME,
        "max_input_channels": 4,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "index": 2,
        "name": "Microsoft サウンド マッパー - Input",
        "hostapi": MME,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "index": 3,
        "name": "Ambiguous Device",
        "hostapi": MME,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "index": 10,
        "name": "Speakers (Realtek(R) Audio)",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "index": 11,
        "name": "Microphone Array (Realtek(R) Audio)",
        "hostapi": WASAPI,
        "max_input_channels": 4,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "index": 12,
        "name": "Ambiguous Device A",
        "hostapi": WASAPI,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "index": 13,
        "name": "Ambiguous Device B",
        "hostapi": WASAPI,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
]


@pytest.fixture(autouse=True)
def _stub_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    import vspeech.lib.audio as audio

    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _HOSTAPIS)
    monkeypatch.setattr(audio.sd, "query_devices", lambda: _DEVICES)


def _device(index: int) -> DeviceInfo:
    for raw in _DEVICES:
        if raw["index"] == index:
            return DeviceInfo.model_validate(raw)
    raise AssertionError(index)


def test_explicit_override_wins() -> None:
    rate, how = resolve_device_rate(
        _device(0), 96000, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 96000
    assert "playback.output_device_rate" in how


def test_wasapi_device_uses_its_own_default_samplerate() -> None:
    rate, how = resolve_device_rate(
        _device(10), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    assert "WASAPI" in how


def test_mme_device_takes_the_rate_from_its_wasapi_counterpart() -> None:
    """PortAudio reports 44100 for this MME device; the endpoint really runs at 48000."""
    rate, _ = resolve_device_rate(
        _device(0), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    rate, _ = resolve_device_rate(
        _device(1), None, input=True, config_key="recording.input_device_rate"
    )
    assert rate == 48000


def test_counterpart_match_respects_direction() -> None:
    """An output-only WASAPI row must not answer for an input device."""
    with pytest.raises(DeviceRateUnresolvedError):
        resolve_device_rate(
            _device(0), None, input=True, config_key="recording.input_device_rate"
        )


def test_pseudo_device_without_a_counterpart_fails_loud() -> None:
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(2), None, input=True, config_key="recording.input_device_rate"
        )
    assert "recording.input_device_rate" in str(excinfo.value)


def test_conflicting_counterparts_fail_loud_rather_than_guess() -> None:
    """Two WASAPI rows match the prefix and disagree: never pick one silently."""
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(3), None, input=True, config_key="stream_vc.input_device_rate"
        )
    assert "stream_vc.input_device_rate" in str(excinfo.value)
