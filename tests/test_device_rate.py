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
    # I1 regression fixture: a WASAPI row (20) whose name is a strict prefix of
    # another WASAPI row (21) with a *different* rate, same direction. This is the
    # real shape VAC/Voicemeeter multi-endpoint setups produce on this machine (e.g.
    # real hardware shows "Line 1 (Virtual Audio Cable)" siblings). Resolving device
    # 20 must use its own default_samplerate and must NOT fall through to the
    # counterpart search, which would find both 20 and 21 and blow up as ambiguous.
    {
        "index": 20,
        "name": "Line 1 (Virtual Audio Cable)",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "index": 21,
        "name": "Line 1 (Virtual Audio Cable) 2",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    # M2 fixture: an MME device whose name matches TWO WASAPI rows (a duplicate-name
    # multi-endpoint setup), but both rows agree on the rate. ADR-0071 promises this
    # resolves (uniqueness is about the rate, not the row count).
    {
        "index": 22,
        "name": "Rear Speaker (Multi)",
        "hostapi": MME,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "index": 23,
        "name": "Rear Speaker (Multi) - Front",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "index": 24,
        "name": "Rear Speaker (Multi) - Rear",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    # M6 fixture: a WASAPI device reporting default_samplerate == 0 (a device in a
    # bad state), both directly (30) and as the sole counterpart of an MME device
    # (31 -> 32).
    {
        "index": 30,
        "name": "Disabled Output",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 0.0,
    },
    {
        "index": 31,
        "name": "Disabled Line",
        "hostapi": MME,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "index": 32,
        "name": "Disabled Line",
        "hostapi": WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 0.0,
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
    """A WASAPI row resolves from its own default_samplerate even when another
    WASAPI row has its name as a strict prefix and a different rate.

    Regression for the WASAPI branch itself: VAC/Voicemeeter commonly produce such
    siblings on this machine, and without the branch, this query would fall through
    to the counterpart search, find both row 20 and row 21, and raise instead of
    returning row 20's own 48000.
    """
    rate, how = resolve_device_rate(
        _device(20), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    assert "ミックス形式" in how


def test_wasapi_own_value_guards_against_a_shifted_device_table() -> None:
    """If the device table shifted since `device` was looked up (e.g. a
    sd._terminate()/_initialize() cycle changed indices), the row now at this index
    may belong to a different device. A name mismatch must be treated the same as an
    out-of-range index, not silently answer with the wrong device's rate.
    """
    stale = DeviceInfo(
        host_api=WASAPI,
        max_input_channels=0,
        max_output_channels=2,
        name="A Device That No Longer Lives At This Index",
        index=10,
    )
    with pytest.raises(DeviceRateUnresolvedError):
        resolve_device_rate(
            stale, None, input=False, config_key="playback.output_device_rate"
        )


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


def test_multiple_matching_counterparts_agreeing_on_the_rate_resolve() -> None:
    """ADR-0071: several WASAPI rows may share a name prefix (a duplicate-name
    multi-endpoint setup, as VAC/Voicemeeter produce). Resolution only requires the
    *rates* to agree, not that exactly one row matches.
    """
    rate, how = resolve_device_rate(
        _device(22), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    assert "から逆引き" in how


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
    message = str(excinfo.value)
    assert "stream_vc.input_device_rate" in message
    assert "一致しません" in message
    assert "44100" in message
    assert "48000" in message


def test_wasapi_own_rate_of_zero_fails_loud() -> None:
    """default_samplerate == 0 (some host APIs report this for a device in a bad
    state) must not silently pass through as a valid rate.
    """
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(30), None, input=False, config_key="playback.output_device_rate"
        )
    assert "playback.output_device_rate" in str(excinfo.value)


def test_counterpart_rate_of_zero_fails_loud() -> None:
    """Same as above, but the zero rate is discovered via the WASAPI counterpart
    lookup rather than the device's own default_samplerate.
    """
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(31), None, input=False, config_key="playback.output_device_rate"
        )
    assert "playback.output_device_rate" in str(excinfo.value)
