"""Resolving the true device rate (ADR-0071).

sounddevice is stubbed: these are pure decisions over the device table, and the real
table differs per machine.
"""

import logging

import pytest
import sounddevice as sd

from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import open_device_stream
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


# --- open_device_stream: the shared resolve -> log -> open -> verify sequence --------


class _FakeStream:
    """Stands in for sd.Raw{Input,Output}Stream: records how it was opened.

    `start_error`, when given, makes `start()` raise instead of succeeding -- the
    Pa_OpenStream-succeeds/Pa_StartStream-fails case a real sounddevice stream can hit
    (e.g. the device grabbed exclusively between open and start).
    """

    def __init__(self, rate: int, start_error: Exception | None = None) -> None:
        self.samplerate = float(rate)
        self.started = False
        self.closed = False
        self._start_error = start_error

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    def close(self) -> None:
        self.closed = True


def test_the_attempted_rate_is_logged_before_the_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing open must still leave a line saying which device and which rate were
    attempted -- that line is the only clue when PortAudio rejects the open, which is
    why it is emitted before the open rather than after it.
    """

    def _explode(rate: int) -> _FakeStream:
        raise OSError(f"Invalid sample rate {rate}")

    with caplog.at_level(logging.INFO):
        with pytest.raises(OSError):
            open_device_stream(
                device=_device(0),
                override=None,
                input=False,
                config_key="playback.output_device_rate",
                opening="use output device",
                subject="playback",
                open_stream=_explode,
            )
    lines = [
        r.getMessage() for r in caplog.records if "use output device" in r.getMessage()
    ]
    assert len(lines) == 1
    # The WASAPI counterpart's rate, not the 44100 the MME row claims.
    assert "48000Hz" in lines[0]


def test_an_unresolvable_rate_raises_before_the_device_is_opened() -> None:
    """Rate resolution comes first, so a device whose rate cannot be decided never
    reaches the open: there is no half-opened stream left behind on that path.
    """
    opened: list[int] = []

    def _open(rate: int) -> _FakeStream:
        opened.append(rate)
        return _FakeStream(rate)

    with pytest.raises(DeviceRateUnresolvedError):
        open_device_stream(
            # "Microsoft サウンド マッパー": no WASAPI counterpart to borrow a rate from.
            device=_device(2),
            override=None,
            input=True,
            config_key="recording.input_device_rate",
            opening="use input device",
            subject="recording",
            open_stream=_open,
        )
    assert opened == []


def test_a_start_failure_still_closes_the_stream() -> None:
    """Pa_OpenStream succeeding but Pa_StartStream failing must still close the stream.

    sounddevice's stream objects have no `__del__` and `close()` is the sole caller of
    Pa_CloseStream, so an unclosed stream here leaks the native handle for good once
    this function's frame is torn down. Every caller of open_device_stream retries a
    device fault in a loop (stream_vc's steady-state reconnect, worker/playback.py's
    per-utterance reopen) -- a persistent fault (e.g. the device grabbed exclusively by
    another process) would otherwise leak one handle per retry, unboundedly, for as
    long as the pipeline keeps running. Found in review, not by a mutation -- this test
    asserts the fix directly rather than only proving some other test would fail
    without it.
    """
    made: list[_FakeStream] = []

    def _open(rate: int) -> _FakeStream:
        stream = _FakeStream(rate, start_error=sd.PortAudioError("device busy"))
        made.append(stream)
        return stream

    with pytest.raises(sd.PortAudioError):
        open_device_stream(
            device=_device(0),
            override=None,
            input=False,
            config_key="playback.output_device_rate",
            opening="use output device",
            subject="playback",
            open_stream=_open,
        )
    assert len(made) == 1
    assert made[0].closed is True


def test_a_successful_start_is_not_closed_by_open_device_stream() -> None:
    """The success path must NOT close the stream -- the caller needs it open to
    actually use it. Only the start-failure path (above) closes early."""
    stream, rate = open_device_stream(
        device=_device(0),
        override=None,
        input=False,
        config_key="playback.output_device_rate",
        opening="use output device",
        subject="playback",
        open_stream=_FakeStream,
    )
    assert rate == 48000
    assert stream.started is True
    assert stream.closed is False
