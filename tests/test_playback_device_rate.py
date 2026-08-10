"""Native-rate open + per-utterance conversion for the utterance playback path
(ADR-0070/0071, Task 8).

Mirrors the fixture shapes already used in tests/test_recording_device_rate.py and
tests/test_stream_vc_playback.py (there is no tests/conftest.py in this repo, so per-file
duplication is the house pattern).
"""

import logging
from asyncio import Queue
from asyncio import wait_for
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from vspeech.config import EventType
from vspeech.config import PlaybackConfig
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.telemetry import telemetry
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker import playback as playback_mod
from vspeech.worker.playback import MAX_CACHED_RESAMPLERS
from vspeech.worker.playback import OutputStream
from vspeech.worker.playback import sd_playback_worker

_MME = 0
_WASAPI = 1
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
# The MME row lies about the rate (PortAudio hardcodes 44100 there) and truncates the name
# to 31 characters; the WASAPI row for the same endpoint carries the true mix rate.
_DEVICES = [
    {
        "index": 0,
        "name": "Speakers (Realtek(R) Audio)",
        "hostapi": _MME,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "index": 1,
        "name": "Speakers (Realtek(R) Audio) 2ch",
        "hostapi": _WASAPI,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        # A pseudo device with no WASAPI counterpart: its rate cannot be decided.
        "index": 2,
        "name": "プライマリ サウンド ドライバー",
        "hostapi": _MME,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
]

DEVICE_RATE = 48000
TTS_RATE = 24000
VC_RATE = 40000


class _FakeDevice:
    """Stands in for sd.RawOutputStream: records the bytes written to the device.

    `fail_on_write` makes the n-th write raise, which is how a runtime device fault is
    reproduced. Like a real stream it always reports a `samplerate`.
    """

    def __init__(self, fail_on_write: int | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.samplerate = float(kwargs.get("samplerate", DEVICE_RATE))
        self.writes: list[bytes] = []
        self.attempts = 0
        self.started = False
        self.closed = False
        self._fail_on_write = fail_on_write

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        # Counted by attempt, not by len(writes): a failed write records nothing, so
        # counting the successes would make `fail_on_write=1` fail forever.
        self.attempts += 1
        if self.attempts == self._fail_on_write:
            raise OSError("output sink gone")
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def stubbed_device_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the device table alone, without also patching sd.RawOutputStream."""
    import vspeech.lib.audio as audio

    def _query_devices(index: int | None = None):
        if index is None:
            return _DEVICES
        return next(d for d in _DEVICES if d["index"] == index)

    monkeypatch.setattr(audio.sd, "query_devices", _query_devices)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _HOSTAPIS)


@pytest.fixture
def opened_streams(
    stubbed_device_table: None, monkeypatch: pytest.MonkeyPatch
) -> list[_FakeDevice]:
    """Stub sd.RawOutputStream on top of the device table; yield the streams opened."""
    opened: list[_FakeDevice] = []

    def _open(**kwargs: Any) -> _FakeDevice:
        stream = _FakeDevice(**kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(playback_mod.sd, "RawOutputStream", _open)
    return opened


@pytest.fixture
def enabled_telemetry():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=1000)
    yield telemetry
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


def _i16(pcm: bytes) -> NDArray[np.int16]:
    return np.frombuffer(pcm, dtype=np.int16)


def _sine(rate: int, samples: int, freq: float = 440.0) -> bytes:
    """int16 mono PCM of a sine -- the shape an utterance carries."""
    t = np.arange(samples, dtype=np.float64) / rate
    return np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16).tobytes()


def _peak_frequency(pcm: bytes, rate: int) -> float:
    """The dominant frequency of `pcm` **read at `rate`**."""
    samples = _i16(pcm).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))])


def _sound(
    data: bytes,
    rate: int,
    format: SampleFormat = SampleFormat.INT16,
    channels: int = 1,
) -> SoundInput:
    return SoundInput(data=data, rate=rate, format=format, channels=channels)


def _speech(sound: SoundInput, volume: int | None = None, origin_ts: float = 0.0):
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(
            event=EventType.playback, params=Params(volume=volume)
        ),
        following_events=[],
        text="",
        sound=sound,
        file_path="",
        filters=[],
        trace_id="abc",
        origin_ts=origin_ts,
    )


def _opened() -> OutputStream:
    """An OutputStream with its device already open (INT16 mono).

    Callers must be inside the `opened_streams` fixture (it is what stubs the device table
    and sd.RawOutputStream), so they take it as a parameter even where they only assert on
    the OutputStream.
    """
    output = OutputStream(PlaybackConfig(output_device_index=0))
    output.update_stream_if_changed(format=SampleFormat.INT16, channels=1)
    return output


def _open_log(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        r.getMessage() for r in caplog.records if "output device" in r.getMessage()
    ]
    assert len(lines) == 1, lines
    return lines[0]


# --- the open ------------------------------------------------------------------------


def test_output_device_is_opened_at_the_resolved_native_rate(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    """The endpoint really runs at 48000, so that is what is opened -- not the 24000Hz the
    utterance happens to carry. Asking for the source's rate would hand the conversion to
    the OS, and WASAPI shared mode would refuse the open outright (ADR-0070)."""
    with caplog.at_level(logging.INFO):
        output = _opened()
        output.convert(_sine(TTS_RATE, 240), TTS_RATE, SampleFormat.INT16, 1)
    assert output.device_rate == DEVICE_RATE
    assert output.stream is opened_streams[0]
    assert opened_streams[0].kwargs["samplerate"] == DEVICE_RATE
    assert opened_streams[0].kwargs["channels"] == 1
    assert opened_streams[0].kwargs["dtype"] == "int16"
    assert opened_streams[0].started
    line = _open_log(caplog)
    assert "48000Hz" in line
    assert "WASAPI" in line
    # The source rate is not part of the open; it is reported where it is decided.
    conversions = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("playback ")
    ]
    assert conversions == ["playback 24000Hz -> 48000Hz (プロセス内で変換)"]


def test_configured_output_device_rate_wins_over_the_resolved_one(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    output = OutputStream(
        PlaybackConfig(output_device_index=0, output_device_rate=44100)
    )
    with caplog.at_level(logging.INFO):
        output.update_stream_if_changed(format=SampleFormat.INT16, channels=1)
    assert output.device_rate == 44100
    assert opened_streams[0].kwargs["samplerate"] == 44100
    assert "playback.output_device_rate" in _open_log(caplog)


def test_a_device_reporting_another_rate_is_warned_about(
    stubbed_device_table: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The conversion keeps using the requested rate (a reported 47999 would blow the
    polyphase ratio up to 48000 phases), so a hardware rate that differs is a slow drift in
    the audio and nothing else. This warning is its only trace."""

    def _open(**kwargs: Any) -> _FakeDevice:
        stream = _FakeDevice(**kwargs)
        stream.samplerate = 47999.0
        return stream

    monkeypatch.setattr(playback_mod.sd, "RawOutputStream", _open)
    output = OutputStream(PlaybackConfig(output_device_index=0))
    with caplog.at_level(logging.WARNING):
        output.update_stream_if_changed(format=SampleFormat.INT16, channels=1)
    assert output.device_rate == DEVICE_RATE  # the requested rate, not the reported one
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "47999" in warnings[0]
    assert "48000" in warnings[0]


# --- the device is not reopened for a rate change -------------------------------------


def test_a_source_rate_change_does_not_reopen_the_device(
    opened_streams: list[_FakeDevice],
) -> None:
    """The headline of ADR-0070 for this boundary: 24000Hz TTS and 40000Hz VC utterances
    alternating used to close and reopen the device on every single one."""
    output = _opened()
    for rate in (TTS_RATE, VC_RATE, TTS_RATE, VC_RATE):
        output.update_stream_if_changed(format=SampleFormat.INT16, channels=1)
        output.convert(_sine(rate, rate // 10), rate, SampleFormat.INT16, 1)
    assert len(opened_streams) == 1
    assert not opened_streams[0].closed


def test_a_format_or_channel_change_still_reopens_the_device(
    opened_streams: list[_FakeDevice],
) -> None:
    """Only the rate stopped being a reason to reopen. The format and the channel count
    are part of how the stream itself is opened, so they still are."""
    output = _opened()
    output.update_stream_if_changed(format=SampleFormat.INT16, channels=2)
    assert len(opened_streams) == 2
    assert opened_streams[0].closed
    assert opened_streams[1].kwargs["channels"] == 2
    output.update_stream_if_changed(format=SampleFormat.FLOAT32, channels=2)
    assert len(opened_streams) == 3
    assert opened_streams[2].kwargs["dtype"] == "float32"


def test_a_reopen_drops_the_cached_resamplers(
    opened_streams: list[_FakeDevice],
) -> None:
    """A reopen may land on another device, and therefore another device rate: a resampler
    built for the old rate would then convert to the wrong one, silently."""
    output = _opened()
    output.convert(_sine(TTS_RATE, 2400), TTS_RATE, SampleFormat.INT16, 1)
    assert list(output.resamplers) == [TTS_RATE]
    output.update_stream_if_changed(format=SampleFormat.INT16, channels=2)
    assert output.resamplers == {}


# --- the conversion itself -------------------------------------------------------------


def test_a_source_at_the_device_rate_is_written_untouched(
    opened_streams: list[_FakeDevice],
) -> None:
    """No resampler in the path: the bytes handed to the device are the utterance's own.

    Identity, not equality -- this path must stay bit-identical to the pre-ADR-0070 code,
    which wrote the source bytes straight to the stream.
    """
    output = _opened()
    pcm = _sine(DEVICE_RATE, 4800)
    assert output.convert(pcm, DEVICE_RATE, SampleFormat.INT16, 1) is pcm
    assert output.resamplers == {DEVICE_RATE: None}


def test_an_utterance_is_converted_to_the_device_rate(
    opened_streams: list[_FakeDevice],
) -> None:
    """24 kHz in, 48 kHz out: twice the samples, same 440 Hz tone."""
    output = _opened()
    pcm = _sine(TTS_RATE, 2400)
    out = output.convert(pcm, TTS_RATE, SampleFormat.INT16, 1)
    assert _i16(out).size == 2400 * DEVICE_RATE // TTS_RATE
    assert abs(_peak_frequency(out, DEVICE_RATE) - 440.0) < 3.0
    # The same bytes read at the source rate would be an 880 Hz tone: the assertion above
    # only means something because this one fails.
    assert abs(_peak_frequency(out, TTS_RATE) - 440.0) > 100.0


@pytest.mark.parametrize("rate", [TTS_RATE, VC_RATE, 44100, 11025])
def test_the_converted_utterance_has_its_nominal_length(
    opened_streams: list[_FakeDevice], rate: int
) -> None:
    """One utterance is one buffer: its converted length is the nominal one, with no
    samples held back inside the filter."""
    output = _opened()
    samples = rate // 10  # 100 ms
    out = output.convert(_sine(rate, samples), rate, SampleFormat.INT16, 1)
    assert _i16(out).size == round(samples * DEVICE_RATE / rate)


def test_the_utterance_is_not_shifted_in_time_and_keeps_its_tail(
    opened_streams: list[_FakeDevice],
) -> None:
    """The streaming entry point would delay the output by the filter's group delay and
    leave that much audio inside the filter, so every utterance would start late and lose
    that much of its tail. Two signatures of that, on a step in the middle of a 2400-sample
    24000Hz utterance (4800 samples out at 48000; the group delay of the pair is 100 output
    samples):

    - the step lands at the nominal position, not 100 samples late (measured: 2399 through
      resample_full, 2499 through process);
    - the last output sample is centred on the last input sample, so half the filter window
      sits past the end of the utterance and it reads half scale. Through `process` the
      output stops 100 samples earlier in the signal and the last sample is still full
      scale (measured 20000) -- i.e. the last 100 samples of the utterance never played.
    """
    output = _opened()
    step = np.zeros(2400, dtype=np.int16)
    step[1200:] = 20000
    out = _i16(output.convert(step.tobytes(), TTS_RATE, SampleFormat.INT16, 1))
    assert out.size == 4800
    crossing = int(np.argmax(out > 10000))
    assert abs(crossing - 2400) <= 2, crossing
    # Everything up to the filter's edge taper is intact and full scale...
    assert out[-250:-150].min() > 18000
    # ...and the taper ends exactly on the last input sample.
    assert 9000 < out[-1] < 11000


def test_the_same_utterance_twice_gives_the_same_bytes(
    opened_streams: list[_FakeDevice],
) -> None:
    """Utterances are independent buffers, not a continuous stream: no filter tail may
    survive from one to the next, or the same input would play differently the second
    time (and every utterance would open with a ring from the previous one)."""
    output = _opened()
    pcm = _sine(TTS_RATE, 2400)
    first = output.convert(pcm, TTS_RATE, SampleFormat.INT16, 1)
    second = output.convert(pcm, TTS_RATE, SampleFormat.INT16, 1)
    assert first == second
    # And a loud utterance does not ring into a silent one that follows it.
    loud = np.full(2400, 20000, dtype=np.int16).tobytes()
    silence = np.zeros(2400, dtype=np.int16).tobytes()
    output.convert(loud, TTS_RATE, SampleFormat.INT16, 1)
    after = _i16(output.convert(silence, TTS_RATE, SampleFormat.INT16, 1))
    assert np.max(np.abs(after)) == 0


def test_the_conversion_saturates_instead_of_wrapping(
    opened_streams: list[_FakeDevice],
) -> None:
    """Resampling a full-scale square wave overshoots past +1.0 (Gibbs). A wrapping cast
    would sign-flip those samples into a loud click."""
    output = _opened()
    square = (np.tile([1, 1, 1, 1, -1, -1, -1, -1], 300) * 32767).astype(np.int16)
    reference = PolyphaseResampler(TTS_RATE, DEVICE_RATE).resample_full(
        square.astype(np.float32) / 32768.0
    )
    overshoot = np.flatnonzero(reference > 1.0)
    assert overshoot.size > 0, "this signal must overshoot or the test proves nothing"
    out = _i16(output.convert(square.tobytes(), TTS_RATE, SampleFormat.INT16, 1))
    assert np.array_equal(
        out[overshoot], np.full(overshoot.size, 32767, dtype=np.int16)
    )
    assert out.max() <= 32767
    assert out.min() >= -32768


def test_multichannel_utterances_keep_their_channel_layout(
    opened_streams: list[_FakeDevice],
) -> None:
    """channels > 1 must not be folded to mono by the resample round trip."""
    output = OutputStream(PlaybackConfig(output_device_index=0))
    output.update_stream_if_changed(format=SampleFormat.INT16, channels=2)
    stereo = np.empty(2400 * 2, dtype=np.int16)
    stereo[0::2] = 20000
    stereo[1::2] = -20000
    out = _i16(output.convert(stereo.tobytes(), TTS_RATE, SampleFormat.INT16, 2))
    assert out.size == 4800 * 2
    # Interior samples only: both ends taper, which is the filter's edge, not a downmix.
    assert out[1000:-1000:2].mean() > 15000
    assert out[1001:-1000:2].mean() < -15000


def test_an_empty_utterance_converts_to_nothing(
    opened_streams: list[_FakeDevice],
) -> None:
    """A TTS/VC step that produced no audio still reaches this worker. It must come out
    as an empty write, not as an exception out of the resampler."""
    output = _opened()
    assert output.convert(b"", TTS_RATE, SampleFormat.INT16, 1) == b""


def test_the_resampler_cache_is_bounded(opened_streams: list[_FakeDevice]) -> None:
    """The key arrives with the audio and may come from another machine, so the table
    cannot be allowed to grow without limit."""
    output = _opened()
    rates = [8000 + i * 1000 for i in range(MAX_CACHED_RESAMPLERS + 3)]
    for rate in rates:
        output.convert(_sine(rate, 800), rate, SampleFormat.INT16, 1)
    assert len(output.resamplers) == MAX_CACHED_RESAMPLERS
    # Overflow evicts the least recently used one at a time; it does not drop the table
    # and make the next few utterances pay for a rebuild each.
    assert list(output.resamplers) == rates[-MAX_CACHED_RESAMPLERS:]


def test_the_cache_keeps_the_rates_that_are_actually_in_use(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    """Recency, not insertion order: a rate that keeps being used must survive a stream of
    one-off rates far longer than the table is wide. A TTS voice alternating with anything
    else for hours is exactly this shape, and it is the case the cache exists for.

    Counted in builds (the info line), not in table contents: a table that evicted the
    live rate and rebuilt it would still *contain* it at the end.
    """
    output = _opened()
    with caplog.at_level(logging.INFO):
        for i in range(MAX_CACHED_RESAMPLERS * 2):
            output.convert(_sine(TTS_RATE, 800), TTS_RATE, SampleFormat.INT16, 1)
            one_off = 9000 + i * 1000
            output.convert(_sine(one_off, 800), one_off, SampleFormat.INT16, 1)
    builds = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(f"playback {TTS_RATE}Hz")
    ]
    assert len(builds) == 1, builds
    assert len(output.resamplers) == MAX_CACHED_RESAMPLERS


def test_an_unusable_source_rate_is_not_remembered_as_needing_no_conversion(
    opened_streams: list[_FakeDevice],
) -> None:
    """A rate the resampler refuses (only reachable from a corrupt/invalid sound) must not
    land in the table: the next utterance at that rate would take the "rates already
    match" branch and play unconverted -- silently at the wrong speed, instead of
    failing."""
    output = _opened()
    for _ in range(2):
        with pytest.raises(ValueError):
            output.convert(b"\x00\x00", 0, SampleFormat.INT16, 1)
    assert output.resamplers == {}


def test_a_pathological_source_rate_is_refused_before_it_costs_anything(
    opened_streams: list[_FakeDevice],
) -> None:
    """`WorkerInput.sound.rate` crosses gRPC from another machine unvalidated, and a value
    well inside any plausible range (44101 against a 48000 device) demands a 4.85M-tap
    filter: 563MB and 1.5s, measured. The cap in resample.py refuses it, and nothing about
    it is remembered (ADR-0075)."""
    output = _opened()
    with pytest.raises(ValueError, match="病的"):
        output.convert(_sine(44101, 800), 44101, SampleFormat.INT16, 1)
    assert output.resamplers == {}


# --- the volume and the write ----------------------------------------------------------


async def test_the_volume_is_applied_exactly_as_before(
    opened_streams: list[_FakeDevice],
) -> None:
    """At a matching rate the bytes reaching the device are byte-for-byte what the
    pre-ADR-0070 code wrote: audioop.mul over the source bytes, nothing else."""
    import audioop

    output = _opened()
    pcm = _sine(DEVICE_RATE, 4800)
    await output.playback(volume=50, sound=_sound(pcm, DEVICE_RATE))
    assert opened_streams[0].writes == [audioop.mul(pcm, 2, 0.5)]


async def test_volume_100_writes_the_source_bytes_untouched(
    opened_streams: list[_FakeDevice],
) -> None:
    output = _opened()
    pcm = _sine(DEVICE_RATE, 4800)
    await output.playback(volume=100, sound=_sound(pcm, DEVICE_RATE))
    assert opened_streams[0].writes == [pcm]


async def test_the_volume_is_applied_to_the_converted_audio_too(
    opened_streams: list[_FakeDevice],
) -> None:
    """Attenuation and resampling are both linear, so the order between them does not
    change the audio; what matters is that a converted utterance is still attenuated."""
    output = _opened()
    pcm = _sine(TTS_RATE, 2400)
    await output.playback(volume=50, sound=_sound(pcm, TTS_RATE))
    quiet = _i16(opened_streams[0].writes[0])
    loud = _i16(output.convert(pcm, TTS_RATE, SampleFormat.INT16, 1))
    assert quiet.size == loud.size
    assert 0.4 < np.max(np.abs(quiet)) / np.max(np.abs(loud)) < 0.6


# --- the worker loop --------------------------------------------------------------------


async def _play(
    queue: Queue[WorkerInput], count: int, config: PlaybackConfig | None = None
) -> list[Any]:
    gen = sd_playback_worker(
        config=config if config is not None else PlaybackConfig(output_device_index=0),
        telemetry_config=TelemetryConfig(),
        in_queue=queue,
    )
    try:
        # Through wait_for: the worker loops back to an empty queue whenever it swallows
        # an utterance, so a regression that swallows one has to fail these tests with a
        # timeout rather than hang the suite.
        return [await wait_for(anext(gen), timeout=5.0) for _ in range(count)]
    finally:
        await gen.aclose()


async def test_the_worker_plays_alternating_rates_through_one_open(
    opened_streams: list[_FakeDevice],
) -> None:
    """End to end: a 24000Hz TTS utterance and a 40000Hz VC one, back to back. The device
    is opened once, at its own rate, and both are converted into it."""
    queue: Queue[WorkerInput] = Queue()
    tts = _sine(TTS_RATE, 2400)
    vc = _sine(VC_RATE, 4000)
    queue.put_nowait(_speech(_sound(tts, TTS_RATE)))
    queue.put_nowait(_speech(_sound(vc, VC_RATE)))

    outputs = await _play(queue, 2)

    assert len(opened_streams) == 1
    assert opened_streams[0].kwargs["samplerate"] == DEVICE_RATE
    written = [_i16(w).size for w in opened_streams[0].writes]
    assert written == [4800, 4800]  # 100 ms each, at the device rate
    # The utterance travelling on to the following steps is the original, not the
    # device-rate conversion.
    assert [o.sound.rate for o in outputs] == [TTS_RATE, VC_RATE]
    assert outputs[0].sound.data == tts
    assert opened_streams[0].closed  # the finally block still closes it


async def test_the_worker_still_records_e2e_telemetry(
    opened_streams: list[_FakeDevice], enabled_telemetry
) -> None:
    """The bookkeeping around the write is untouched by the conversion."""
    queue: Queue[WorkerInput] = Queue()
    queue.put_nowait(_speech(_sound(_sine(TTS_RATE, 2400), TTS_RATE), origin_ts=1.0))
    with pytest.MonkeyPatch.context() as m:
        m.setattr(playback_mod, "time", lambda: 2.5)
        await _play(queue, 1)
    summary = enabled_telemetry.summary()
    assert summary["e2e"]["count"] == 1
    assert summary["e2e"]["max"] == pytest.approx(1.5)
    assert summary["playback"]["count"] == 1


async def test_a_device_fault_during_the_write_is_warned_about_and_the_loop_goes_on(
    stubbed_device_table: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unchanged from before ADR-0070: a runtime device fault is logged as a warning and
    the next utterance is attempted."""
    device = _FakeDevice(samplerate=DEVICE_RATE, fail_on_write=1)
    monkeypatch.setattr(playback_mod.sd, "RawOutputStream", lambda **kw: device)
    queue: Queue[WorkerInput] = Queue()
    for _ in range(2):
        queue.put_nowait(_speech(_sound(_sine(TTS_RATE, 2400), TTS_RATE)))

    with caplog.at_level(logging.WARNING):
        outputs = await _play(queue, 1)  # the first utterance faults, the second yields

    assert len(outputs) == 1
    assert len(device.writes) == 1  # only the second utterance got through
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert "output sink gone" in warnings[0]


async def test_a_pathological_utterance_rate_warns_and_the_next_utterance_plays(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    """The failure classification of a refused ratio, end to end.

    Before ADR-0070 the same value reached `sd.RawOutputStream(samplerate=...)`, came back
    as a PortAudioError, was warned about, and the next utterance played. That is exactly
    what must still happen now that the value reaches the filter design instead -- the
    worker must not die on one bad utterance from a remote peer, and must not spend 1.5s
    and 563MB on it either.
    """
    queue: Queue[WorkerInput] = Queue()
    queue.put_nowait(_speech(_sound(_sine(44101, 4410), 44101)))
    queue.put_nowait(_speech(_sound(_sine(TTS_RATE, 2400), TTS_RATE)))

    with caplog.at_level(logging.WARNING):
        outputs = await _play(queue, 1)

    assert len(outputs) == 1
    assert outputs[0].sound.rate == TTS_RATE  # the good one, not the refused one
    assert len(opened_streams) == 1
    assert [_i16(w).size for w in opened_streams[0].writes] == [4800]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "病的" in warnings[0]
    assert "44101" in warnings[0]


async def test_an_unresolvable_device_rate_fails_loud(
    opened_streams: list[_FakeDevice],
) -> None:
    """A rate that cannot be decided is a config problem, not a device fault: it must NOT
    be swallowed into the per-utterance warning, which would leave the pipeline silently
    playing nothing forever (ADR-0071, the same as the three other boundaries)."""
    queue: Queue[WorkerInput] = Queue()
    queue.put_nowait(_speech(_sound(_sine(TTS_RATE, 2400), TTS_RATE)))
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        await _play(queue, 1, config=PlaybackConfig(output_device_index=2))
    assert "playback.output_device_rate" in str(excinfo.value)
    assert opened_streams == []
