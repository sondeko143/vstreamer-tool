import logging
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.telemetry import telemetry
from vspeech.stream_vc import playback as playback_mod
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.playback import OutputSink
from vspeech.stream_vc.playback import detect_gap
from vspeech.stream_vc.playback import open_stream_vc_output
from vspeech.stream_vc.playback import playback_loop
from vspeech.stream_vc.transport import Transport


def test_detect_gap_none_prev():
    assert detect_gap(None, 0) == 0


def test_detect_gap_contiguous():
    assert detect_gap(4, 5) == 0


def test_detect_gap_missing():
    assert detect_gap(4, 7) == 2  # 5, 6 missing


def test_detect_gap_reorder_or_dup_is_zero():
    assert detect_gap(7, 5) == 0  # out-of-order/dup -> not a forward gap


# --- Native-rate open + in-process conversion (ADR-0073 / ADR-0074) ------------------

_MME = 0
_WASAPI = 1
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
# The MME row lies about the rate (PortAudio hardcodes 44100 there) and truncates the
# name to 31 characters; the WASAPI row for the same endpoint carries the true mix rate.
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
]

DEVICE_RATE = 48000
PACKET_RATE = 16000


class _FakeDevice:
    """Stands in for sd.RawOutputStream: records the bytes written to the device.

    `fail_on_write` makes the n-th write raise, which is how a runtime device fault is
    reproduced (the loop's `(OSError, sd.PortAudioError)` handler).

    Like a real stream it always reports a `samplerate` -- the one it was opened at when
    it stands in for the opener's product, and the device rate when a test wraps it in an
    OutputSink directly (nothing reads it there).

    `latency` is what PortAudio granted, which is not required to equal what was
    requested -- a fixed number here so the log assertion can tell the two apart.
    """

    latency = 0.048

    def __init__(
        self, underflowed: bool = False, fail_on_write: int | None = None, **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        self.samplerate = float(kwargs.get("samplerate", DEVICE_RATE))
        self.writes: list[bytes] = []
        self.started = False
        self.closed = False
        self._underflowed = underflowed
        self._fail_on_write = fail_on_write

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> bool:
        if self._fail_on_write is not None and len(self.writes) + 1 == (
            self._fail_on_write
        ):
            raise OSError("output sink gone")
        self.writes.append(bytes(data))
        return self._underflowed

    def close(self) -> None:
        self.closed = True


def _sink(device_rate: int = DEVICE_RATE, **kwargs: Any) -> OutputSink:
    return OutputSink(_FakeDevice(**kwargs), device_rate)  # ty: ignore[invalid-argument-type]


def _sine(rate: int, samples: int, freq: float = 440.0) -> bytes:
    """int16 mono PCM of a sine, the shape a StreamPacket carries."""
    t = np.arange(samples, dtype=np.float64) / rate
    return np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16).tobytes()


def _i16(pcm: bytes) -> NDArray[np.int16]:
    return np.frombuffer(pcm, dtype=np.int16)


def _peak_frequency(pcm: bytes, rate: int) -> float:
    """The dominant frequency of `pcm` **read at `rate`**.

    Reading the converted bytes at the device rate is what makes this a test of the
    conversion: audio left at the packet rate peaks at 3x the input frequency here.
    """
    samples = _i16(pcm).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))])


def test_a_packet_at_the_device_rate_is_written_untouched() -> None:
    """No resampler in the path: the bytes handed to the device are the packet's own.

    Identity, not equality -- this path must stay bit-identical to the pre-ADR-0073 code,
    which wrote `packet.pcm` straight to the stream.
    """
    sink = _sink(DEVICE_RATE)
    pcm = _sine(DEVICE_RATE, 4800)
    assert sink.convert(pcm, DEVICE_RATE) is pcm
    assert sink.write(pcm, DEVICE_RATE) is False
    assert sink.stream.writes == [pcm]  # ty: ignore[unresolved-attribute]


def test_a_packet_at_another_rate_is_converted_to_the_device_rate() -> None:
    """16 kHz in, 48 kHz out: three times the samples, same 440 Hz tone."""
    sink = _sink(DEVICE_RATE)
    pcm = _sine(PACKET_RATE, 8000)
    out = sink.convert(pcm, PACKET_RATE)
    assert _i16(out).size == 8000 * DEVICE_RATE // PACKET_RATE
    assert abs(_peak_frequency(out, DEVICE_RATE) - 440.0) < 2.0
    # The same bytes read at the packet rate would be a 1320 Hz tone: the assertion above
    # only means something because this one fails.
    assert abs(_peak_frequency(out, PACKET_RATE) - 440.0) > 100.0


def test_the_filter_state_carries_across_packet_boundaries() -> None:
    """Packets are one continuous signal, so converting them one by one must equal
    converting the whole stream at once.

    A per-packet resampler (or a per-packet reset) would start each block from a zeroed
    filter tail, which fades in the first taps of every block -- an audible buzz at the
    block rate. Comparing against the one-shot conversion is the direct statement that no
    such seam exists.
    """
    whole = _sine(PACKET_RATE, 16000)
    packets = [_i16(whole)[i : i + 1600].tobytes() for i in range(0, 16000, 1600)]
    sink = _sink(DEVICE_RATE)
    piecewise = _i16(b"".join(sink.convert(p, PACKET_RATE) for p in packets))
    continuous = _i16(_sink(DEVICE_RATE).convert(whole, PACKET_RATE))
    assert piecewise.size == continuous.size
    # Not bit-equality: BLAS sums one large matvec in a different order than ten small
    # ones (the same effect tests/test_resample.py records for irregular blocks). One LSB
    # of int16 is 300 dB below a seam.
    assert np.max(np.abs(piecewise.astype(np.int32) - continuous.astype(np.int32))) <= 1


def test_reset_drops_the_filter_tail_without_touching_the_device() -> None:
    """A new sender session is a cut in the audio: the previous session's tail must not
    ring into the first block of the next one."""
    sink = _sink(DEVICE_RATE)
    loud = np.full(1600, 20000, dtype=np.int16).tobytes()
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    sink.convert(loud, PACKET_RATE)
    without_reset = _i16(sink.convert(silence, PACKET_RATE))
    assert np.max(np.abs(without_reset)) > 1000  # the tail really does ring on
    sink.reset()
    after_reset = _i16(sink.convert(silence, PACKET_RATE))
    assert not sink.stream.closed
    assert np.array_equal(after_reset, _i16(_sink().convert(silence, PACKET_RATE)))
    assert np.max(np.abs(after_reset)) == 0


def test_a_new_model_rate_rebuilds_the_resampler() -> None:
    """The sender's rate travels with every packet; when it changes the ratio must change
    with it rather than keep filtering at the old one."""
    sink = _sink(DEVICE_RATE)
    sink.convert(_sine(PACKET_RATE, 1600), PACKET_RATE)
    pcm = _sine(24000, 2400)
    switched = sink.convert(pcm, 24000)
    assert _i16(switched).size == 2400 * DEVICE_RATE // 24000  # 2x, no longer 3x
    assert switched == _sink().convert(pcm, 24000)


def test_a_rejected_rate_leaves_the_sink_unable_to_pass_audio_through() -> None:
    """A rate the resampler refuses (only reachable from a corrupt packet) must not be
    remembered as the current one.

    Recording it before the build succeeded would make the next packet at that rate take
    the "rates already match" branch and play unconverted -- silently at the wrong speed,
    instead of failing.
    """
    sink = _sink(DEVICE_RATE)
    pcm = _sine(PACKET_RATE, 160)
    for _ in range(2):
        with pytest.raises(ValueError):
            sink.convert(pcm, 0)


def test_the_conversion_saturates_instead_of_wrapping() -> None:
    """Resampling a full-scale square wave overshoots past +1.0 (Gibbs). A wrapping cast
    would sign-flip those samples into a loud click."""
    square = (np.tile([1, 1, 1, 1, -1, -1, -1, -1], 200) * 32767).astype(np.int16)
    pcm = square.tobytes()
    reference = PolyphaseResampler(PACKET_RATE, DEVICE_RATE).process(
        square.astype(np.float32) / 32768.0
    )
    overshoot = np.flatnonzero(reference > 1.0)
    assert overshoot.size > 0, "this signal must overshoot or the test proves nothing"
    out = _i16(_sink().convert(pcm, PACKET_RATE))
    assert np.array_equal(
        out[overshoot], np.full(overshoot.size, 32767, dtype=np.int16)
    )
    assert out.max() <= 32767
    assert out.min() >= -32768


@pytest.fixture
def opened_streams(monkeypatch: pytest.MonkeyPatch) -> list[_FakeDevice]:
    """Stub the device table and sd.RawOutputStream; yield the streams that got opened."""
    import vspeech.lib.audio as audio

    def _query_devices(index: int | None = None):
        if index is None:
            return _DEVICES
        return next(d for d in _DEVICES if d["index"] == index)

    monkeypatch.setattr(audio.sd, "query_devices", _query_devices)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _HOSTAPIS)
    opened: list[_FakeDevice] = []

    def _open(**kwargs: Any) -> _FakeDevice:
        stream = _FakeDevice(**kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(playback_mod.sd, "RawOutputStream", _open)
    return opened


def _open_log(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        r.getMessage() for r in caplog.records if "output device" in r.getMessage()
    ]
    assert len(lines) == 1, lines
    return lines[0]


def test_output_device_is_opened_at_the_resolved_native_rate(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    """The endpoint really runs at 48000, so that is what is opened -- whatever rate the
    packets happen to carry."""
    with caplog.at_level(logging.INFO):
        sink = open_stream_vc_output(StreamVcConfig(output_device_index=0))
    assert sink.device_rate == 48000
    assert sink.stream is opened_streams[0]
    assert opened_streams[0].kwargs["samplerate"] == 48000
    assert opened_streams[0].kwargs["channels"] == 1
    assert opened_streams[0].kwargs["dtype"] == "int16"
    assert opened_streams[0].started
    line = _open_log(caplog)
    assert "48000Hz" in line
    assert "WASAPI" in line


def test_configured_output_device_rate_wins_over_the_resolved_one(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        sink = open_stream_vc_output(
            StreamVcConfig(output_device_index=0, output_device_rate=44100)
        )
    assert sink.device_rate == 44100
    assert opened_streams[0].kwargs["samplerate"] == 44100
    assert "stream_vc.output_device_rate" in _open_log(caplog)


def test_a_device_reporting_another_rate_is_warned_about(
    monkeypatch: pytest.MonkeyPatch,
    opened_streams: list[_FakeDevice],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The conversion keeps using the requested rate (a reported 47999 would blow the
    polyphase ratio up to 48000 phases), so a hardware rate that differs is a slow drift
    in the audio and nothing else. This warning is its only trace."""

    def _open(**kwargs: Any) -> _FakeDevice:
        stream = _FakeDevice(**kwargs)
        stream.samplerate = 47999.0
        return stream

    monkeypatch.setattr(playback_mod.sd, "RawOutputStream", _open)
    with caplog.at_level(logging.WARNING):
        sink = open_stream_vc_output(StreamVcConfig(output_device_index=0))
    assert sink.device_rate == 48000  # the requested rate, not the reported one
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "47999" in warnings[0]
    assert "48000" in warnings[0]


# --- Requested / granted device latency (ADR-0071) -----------------------------------
#
# The rate is decided by the shared opener above; the latency is the one part of the
# stream's shape this boundary still decides for itself, so it is asserted at the sd
# boundary -- `latency` is a passthrough and there is nothing downstream to observe it on.


def test_open_output_stream_requests_configured_latency(
    opened_streams: list[_FakeDevice],
) -> None:
    """The configured value reaches sounddevice unconverted (ADR-0071)."""
    open_stream_vc_output(StreamVcConfig(output_device_index=0, output_latency="high"))
    assert opened_streams[0].kwargs["latency"] == "high"


def test_open_output_stream_defaults_to_low(
    opened_streams: list[_FakeDevice],
) -> None:
    """No setting = the value that used to be hardcoded."""
    open_stream_vc_output(StreamVcConfig(output_device_index=0))
    assert opened_streams[0].kwargs["latency"] == "low"


def test_open_output_stream_uses_output_latency_not_input(
    opened_streams: list[_FakeDevice],
) -> None:
    """The input setting must not leak into the output stream."""
    config = StreamVcConfig(
        output_device_index=0, input_latency="high", output_latency=0.02
    )
    open_stream_vc_output(config)
    assert opened_streams[0].kwargs["latency"] == 0.02


def test_open_output_stream_logs_requested_and_granted_latency(
    opened_streams: list[_FakeDevice], caplog: pytest.LogCaptureFixture
) -> None:
    """consumer.py reuses this function, so the consumer machine gets the same line."""
    with caplog.at_level(logging.INFO):
        open_stream_vc_output(StreamVcConfig(output_device_index=0))
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Speakers" in messages  # the device line still names the device
    assert "low" in messages  # requested
    assert "0.048" in messages  # granted


# --- playback_loop ------------------------------------------------------------------


class _EndOfTest(Exception):
    """Not a device error, so it ends the loop instead of triggering a reopen."""


class _ScriptedTransport(Transport):
    """Hands out `packets` one per recv, then raises _EndOfTest to end the loop.

    With `burst=True` everything that has not been recv'd yet counts as "already waiting
    in the queue", which is what `drain_to_latest` folds forward -- the stale-drop path.
    Left paced (the default), nothing is ever waiting and that path stays out of the way.
    """

    def __init__(self, packets: list[StreamPacket], burst: bool = False) -> None:
        self._packets = list(packets)
        self._burst = burst

    async def send(self, packet: StreamPacket) -> bool:  # unused
        raise NotImplementedError

    async def recv(self) -> StreamPacket:
        if not self._packets:
            raise _EndOfTest
        return self._packets.pop(0)

    def drain_to_latest(self, keep: int = 1) -> list[StreamPacket]:
        if not self._burst:
            return []
        dropped: list[StreamPacket] = []
        while len(self._packets) > keep:
            dropped.append(self._packets.pop(0))
        return dropped


def _packets(
    count: int, rate: int = PACKET_RATE, seqs: list[int] | None = None
) -> list[StreamPacket]:
    """`count` contiguous 100 ms packets of one continuous sine at `rate`."""
    per_packet = rate // 10
    whole = _i16(_sine(rate, per_packet * count))
    return [
        StreamPacket(
            session_id="ab" * 16,
            seq=seqs[i] if seqs else i,
            pts=0.1 * i,
            pcm=whole[i * per_packet : (i + 1) * per_packet].tobytes(),
            sample_rate=rate,
        )
        for i in range(count)
    ]


@pytest.fixture
def enabled_telemetry():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=1000)
    yield telemetry
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


def _patch_open(monkeypatch: pytest.MonkeyPatch, *devices: _FakeDevice) -> list[Any]:
    """Make the loop's opener hand out `devices` in turn; return the opened sinks."""
    sinks: list[Any] = []
    remaining = list(devices)

    def _open(config: StreamVcConfig) -> OutputSink:
        sink = OutputSink(remaining.pop(0), DEVICE_RATE)  # ty: ignore[invalid-argument-type]
        sinks.append(sink)
        return sink

    monkeypatch.setattr(playback_mod, "open_stream_vc_output", _open)
    return sinks


async def test_playback_loop_opens_at_the_device_rate_and_converts_the_packets(
    opened_streams: list[_FakeDevice],
) -> None:
    """End to end: the device table says 48000, the packets say 16000, and the open goes
    with the device (ADR-0073). Opening at packet.sample_rate is what this replaces."""
    packets = _packets(4)
    with pytest.raises(_EndOfTest):
        await playback_loop(
            StreamVcConfig(output_device_index=0), _ScriptedTransport(packets)
        )
    assert len(opened_streams) == 1  # one open for the whole run
    assert opened_streams[0].kwargs["samplerate"] == DEVICE_RATE
    written = opened_streams[0].writes
    assert [len(w) // 2 for w in written] == [
        len(p.pcm) // 2 * DEVICE_RATE // PACKET_RATE for p in packets
    ]
    # The device was fed the same continuous audio a single conversion would produce.
    reference = _i16(_sink().convert(b"".join(p.pcm for p in packets), PACKET_RATE))
    played = _i16(b"".join(written))
    assert played.size == reference.size
    assert np.max(np.abs(played.astype(np.int32) - reference.astype(np.int32))) <= 1
    assert opened_streams[0].closed  # the finally block still closes it


async def test_playback_loop_keeps_gap_drop_and_underflow_telemetry(
    monkeypatch: pytest.MonkeyPatch, enabled_telemetry, caplog: pytest.LogCaptureFixture
) -> None:
    """The bookkeeping around the write is unchanged by the conversion: a seq jump, the
    stale packets folded away and the device's underflow flag are all still recorded, and
    the warnings are still thinned to one line per episode (ADR-0062)."""
    device = _FakeDevice(underflowed=True)
    _patch_open(monkeypatch, device)
    # seq 0 then 3 = two missing; burst = everything but the newest is dropped as stale.
    packets = _packets(4, seqs=[0, 3, 4, 5])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_EndOfTest):
            await playback_loop(
                StreamVcConfig(), _ScriptedTransport(packets, burst=True)
            )
    summary = enabled_telemetry.summary()
    # recv'd seq 0 + the stale seq 3 and 4; seq 5 is the one that survives and plays.
    assert summary["stream_vc_playback_drop"]["count"] == 3
    assert summary["stream_vc_gap"]["count"] == 1  # 0 -> 3 (the only forward jump)
    assert summary["stream_vc_gap"]["max"] == 2.0
    assert summary["stream_vc_playback_underflow"]["count"] == 1
    assert len(device.writes) == 1
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len([m for m in messages if "dropped stale packet" in m]) == 1
    assert len([m for m in messages if "playback gap" in m]) == 1
    assert len([m for m in messages if "output underflow" in m]) == 1


async def test_playback_loop_reopens_after_a_device_fault(
    monkeypatch: pytest.MonkeyPatch, enabled_telemetry
) -> None:
    """A runtime fault closes the sink, backs off and lazily reopens on the next packet
    (ADR-0050) -- the same shape as before, now carrying the resampler with it."""
    slept: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(playback_mod, "sleep", _no_sleep)
    faulting = _FakeDevice(fail_on_write=2)
    healthy = _FakeDevice()
    sinks = _patch_open(monkeypatch, faulting, healthy)
    with pytest.raises(_EndOfTest):
        await playback_loop(StreamVcConfig(), _ScriptedTransport(_packets(4)))
    assert len(sinks) == 2  # opened, faulted, reopened
    assert slept == [playback_mod.BACKOFF_START]
    assert faulting.closed
    assert len(faulting.writes) == 1  # packet 0; packet 1 faulted and was not retried
    assert len(healthy.writes) == 2  # packets 2 and 3
    assert enabled_telemetry.summary()["stream_vc_playback_reopen"]["count"] == 1
