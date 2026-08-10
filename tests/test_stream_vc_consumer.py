from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.lib.telemetry import telemetry
from vspeech.stream_vc import consumer as consumer_mod
from vspeech.stream_vc.consumer import consume_into_buffer
from vspeech.stream_vc.consumer import network_playback_loop
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.playback import OutputSink
from vspeech.stream_vc.transport import Transport


class _FakeTransport(Transport):
    def __init__(self, queued):
        self._queued = list(queued)

    async def send(self, packet):  # unused
        raise NotImplementedError

    async def recv(self):
        return self._queued.pop(0)

    def poll(self):
        out, self._queued = self._queued, []
        return out


def _pkt(seq, session="cd" * 16):
    return StreamPacket(
        session_id=session,
        seq=seq,
        pts=0.0,
        pcm=bytes([seq % 256]) * 4,
        sample_rate=16000,
    )


async def test_consume_into_buffer_drains_recv_and_poll():
    t = _FakeTransport([_pkt(0), _pkt(1), _pkt(2)])
    buf = JitterBuffer(target_depth=0)
    first = await t.recv()
    consume_into_buffer(t, buf, first, first.session_id)
    assert buf.depth == 3  # first(0) + poll(1,2)
    assert buf.pop().kind is PopKind.NORMAL


async def test_consume_into_buffer_skips_foreign_session_poll_packets():
    # poll batch mixes current-session (0,1) and a stale prior-session packet;
    # only the current-session ones are pushed.
    t = _FakeTransport([_pkt(0), _pkt(1), _pkt(99, session="ff" * 16)])
    buf = JitterBuffer(target_depth=0)
    first = await t.recv()
    consume_into_buffer(t, buf, first, first.session_id)
    assert buf.depth == 2  # first(0) + poll(1); foreign-session 99 skipped


# --- The output boundary: one device rate, converted per packet (ADR-0073) -----------

DEVICE_RATE = 48000
PACKET_RATE = 16000
_SESSION_A = "ab" * 16
_SESSION_B = "cd" * 16


class _EndOfTest(Exception):
    """Not a device error, so it ends the loop instead of triggering a reopen."""


class _PacedTransport(Transport):
    """One packet per recv and nothing waiting in between (a paced wired LAN).

    `poll` stays the base no-op, so every packet goes through the jitter buffer on its
    own recv -- which is what lets a test lay out gaps and session changes in order.
    """

    def __init__(self, packets: list[StreamPacket]) -> None:
        self._packets = list(packets)

    async def send(self, packet: StreamPacket) -> bool:  # unused
        raise NotImplementedError

    async def recv(self) -> StreamPacket:
        if not self._packets:
            raise _EndOfTest
        return self._packets.pop(0)


class _FakeDevice:
    """Stands in for sd.RawOutputStream: records the bytes written to the device."""

    def __init__(self, fail_on_write: int | None = None) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self._fail_on_write = fail_on_write

    def write(self, data: bytes) -> bool:
        if self._fail_on_write is not None and len(self.writes) + 1 == (
            self._fail_on_write
        ):
            raise OSError("output sink gone")
        self.writes.append(bytes(data))
        return False

    def close(self) -> None:
        self.closed = True


def _i16(pcm: bytes) -> NDArray[np.int16]:
    return np.frombuffer(pcm, dtype=np.int16)


def _audio_packets(
    count: int,
    session: str,
    rate: int = PACKET_RATE,
    seqs: list[int] | None = None,
    freq: float = 440.0,
) -> list[StreamPacket]:
    """`count` contiguous 100 ms packets of one continuous sine."""
    per_packet = rate // 10
    t = np.arange(per_packet * count, dtype=np.float64) / rate
    whole = np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16)
    return [
        StreamPacket(
            session_id=session,
            seq=seqs[i] if seqs else i,
            pts=0.1 * i,
            pcm=whole[i * per_packet : (i + 1) * per_packet].tobytes(),
            sample_rate=rate,
        )
        for i in range(count)
    ]


def _fresh_sink() -> OutputSink:
    return OutputSink(_FakeDevice(), DEVICE_RATE)  # ty: ignore[invalid-argument-type]


def _converted(packets: list[StreamPacket]) -> list[bytes]:
    """What a sink that has seen nothing else would write for `packets`."""
    sink = _fresh_sink()
    return [sink.convert(p.pcm, p.sample_rate) for p in packets]


def _patch_open(monkeypatch: pytest.MonkeyPatch, *devices: _FakeDevice) -> list[Any]:
    """Make the loop's opener hand out `devices` in turn; return the opened sinks."""
    remaining = list(devices)
    sinks: list[Any] = []

    def _open(config: StreamVcConfig) -> OutputSink:
        sink = OutputSink(remaining.pop(0), DEVICE_RATE)  # ty: ignore[invalid-argument-type]
        sinks.append(sink)
        return sink

    monkeypatch.setattr(consumer_mod, "open_stream_vc_output", _open)
    return sinks


@pytest.fixture
def enabled_telemetry():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=1000)
    yield telemetry
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


async def test_a_new_sender_session_resets_the_resampler_but_not_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device is open at its own rate, which no sender can change, so a producer
    restart must no longer cost a device reopen -- only the filter tail is discontinuous.

    That the tail really was dropped is checked by value: the second session's blocks
    equal what a sink that had never seen the first session would produce.
    """
    device = _FakeDevice()
    sinks = _patch_open(monkeypatch, device)
    first = _audio_packets(3, _SESSION_A)
    second = _audio_packets(3, _SESSION_B, freq=880.0)
    with pytest.raises(_EndOfTest):
        await network_playback_loop(StreamVcConfig(), _PacedTransport(first + second))
    assert len(sinks) == 1  # one open for both sessions
    assert device.writes[:3] == _converted(first)
    assert device.writes[3:] == _converted(second)


async def test_a_new_model_rate_rebuilds_the_resampler_without_reopening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer that restarts with another model sends another sample_rate. The ratio
    has to follow it; the device (and its rate) does not move."""
    device = _FakeDevice()
    sinks = _patch_open(monkeypatch, device)
    first = _audio_packets(2, _SESSION_A, rate=PACKET_RATE)
    second = _audio_packets(2, _SESSION_B, rate=24000)
    with pytest.raises(_EndOfTest):
        await network_playback_loop(StreamVcConfig(), _PacedTransport(first + second))
    assert len(sinks) == 1
    assert [_i16(w).size for w in device.writes[:2]] == [4800, 4800]  # 1600 * 3
    assert [_i16(w).size for w in device.writes[2:]] == [4800, 4800]  # 2400 * 2
    assert device.writes[2:] == _converted(second)


async def test_a_concealed_block_is_converted_like_a_real_one(
    monkeypatch: pytest.MonkeyPatch, enabled_telemetry
) -> None:
    """Concealment blocks are sized from inside the jitter buffer, not from
    `packet.sample_rate`. The two agree -- the buffer settles its block size from the
    first packet of the current session, whose rate is exactly the one the loop converts
    with -- so a concealed block occupies the same amount of device time as a real one.
    """
    device = _FakeDevice()
    _patch_open(monkeypatch, device)
    # seq 1 never arrives: pop 0 plays, pop 1 conceals (2 is already buffered), pop 2
    # plays.
    packets = _audio_packets(3, _SESSION_A, seqs=[0, 2, 3])
    with pytest.raises(_EndOfTest):
        await network_playback_loop(StreamVcConfig(), _PacedTransport(packets))
    assert enabled_telemetry.summary()["stream_vc_conceal"]["count"] == 1
    assert enabled_telemetry.summary()["stream_vc_gap"]["count"] == 1
    assert [len(w) for w in device.writes] == [len(packets[0].pcm) * 3] * 3


async def test_prebuffer_blocks_are_silence_at_the_device_rate(
    monkeypatch: pytest.MonkeyPatch, enabled_telemetry
) -> None:
    """With a jitter buffer configured, the first pops are the buffer's own silence.

    Those blocks come from `_block_bytes`, the same internal size a concealment uses, so
    they ride the same conversion at `packet.sample_rate`. A prebuffer block has to occupy
    exactly as much device time as a real one, or playback starts out of step with the
    sender by however much the prebuffer was worth.
    """
    device = _FakeDevice()
    _patch_open(monkeypatch, device)
    packets = _audio_packets(4, _SESSION_A)
    with pytest.raises(_EndOfTest):
        # 320 / block_ms 160 = 2 blocks deep: the first two pops prebuffer, then the
        # third arrival primes the buffer and packets 0 and 1 play.
        await network_playback_loop(
            StreamVcConfig(jitter_buffer_ms=320.0), _PacedTransport(packets)
        )
    assert [len(w) for w in device.writes] == [len(packets[0].pcm) * 3] * 4
    assert all(sample == 0 for w in device.writes[:2] for sample in _i16(w))
    assert any(sample != 0 for sample in _i16(device.writes[2]))  # audio, not silence
    assert "stream_vc_conceal" not in enabled_telemetry.summary()


def test_the_conceal_block_follows_the_session_it_is_covering() -> None:
    """The buffer's own block size is re-settled by the session change, so it cannot go
    on emitting the previous session's block length (which a different model rate would
    make the wrong duration)."""
    buffer = JitterBuffer(target_depth=0)
    buffer.push(_pkt(0))  # 4-byte blocks
    assert len(buffer.pop().pcm) == 4
    buffer.reset()
    long_block = StreamPacket(
        session_id=_SESSION_B, seq=0, pts=0.0, pcm=b"\x01\x02" * 240, sample_rate=24000
    )
    buffer.push(long_block)
    assert buffer.pop().kind is PopKind.NORMAL
    buffer.push(
        StreamPacket(
            session_id=_SESSION_B,
            seq=2,
            pts=0.0,
            pcm=long_block.pcm,
            sample_rate=24000,
        )
    )
    concealed = buffer.pop()
    assert concealed.kind is PopKind.CONCEAL
    assert len(concealed.pcm) == len(long_block.pcm)


async def test_consumer_reopens_after_a_device_fault(
    monkeypatch: pytest.MonkeyPatch, enabled_telemetry
) -> None:
    """A runtime output fault self-heals inside the subsystem exactly as before: close,
    back off, reopen lazily on the next packet (ADR-0050)."""
    slept: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(consumer_mod, "sleep", _no_sleep)
    faulting = _FakeDevice(fail_on_write=2)
    healthy = _FakeDevice()
    sinks = _patch_open(monkeypatch, faulting, healthy)
    with pytest.raises(_EndOfTest):
        await network_playback_loop(
            StreamVcConfig(), _PacedTransport(_audio_packets(4, _SESSION_A))
        )
    assert len(sinks) == 2
    assert slept == [consumer_mod.BACKOFF_START]
    assert faulting.closed
    assert len(faulting.writes) == 1
    assert len(healthy.writes) == 2
    assert enabled_telemetry.summary()["stream_vc_playback_reopen"]["count"] == 1
