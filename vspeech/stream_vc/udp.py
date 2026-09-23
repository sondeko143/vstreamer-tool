"""The raw UDP transport of streaming VC (ADR-0051 T3).

The producer just sends one datagram per block; the consumer pushes what it receives
onto a Queue and hands it out through recv/poll. It holds no reordering, concealment or
delay ceiling (that is the JitterBuffer = ADR-0056). It uses asyncio's datagram
endpoint, so there is no extra dependency.
"""

from __future__ import annotations

from asyncio import DatagramProtocol
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import get_running_loop
from time import perf_counter
from typing import Any

from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport
from vspeech.stream_vc.transport import drop_oldest_put
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet

# A send error streak this long means the socket itself is dead, so it is replaced
# (ADR-0094). At 160ms blocks that is ~6 consecutive failures.
REOPEN_AFTER_S = 1.0
# No send error for this long ends the streak.
ERROR_STREAK_GAP_S = 2.0


class _SendProtocol(DatagramProtocol):
    """The protocol of the producer's send-only endpoint.

    In asyncio a UDP send failure (unreachable peer, lost route, ICMP port-unreachable,
    ...) does not raise synchronously from sendto(); it arrives asynchronously at
    error_received later (on both the Windows Proactor and Selector loops). Catch it here
    and route it to the log and telemetry so a send-side failure never disappears
    silently (never punch a silent hole). Being asynchronous it is not tied to a
    particular packet.
    """

    def __init__(self) -> None:
        # UDP protocol callbacks can fire at packet rate (one ICMP per datagram when the
        # peer is down). Throttle the log by time and record telemetry every time
        # (ADR-0062).
        self._error_throttle = LogThrottle()
        self._streak_start: float | None = None
        self._last_error: float | None = None

    def error_received(self, exc: Exception) -> None:
        telemetry.record("stream_vc_send_error", 1.0)
        now = perf_counter()
        if self._last_error is None or now - self._last_error > ERROR_STREAK_GAP_S:
            self._streak_start = now
        self._last_error = now
        if (n := self._error_throttle.hit()) is not None:
            logger.warning("stream_vc udp send error (async, total %d): %r", n, exc)

    def failing_for(self, now: float) -> float:
        """How long send errors have kept arriving without a gap, or 0 if they
        stopped."""
        if self._last_error is None or self._streak_start is None:
            return 0.0
        if now - self._last_error > ERROR_STREAK_GAP_S:
            return 0.0
        return now - self._streak_start

    def restart_streak(self, now: float) -> None:
        """Count the current streak from `now`, so a failed reopen is retried only
        after another REOPEN_AFTER_S rather than on every block."""
        if self._streak_start is not None:
            self._streak_start = now


async def _open_send_endpoint(
    peer_host: str, peer_port: int
) -> tuple[Any, _SendProtocol]:
    loop = get_running_loop()
    return await loop.create_datagram_endpoint(
        _SendProtocol, remote_addr=(peer_host, peer_port)
    )


class UdpProducerTransport(Transport):
    def __init__(
        self, transport: Any, protocol: _SendProtocol, peer: tuple[str, int]
    ) -> None:
        self._transport = transport
        self._protocol = protocol
        self._peer = peer
        self._reopen_throttle = LogThrottle()

    async def _reopen(self, failing_s: float) -> None:
        # On Windows a connected UDP socket caches its route at connect(). After the
        # link under that route bounces, every send on it fails with WinError 1168
        # even once the link is back, while a new socket works (ADR-0094).
        now = perf_counter()
        try:
            transport, protocol = await _open_send_endpoint(*self._peer)
        except OSError as e:
            self._protocol.restart_streak(now)
            if (n := self._reopen_throttle.hit()) is not None:
                logger.warning(
                    "stream_vc udp: reopening the send socket failed (total %d): %r",
                    n,
                    e,
                )
            return
        self._transport.close()
        self._transport, self._protocol = transport, protocol
        telemetry.record("stream_vc_send_reopen", 1.0)
        if (n := self._reopen_throttle.hit()) is not None:
            logger.warning(
                "stream_vc udp: send errors persisted %.1fs; reopened the send socket"
                " (total %d)",
                failing_s,
                n,
            )

    async def send(self, packet: StreamPacket) -> bool:
        if (failing_s := self._protocol.failing_for(perf_counter())) >= REOPEN_AFTER_S:
            await self._reopen(failing_s)
        # sendto does not turn asynchronous send failures (unreachable peer, ...) into
        # synchronous exceptions -- those are logged and recorded by
        # _SendProtocol.error_received. The OSError caught here is only the rare
        # synchronous failure (message too long, ...), which returns False as a send_drop.
        try:
            self._transport.sendto(encode_packet(packet))
            return True
        except OSError as e:
            logger.warning("stream_vc udp send failed synchronously; dropping: %r", e)
            return False

    async def recv(self) -> StreamPacket:
        raise NotImplementedError("producer transport does not receive")

    def close(self) -> None:
        self._transport.close()


class _RecvProtocol:
    """Decode a datagram onto the Queue. When full, drop the oldest (so latency does not
    stick)."""

    def __init__(self, queue: Queue[StreamPacket]) -> None:
        self._queue = queue
        # Time-based thinning for the same reason as _SendProtocol (ADR-0062).
        self._malformed_throttle = LogThrottle()
        self._error_throttle = LogThrottle()

    def connection_made(self, transport: Any) -> None:
        self._transport = transport

    # asyncio drives protocols by duck typing, so subclassing DatagramProtocol is not
    # needed. The parameter name `_addr` is kept to match vulture's unused-arg convention
    # (from the existing fix pass 2).
    def datagram_received(self, data: bytes, _addr: Any) -> None:
        try:
            packet = decode_packet(data)
        except WireError as e:
            telemetry.record("stream_vc_malformed_drop", 1.0)
            if (n := self._malformed_throttle.hit()) is not None:
                logger.warning(
                    "stream_vc udp: dropping malformed datagram (total %d): %r",
                    n,
                    e,
                )
            return
        if not drop_oldest_put(self._queue, packet):
            telemetry.record("stream_vc_recv_drop", 1.0)

    def error_received(self, exc: Exception) -> None:
        telemetry.record("stream_vc_recv_error", 1.0)
        if (n := self._error_throttle.hit()) is not None:
            logger.warning("stream_vc udp recv error (total %d): %r", n, exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            logger.warning("stream_vc udp connection lost: %r", exc)


class UdpConsumerTransport(Transport):
    def __init__(self, transport: Any, queue: Queue[StreamPacket]) -> None:
        self._transport = transport
        self._queue = queue

    @property
    def local_port(self) -> int:
        return int(self._transport.get_extra_info("sockname")[1])

    async def send(self, packet: StreamPacket) -> bool:
        raise NotImplementedError("consumer transport does not send")

    async def recv(self) -> StreamPacket:
        return await self._queue.get()

    def poll(self) -> list[StreamPacket]:
        out: list[StreamPacket] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except QueueEmpty:
                return out

    def close(self) -> None:
        self._transport.close()


async def create_udp_producer_transport(
    peer_host: str, peer_port: int
) -> UdpProducerTransport:
    transport, protocol = await _open_send_endpoint(peer_host, peer_port)
    return UdpProducerTransport(transport, protocol, (peer_host, peer_port))


async def create_udp_consumer_transport(
    bind_host: str, bind_port: int, max_queued: int
) -> UdpConsumerTransport:
    loop = get_running_loop()
    queue: Queue[StreamPacket] = Queue(maxsize=max_queued)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RecvProtocol(queue), local_addr=(bind_host, bind_port)
    )
    return UdpConsumerTransport(transport, queue)
