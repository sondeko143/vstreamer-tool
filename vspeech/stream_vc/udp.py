"""ストリーミング VC の生 UDP transport(ADR-0051 T3)。

producer は 1 ブロック 1 データグラムを送るだけ、consumer は受信を Queue に積み
recv/poll で出す。並べ替え・穴埋め・遅延上限は持たない(それは JitterBuffer =
ADR-0056)。asyncio の datagram endpoint を使うので追加依存は無い。
"""

from __future__ import annotations

from asyncio import DatagramProtocol
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull
from asyncio import get_running_loop
from typing import Any

from vspeech.logger import logger
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet


class UdpProducerTransport(Transport):
    def __init__(self, transport: Any) -> None:
        self._transport = transport

    async def send(self, packet: StreamPacket) -> bool:
        try:
            self._transport.sendto(encode_packet(packet))
            return True
        except OSError as e:  # socket buffer full / route gone: drop, don't crash
            logger.warning("stream_vc udp send failed; dropping packet: %r", e)
            return False

    async def recv(self) -> StreamPacket:
        raise NotImplementedError("producer transport does not receive")

    def close(self) -> None:
        self._transport.close()


class _RecvProtocol:
    """datagram を decode して Queue へ。満杯なら最古を捨てる(遅延の張り付き防止)。"""

    def __init__(self, queue: Queue[StreamPacket]) -> None:
        self._queue = queue

    def connection_made(self, transport: Any) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: Any) -> None:
        try:
            packet = decode_packet(data)
        except WireError as e:
            logger.warning("stream_vc udp: dropping malformed datagram: %r", e)
            return
        try:
            self._queue.put_nowait(packet)
        except QueueFull:
            try:
                self._queue.get_nowait()
            except QueueEmpty:
                pass
            self._queue.put_nowait(packet)

    def error_received(self, exc: Exception) -> None:
        logger.warning("stream_vc udp recv error: %r", exc)

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
    loop = get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        DatagramProtocol,  # producer never receives; base no-op protocol
        remote_addr=(peer_host, peer_port),
    )
    return UdpProducerTransport(transport)


async def create_udp_consumer_transport(
    bind_host: str, bind_port: int, max_queued: int
) -> UdpConsumerTransport:
    loop = get_running_loop()
    queue: Queue[StreamPacket] = Queue(maxsize=max_queued)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RecvProtocol(queue), local_addr=(bind_host, bind_port)
    )
    return UdpConsumerTransport(transport, queue)
