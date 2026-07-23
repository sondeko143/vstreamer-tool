"""ストリーミング VC の生 UDP transport(ADR-0051 T3)。

producer は 1 ブロック 1 データグラムを送るだけ、consumer は受信を Queue に積み
recv/poll で出す。並べ替え・穴埋め・遅延上限は持たない(それは JitterBuffer =
ADR-0056)。asyncio の datagram endpoint を使うので追加依存は無い。
"""

from __future__ import annotations

from asyncio import DatagramProtocol
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import get_running_loop
from typing import Any

from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport
from vspeech.stream_vc.transport import drop_oldest_put
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet


class _SendProtocol(DatagramProtocol):
    """producer 送信専用エンドポイントのプロトコル。

    UDP の送信失敗(到達不可・route 消失・ICMP port-unreachable 等)は asyncio では
    sendto() が同期例外を投げず、後から error_received へ非同期に届く(Windows Proactor /
    Selector 双方)。ここで捕えてログ+telemetry に通し、送信側の失敗が黙って消えない
    ようにする(silent な無音穴を作らない)。非同期なので特定 packet には紐付かない。
    """

    def __init__(self) -> None:
        self.error_count = 0

    def error_received(self, exc: Exception) -> None:
        self.error_count += 1
        telemetry.record("stream_vc_send_error", 1.0)
        logger.warning("stream_vc udp send error (async): %r", exc)


class UdpProducerTransport(Transport):
    def __init__(self, transport: Any, protocol: _SendProtocol) -> None:
        self._transport = transport
        self._protocol = protocol

    async def send(self, packet: StreamPacket) -> bool:
        # sendto は非同期な送信失敗(到達不可等)を同期例外にしない — それらは
        # _SendProtocol.error_received でログ+telemetry される。ここで捕える OSError は
        # 稀な同期失敗(message too long 等)のみで、その場合は send_drop として False。
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
    """datagram を decode して Queue へ。満杯なら最古を捨てる(遅延の張り付き防止)。"""

    def __init__(self, queue: Queue[StreamPacket]) -> None:
        self._queue = queue

    def connection_made(self, transport: Any) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, _addr: Any) -> None:
        try:
            packet = decode_packet(data)
        except WireError as e:
            logger.warning("stream_vc udp: dropping malformed datagram: %r", e)
            return
        if not drop_oldest_put(self._queue, packet):
            telemetry.record("stream_vc_recv_drop", 1.0)

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
    transport, protocol = await loop.create_datagram_endpoint(
        _SendProtocol, remote_addr=(peer_host, peer_port)
    )
    return UdpProducerTransport(transport, protocol)


async def create_udp_consumer_transport(
    bind_host: str, bind_port: int, max_queued: int
) -> UdpConsumerTransport:
    loop = get_running_loop()
    queue: Queue[StreamPacket] = Queue(maxsize=max_queued)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RecvProtocol(queue), local_addr=(bind_host, bind_port)
    )
    return UdpConsumerTransport(transport, queue)
