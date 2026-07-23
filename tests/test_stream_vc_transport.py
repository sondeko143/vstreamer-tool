from asyncio import Queue

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import InProcessTransport
from vspeech.stream_vc.transport import drop_oldest_put


def test_drop_oldest_put_keeps_newest():
    q: Queue[int] = Queue(maxsize=2)
    assert drop_oldest_put(q, 1) is True
    assert drop_oldest_put(q, 2) is True
    assert drop_oldest_put(q, 3) is False  # full -> drop 1, keep 3
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3


async def test_in_process_transport_send_recv_order():
    t = InProcessTransport(max_queued=4)
    for i in range(3):
        assert await t.send(StreamPacket("s", i, float(i), b"\x00\x00", 16000)) is True
    got = [(await t.recv()).seq for _ in range(3)]
    assert got == [0, 1, 2]
    assert t.dropped == 0


async def test_in_process_transport_drops_oldest_when_full():
    t = InProcessTransport(max_queued=2)
    assert await t.send(StreamPacket("s", 0, 0.0, b"", 16000)) is True
    assert await t.send(StreamPacket("s", 1, 0.0, b"", 16000)) is True
    assert await t.send(StreamPacket("s", 2, 0.0, b"", 16000)) is False
    assert t.dropped == 1
    assert (await t.recv()).seq == 1  # oldest (0) was dropped
    assert (await t.recv()).seq == 2
