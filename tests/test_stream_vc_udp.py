import asyncio

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.udp import create_udp_consumer_transport
from vspeech.stream_vc.udp import create_udp_producer_transport


def _pkt(seq):
    return StreamPacket(
        session_id="ab" * 16,
        seq=seq,
        pts=float(seq),
        pcm=bytes([seq % 256]) * 320,
        sample_rate=16000,
    )


async def test_producer_to_consumer_loopback():
    consumer = await create_udp_consumer_transport("127.0.0.1", 0, max_queued=8)
    port = consumer.local_port
    producer = await create_udp_producer_transport("127.0.0.1", port)
    try:
        assert await producer.send(_pkt(0)) is True
        got = await consumer.recv()
        assert got == _pkt(0)
    finally:
        producer.close()
        consumer.close()


async def test_consumer_poll_drains_all_arrived():
    consumer = await create_udp_consumer_transport("127.0.0.1", 0, max_queued=8)
    producer = await create_udp_producer_transport("127.0.0.1", consumer.local_port)
    try:
        for s in range(3):
            await producer.send(_pkt(s))
        first = await consumer.recv()
        await asyncio.sleep(0)  # let remaining loopback datagrams arrive before poll()
        rest = consumer.poll()
        seqs = [first.seq, *[p.seq for p in rest]]
        assert sorted(seqs) == [0, 1, 2]
    finally:
        producer.close()
        consumer.close()
