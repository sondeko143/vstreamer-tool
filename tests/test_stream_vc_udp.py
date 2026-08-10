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


async def test_a_corrupt_sample_rate_is_dropped_not_raised():
    """A single datagram whose rate field was corrupted must go down the existing
    malformed-drop path, not into the sink's resampler.

    2**32-1 is what the header's unsigned field permits; letting it through would ask for
    a 233 GB filter. The receiver is a callback on the event loop, so anything that
    escaped `datagram_received` would be an unhandled exception in the loop rather than a
    catchable fault -- hence: dropped, counted, warned about once, queue untouched.
    """
    import struct
    from unittest.mock import patch

    from vspeech.stream_vc.udp import _RecvProtocol
    from vspeech.stream_vc.wire import _HEADER
    from vspeech.stream_vc.wire import encode_packet

    queue = asyncio.Queue(maxsize=8)
    proto = _RecvProtocol(queue)
    good = encode_packet(_pkt(0))
    # Only the rate field is overwritten, i.e. what a bit flip does. encode_packet cannot
    # build this one: it refuses the same rates its decoder refuses.
    corrupt = bytearray(good)
    corrupt[_HEADER.size - 4 : _HEADER.size] = struct.pack("!I", 2**32 - 1)
    with patch("vspeech.stream_vc.udp.logger") as mock_logger:
        with patch("vspeech.stream_vc.udp.telemetry") as mock_telemetry:
            proto.datagram_received(bytes(corrupt), ("127.0.0.1", 1))
            proto.datagram_received(good, ("127.0.0.1", 1))
    assert queue.qsize() == 1  # only the good one got through
    assert (await queue.get()).seq == 0
    mock_telemetry.record.assert_called_once_with("stream_vc_malformed_drop", 1.0)
    assert mock_logger.warning.call_count == 1
    assert "4294967295" in repr(mock_logger.warning.call_args)


def test_send_protocol_error_received_records_telemetry():
    from unittest.mock import patch

    from vspeech.stream_vc.udp import _SendProtocol

    proto = _SendProtocol()
    with patch("vspeech.stream_vc.udp.telemetry") as mock_telemetry:
        proto.error_received(OSError("route gone"))
        proto.error_received(OSError("again"))
    assert mock_telemetry.record.call_count == 2


def test_send_protocol_error_logging_is_throttled():
    from unittest.mock import patch

    from vspeech.stream_vc.udp import _SendProtocol

    proto = _SendProtocol()
    with patch("vspeech.stream_vc.udp.logger") as mock_logger:
        with patch("vspeech.stream_vc.udp.telemetry") as mock_telemetry:
            for _ in range(120):
                proto.error_received(OSError("peer down"))
    # Telemetry every time. Only one log line at the head of the episode (a tight loop is
    # a single episode).
    assert mock_telemetry.record.call_count == 120
    assert mock_logger.warning.call_count == 1
