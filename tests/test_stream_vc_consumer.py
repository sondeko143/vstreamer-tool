from vspeech.stream_vc.consumer import consume_into_buffer
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket
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
