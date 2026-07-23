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


def _pkt(seq):
    return StreamPacket(
        session_id="cd" * 16,
        seq=seq,
        pts=0.0,
        pcm=bytes([seq % 256]) * 4,
        sample_rate=16000,
    )


async def test_consume_into_buffer_drains_recv_and_poll():
    t = _FakeTransport([_pkt(0), _pkt(1), _pkt(2)])
    buf = JitterBuffer(target_depth=0)
    first = await t.recv()
    consume_into_buffer(t, buf, first)
    assert buf.depth == 3  # first(0) + poll(1,2)
    assert buf.pop().kind is PopKind.NORMAL
