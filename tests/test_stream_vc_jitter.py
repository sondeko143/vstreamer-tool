# tests/test_stream_vc_jitter.py
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket


def _pkt(seq, byte):
    return StreamPacket(
        session_id="00" * 16,
        seq=seq,
        pts=0.0,
        pcm=bytes([byte, byte]),
        sample_rate=16000,
    )


def _prime(buf, seqs):
    for s in seqs:
        buf.push(_pkt(s, s % 256))


def test_prebuffer_emits_silence_until_primed():
    buf = JitterBuffer(target_depth=2)
    buf.push(_pkt(0, 9))
    r = buf.pop()  # only 1 buffered, need depth+1=3
    assert r.kind is PopKind.PREBUFFER
    assert set(r.pcm) == {0}  # silence, sized to block


def test_in_order_playout_after_prime():
    buf = JitterBuffer(target_depth=1)
    _prime(buf, [0, 1])  # depth+1=2 -> primes at seq 0
    assert buf.pop().pcm == bytes([0, 0])
    assert buf.pop().pcm == bytes([1, 1])


def test_reorder_within_depth_is_recovered():
    buf = JitterBuffer(target_depth=1)
    buf.push(_pkt(0, 0))
    buf.push(_pkt(2, 2))  # 1 arrives late, out of order
    buf.push(_pkt(1, 1))
    assert buf.pop().pcm == bytes([0, 0])
    assert buf.pop().pcm == bytes([1, 1])  # recovered in order
    assert buf.pop().pcm == bytes([2, 2])


def test_missing_packet_conceals_and_advances():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(0, 5))
    assert buf.pop().kind is PopKind.NORMAL  # seq 0
    # seq 1 never arrives; seq 2 present -> pop expects 1 -> conceal
    buf.push(_pkt(2, 2))
    r = buf.pop()
    assert r.kind is PopKind.CONCEAL
    assert r.gap == 1
    assert buf.pop().pcm == bytes([2, 2])  # resumes at 2


def test_late_packet_after_playout_is_dropped():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(0, 0))
    buf.pop()  # plays 0, next_seq=1
    assert buf.push(_pkt(0, 0)) is False  # seq 0 already gone -> dropped


def test_overflow_fast_forwards_to_bound_latency():
    buf = JitterBuffer(target_depth=1)
    # far-ahead burst while next_seq stuck at 0 (0 never came)
    for s in range(1, 12):
        buf.push(_pkt(s, s % 256))
    r = buf.pop()  # newest=11, depth=1 -> ff next_seq near 10, dropping the middle
    assert r.dropped > 0
    assert buf.pop().kind is PopKind.NORMAL


def test_reset_clears_state():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(5, 5))
    buf.pop()
    buf.reset()
    buf.push(_pkt(0, 0))
    assert buf.pop().pcm == bytes([0, 0])  # next_seq re-primes from scratch
