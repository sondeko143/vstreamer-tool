import pytest

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet


def _packet(seq=7, pcm=b"\x01\x02\x03\x04"):
    return StreamPacket(
        session_id="0123456789abcdef0123456789abcdef",
        seq=seq,
        pts=1.25,
        pcm=pcm,
        sample_rate=48000,
    )


def test_round_trip_preserves_all_fields():
    p = _packet()
    got = decode_packet(encode_packet(p))
    assert got == p


def test_round_trip_empty_and_large_pcm():
    for pcm in (b"", bytes(range(256)) * 60):  # ~15KB, > MTU
        p = _packet(pcm=pcm)
        assert decode_packet(encode_packet(p)).pcm == pcm


def test_decode_rejects_short_or_bad_magic():
    with pytest.raises(WireError):
        decode_packet(b"too-short")
    good = bytearray(encode_packet(_packet()))
    good[0] = ord("X")  # corrupt magic
    with pytest.raises(WireError):
        decode_packet(bytes(good))
