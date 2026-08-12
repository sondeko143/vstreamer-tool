import struct

import pytest

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.wire import _HEADER
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet

# Every rate in the standard audio family, plus the three RVC model rates this transport
# actually carries (32000 / 40000 / 48000).
STANDARD_RATES = [
    8000,
    11025,
    16000,
    22050,
    24000,
    32000,
    40000,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
]


def _packet(seq=7, pcm=b"\x01\x02\x03\x04", sample_rate=48000):
    return StreamPacket(
        session_id="0123456789abcdef0123456789abcdef",
        seq=seq,
        pts=1.25,
        pcm=pcm,
        sample_rate=sample_rate,
    )


def _with_rate(datagram: bytes, rate: int) -> bytes:
    """A valid datagram with only its rate field overwritten.

    Built by patching the bytes rather than through encode_packet, because encode refuses
    these rates too -- this is what a bit flip on the wire produces, which is the case the
    receiver has to survive.
    """
    body = bytearray(datagram)
    body[_HEADER.size - 4 : _HEADER.size] = struct.pack("!I", rate)
    return bytes(body)


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


# --- sample_rate validation ---------------------------------------------------------


@pytest.mark.parametrize("rate", STANDARD_RATES)
def test_every_standard_sample_rate_still_round_trips(rate: int):
    """The validation must cost the codec nothing it used to carry."""
    p = _packet(sample_rate=rate)
    assert decode_packet(encode_packet(p)) == p


@pytest.mark.parametrize("rate", [0, 1, 7999, 192025, 2**32 - 1])
def test_decode_rejects_a_sample_rate_outside_the_carried_range(rate: int):
    """A rate the pipeline cannot produce is a corrupt datagram, not audio.

    The header's rate field is an unsigned 32-bit int, so a bit flip can put anything
    from 0 to 4294967295 in it. Both ends are fatal downstream if let through: 0 makes
    the resampler raise (and the raise is not a device error, so it takes the subsystem
    down), and 4294967295 asks for a 2.9e10-tap filter, i.e. 233 GB.
    """
    good = encode_packet(_packet())
    with pytest.raises(WireError):
        decode_packet(_with_rate(good, rate))


@pytest.mark.parametrize("rate", [16001, 44101, 47952])
def test_decode_rejects_a_sample_rate_off_the_25hz_lattice(rate: int):
    """In-range is not enough: the cost is set by gcd(rate, device_rate), not magnitude.

    A rate coprime with the device gives the polyphase filter `device_rate` phases
    whatever its size -- measured 44101 -> 48000 = 4.8M taps / 563 MB / 1.4 s, and
    191999 -> 192000 = 19.6M taps / 2.3 GB / 6.5 s, which wedges (or kills) a
    playback-only consumer. Every standard rate is a multiple of 25, so requiring the
    lattice costs nothing real and caps the worst case at 785k taps / 91 MB.
    """
    good = encode_packet(_packet())
    with pytest.raises(WireError):
        decode_packet(_with_rate(good, rate))


@pytest.mark.parametrize("rate", [0, 44101, 2**32 - 1])
def test_encode_refuses_what_decode_would_reject(rate: int):
    """The codec must not be able to emit a datagram its own decoder drops.

    A producer whose every packet is silently dropped at the far end is much harder to
    diagnose from the receiving machine than a sender that fails loud at the first block.
    """
    with pytest.raises(WireError):
        encode_packet(_packet(sample_rate=rate))


def test_the_rejection_message_names_the_rate():
    """It travels into the receiver's throttled warning (udp.py logs `%r`), and it is the
    only clue an operator gets for why the consumer went quiet."""
    with pytest.raises(WireError, match="44101"):
        encode_packet(_packet(sample_rate=44101))
