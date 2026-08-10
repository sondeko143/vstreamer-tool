"""The UDP wire format of streaming VC (ADR-0051 T3).

One datagram per block: a fixed header plus the PCM payload. session_id packs the
32-hex form into a 16-byte raw UUID. A block exceeds the MTU (1500) -- about 15KB at
160ms/48kHz int16 -- but stays within UDP's 64KB limit, so it is sent as-is and let the
IP layer fragment it (losing a fragment is observed as block-granularity loss, i.e. a
seq gap, ADR-0056).

`sample_rate` is the one field that gets a value check (see the constants below): it
arrives as an unsigned 32-bit int from an unauthenticated LAN socket and the receiver
turns it straight into a resampler, so a corrupt one is not merely wrong audio -- it is a
crash or a multi-gigabyte allocation. Everything else here stays a pure codec.
"""

from __future__ import annotations

import struct

from vspeech.stream_vc.packet import StreamPacket

_MAGIC = b"SV"
_VERSION = 1
# network byte order: magic(2s) version(B) flags(B) session(16s) seq(Q) pts(d) rate(I)
_HEADER = struct.Struct("!2sBB16sQdI")

# The sample rates a datagram may carry. The bounds are the outer edge of the standard
# audio rate family (8 kHz telephony .. 192 kHz), and the step is that family's own gcd:
# every standard rate (8000, 11025, 16000, 22050, 24000, 32000, 40000, 44100, 48000,
# 88200, 96000, 176400, 192000) is a multiple of 25, and 25 is the largest step that
# keeps all of them (11025 is not a multiple of 50). The RVC model rates this transport
# actually carries -- 32000 / 40000 / 48000 -- sit well inside it.
#
# The step is not cosmetic, and a range check alone would not do. The receiver builds an
# L/M polyphase filter from (packet rate, device rate) whose phase count is
# `device_rate // gcd(...)`, so a rate coprime with the device explodes it **whatever its
# magnitude** -- measured on this rig: 44101 -> 48000 builds 4.8M taps (563 MB, 1.4 s)
# and 191999 -> 192000 builds 19.6M taps (2.3 GB, 6.5 s), while the header's own maximum
# (2**32-1) reaches 2.9e10 taps = 233 GB and simply dies. Requiring the 25 Hz lattice
# forces gcd >= 25 against every real device rate (they are all multiples of 25 too),
# which caps the worst case in this range at 785k taps (91 MB, 0.24 s, measured).
_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 192000
_SAMPLE_RATE_STEP = 25


class WireError(ValueError):
    """The datagram is not in this codec's format, or is corrupt."""


def _check_sample_rate(rate: int) -> None:
    """Reject a sample_rate this codec will not carry.

    Checked on **both** sides: decode because the field arrives from the network and a
    single corrupt datagram must not be able to crash or wedge the consumer (udp.py drops
    a WireError and carries on), and encode so the codec cannot emit what its own decoder
    would refuse -- a producer whose every packet is dropped at the far end is far harder
    to diagnose from the receiving machine than a sender that says so at once.
    """
    if not _MIN_SAMPLE_RATE <= rate <= _MAX_SAMPLE_RATE:
        raise WireError(
            f"サンプルレート {rate} は範囲外です "
            f"({_MIN_SAMPLE_RATE}〜{_MAX_SAMPLE_RATE}Hz)"
        )
    if rate % _SAMPLE_RATE_STEP:
        raise WireError(
            f"サンプルレート {rate} は {_SAMPLE_RATE_STEP}Hz の倍数ではありません"
        )


def encode_packet(p: StreamPacket) -> bytes:
    _check_sample_rate(p.sample_rate)
    header = _HEADER.pack(
        _MAGIC, _VERSION, 0, bytes.fromhex(p.session_id), p.seq, p.pts, p.sample_rate
    )
    return header + p.pcm


def decode_packet(data: bytes) -> StreamPacket:
    if len(data) < _HEADER.size:
        raise WireError(f"datagram too short: {len(data)} < {_HEADER.size}")
    magic, version, _flags, session, seq, pts, rate = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION:
        raise WireError(f"bad magic/version: {magic!r}/{version}")
    # Before the StreamPacket exists, so an out-of-range rate can never reach the sink's
    # resampler (see the constants above for what it would cost there).
    _check_sample_rate(rate)
    return StreamPacket(
        session_id=session.hex(),
        seq=seq,
        pts=pts,
        pcm=data[_HEADER.size :],
        sample_rate=rate,
    )
