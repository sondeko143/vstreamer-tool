"""The UDP wire format of streaming VC (ADR-0051 T3).

One datagram per block: a fixed header plus the PCM payload. session_id packs the
32-hex form into a 16-byte raw UUID. A block exceeds the MTU (1500) -- about 15KB at
160ms/48kHz int16 -- but stays within UDP's 64KB limit, so it is sent as-is and let the
IP layer fragment it (losing a fragment is observed as block-granularity loss, i.e. a
seq gap, ADR-0056).
"""

from __future__ import annotations

import struct

from vspeech.stream_vc.packet import StreamPacket

_MAGIC = b"SV"
_VERSION = 1
# network byte order: magic(2s) version(B) flags(B) session(16s) seq(Q) pts(d) rate(I)
_HEADER = struct.Struct("!2sBB16sQdI")


class WireError(ValueError):
    """The datagram is not in this codec's format, or is corrupt."""


def encode_packet(p: StreamPacket) -> bytes:
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
    return StreamPacket(
        session_id=session.hex(),
        seq=seq,
        pts=pts,
        pcm=data[_HEADER.size :],
        sample_rate=rate,
    )
