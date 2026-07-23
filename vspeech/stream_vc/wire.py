"""ストリーミング VC の UDP ワイヤ書式(ADR-0051 T3)。

固定ヘッダ + PCM ペイロードの 1 ブロック 1 データグラム。session_id は 32-hex を
16 バイト生 UUID に詰める。1 ブロックは MTU(1500)を超える(160ms/48kHz int16 で
~15KB)が UDP の 64KB 上限には収まるので、そのまま送って IP 層に断片化させる
(断片欠落はブロック単位の loss = seq gap として観測される, ADR-0056)。
"""

from __future__ import annotations

import struct

from vspeech.stream_vc.packet import StreamPacket

_MAGIC = b"SV"
_VERSION = 1
# network byte order: magic(2s) version(B) flags(B) session(16s) seq(Q) pts(d) rate(I)
_HEADER = struct.Struct("!2sBB16sQdI")


class WireError(ValueError):
    """データグラムが本コーデックの書式でない/壊れている。"""


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
