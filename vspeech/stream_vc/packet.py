"""ストリーミング VC のマシン間ユニット(ADR-0051)。

`session_id`/`seq`/`pts` を持つので consumer 側で欠落検出・整列ができる。
M2 は単一マシン in-process だが、M3 で網トランスポートへ差し替わったときに
同じ StreamPacket が流れる。
"""

from dataclasses import dataclass


@dataclass
class StreamPacket:
    session_id: str
    seq: int
    pts: float
    pcm: bytes
    sample_rate: int
