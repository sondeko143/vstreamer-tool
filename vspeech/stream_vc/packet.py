"""ストリーミング VC のマシン間ユニット(ADR-0051)。

`session_id`/`seq`/`pts` を持つので consumer 側で欠落検出・整列ができる。
現状は単一マシン内の in-process 転送だが、将来 網トランスポートへ差し替えても
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
