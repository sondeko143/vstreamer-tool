"""The cross-machine unit of streaming VC (ADR-0051).

Carrying `session_id`/`seq`/`pts` lets the consumer detect loss and reorder. Today the
transfer is in-process on a single machine, but the same StreamPacket flows when the
transport is later swapped for a network one.
"""

from dataclasses import dataclass


@dataclass
class StreamPacket:
    session_id: str
    seq: int
    pts: float
    pcm: bytes
    sample_rate: int
