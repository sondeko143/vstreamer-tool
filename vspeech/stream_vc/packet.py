"""The cross-machine unit of streaming VC (ADR-0051).

Carrying `session_id`/`seq`/`pts` lets the consumer detect loss and reorder. Today the
transfer is in-process on a single machine, but the same StreamPacket flows when the
transport is later swapped for a network one.
"""

from dataclasses import dataclass

from vspeech.config import SampleFormat

# How `StreamPacket.pcm` is encoded (ADR-0051). Stated next to the field it describes,
# because the code that decodes and re-encodes it lives in another module (playback.py's
# sink) and would otherwise be free to drift from the payload it is reading.
PACKET_FORMAT = SampleFormat.INT16
PACKET_CHANNELS = 1


@dataclass
class StreamPacket:
    session_id: str
    seq: int
    pts: float
    pcm: bytes
    sample_rate: int
