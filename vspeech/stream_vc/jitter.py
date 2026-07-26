"""The consumer-side jitter buffer (ADR-0056).

Pure logic holding reordering, prebuffering, concealment and the delay ceiling,
independent of the transport. The prebuffer depth is directly the reordering tolerance:
at pop time we read `depth` blocks behind the newest, so a packet that was merely
reordered is already in the buffer. A packet genuinely absent at pop time is real loss,
and we emit a concealment that fades the previous block and advance. numpy is imported
inside the method only for concealment (to keep importing this module cheap).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vspeech.stream_vc.packet import StreamPacket

# overflow: fast-forward to near-live once newest_seq - next_seq exceeds this slack.
_OVERFLOW_SLACK = 4


class PopKind(Enum):
    PREBUFFER = 0  # not primed yet (startup). Emits silence.
    NORMAL = 1  # the expected seq was present and played.
    CONCEAL = 2  # the expected seq was missing; concealed by fading the previous block.


@dataclass
class PopResult:
    pcm: bytes
    kind: PopKind
    gap: int  # packets confirmed lost by this pop (for telemetry)
    dropped: int  # packets discarded by the overflow fast-forward


class JitterBuffer:
    def __init__(self, target_depth: int) -> None:
        self.target_depth = target_depth
        self._buf: dict[int, bytes] = {}
        self._next_seq: int | None = None  # None = not primed
        self._last_good: bytes | None = None
        self._concealed_since_good = 0
        self._block_bytes = 0  # settled by the first push

    @property
    def depth(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
        self._next_seq = None
        self._last_good = None
        self._concealed_since_good = 0
        self._block_bytes = 0

    def push(self, packet: StreamPacket) -> bool:
        if not self._block_bytes:
            self._block_bytes = len(packet.pcm)
        if self._next_seq is not None and packet.seq < self._next_seq:
            return False  # older than what we played = late. Discard (the caller logs).
        if packet.seq in self._buf:
            return False  # duplicate
        self._buf[packet.seq] = packet.pcm
        return True

    def _silence(self) -> bytes:
        return bytes(self._block_bytes)

    def _conceal(self) -> bytes:
        # The first miss fades the last good block into silence; consecutive misses after
        # that are plain silence.
        if self._last_good is None or self._concealed_since_good > 0:
            self._concealed_since_good += 1
            return self._silence()
        self._concealed_since_good += 1
        import numpy as np

        a = np.frombuffer(self._last_good, dtype=np.int16).astype(np.float32)
        fade = np.linspace(1.0, 0.0, a.shape[0], dtype=np.float32)
        return np.rint(a * fade).astype(np.int16).tobytes()

    def pop(self) -> PopResult:
        if self._next_seq is None:
            if len(self._buf) > self.target_depth:
                self._next_seq = min(self._buf)
            else:
                return PopResult(self._silence(), PopKind.PREBUFFER, gap=0, dropped=0)
        dropped = 0
        # never-arrived packets skipped by fast-forward = real loss (observable)
        gap = 0
        # Delay ceiling: jump to near-live once the backlog exceeds the slack (recording
        # what was discarded).
        if self._buf:
            newest = max(self._buf)
            if newest - self._next_seq > self.target_depth + _OVERFLOW_SLACK:
                target = newest - self.target_depth
                for s in list(self._buf):
                    if s < target:
                        del self._buf[s]
                        dropped += 1
                # of the skipped [next_seq, target) range: dropped = arrived-but-stale
                # (drained for latency), the rest never arrived = real loss. The spec
                # requires loss be observable (never punch a silent hole quietly), so
                # surface it.
                gap = (target - self._next_seq) - dropped
                self._next_seq = target
        pcm = self._buf.pop(self._next_seq, None)
        if pcm is not None:
            self._next_seq += 1
            self._last_good = pcm
            self._concealed_since_good = 0
            return PopResult(pcm, PopKind.NORMAL, gap=gap, dropped=dropped)
        # The expected seq is absent. Only when **a newer seq is in the buffer** is this
        # confirmed real loss (we jumped past the expected seq) -> conceal and advance. An
        # empty buffer means starvation (it simply has not arrived yet), so **do not
        # advance**: moving the cursor lets an empty pop -- right after an expected packet
        # still in flight, or a duplicate/late straggler -- overtake a live seq, after
        # which every in-order packet is judged late and dropped and the output goes
        # silent forever (this happens at the default target_depth=0; demonstrated in the
        # final review).
        if self._buf:
            self._next_seq += 1
            return PopResult(
                self._conceal(), PopKind.CONCEAL, gap=gap + 1, dropped=dropped
            )
        return PopResult(self._conceal(), PopKind.CONCEAL, gap=gap, dropped=dropped)
