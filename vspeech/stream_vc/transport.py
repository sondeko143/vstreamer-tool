"""The swappable transport layer of streaming VC (ADR-0051).

Today only the in-process (asyncio.Queue) implementation exists. Producer and consumer
sit behind this interface, so a network implementation (UDP/TCP/bidi) can be swapped in
without changing any other VC or playback logic. The send side drops the oldest when
full, which keeps latency from growing monotonically (an acceptance criterion). The
receive side folds an already-arrived backlog forward to the newest via
`drain_to_latest`, which suppresses the permanent delay left behind by a transient stall
(the consumer consumes in real time, so it does not shrink on its own even at RTF<1).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull

from vspeech.stream_vc.packet import StreamPacket


def drop_oldest_put[T](q: Queue[T], item: T) -> bool:
    """Put `item`, dropping the oldest when full. Returns False when something was
    dropped.

    The shared backpressure handling of capture/transport. It keeps the queue from
    growing without bound when VC/GPU cannot keep up with real time, and returns a bool
    so the caller can observe the drop (acceptance criteria: latency does not grow
    monotonically, and drops are recordable).

    Assumes a single event loop: this function has no await, so no other coroutine can
    interleave between put_nowait and get_nowait. The get_nowait right after "full" was
    established always succeeds, and the put_nowait right after it always lands in the
    freed slot (no defensive try needed).
    """
    try:
        q.put_nowait(item)
        return True
    except QueueFull:
        # drop the oldest (always succeeds right after "full" was established)
        q.get_nowait()
        # into the freed slot (always succeeds because there is no await)
        q.put_nowait(item)
        return False


class Transport(ABC):
    @abstractmethod
    async def send(self, packet: StreamPacket) -> bool:
        """Send the packet. Returns False when backpressure dropped the oldest."""

    @abstractmethod
    async def recv(self) -> StreamPacket:
        """Receive the next packet (waiting if there is none)."""

    def drain_to_latest(self, keep: int = 1) -> list[StreamPacket]:
        """Keep only the newest `keep` of the arrived, waiting packets and take the rest
        out non-blockingly, returning the old packets that were discarded.

        It only touches the already-arrived queue and has no await, so it can be
        synchronous. A transport that cannot peek at its queue (none exists yet) does
        nothing by default.
        """
        return []

    def poll(self) -> list[StreamPacket]:
        """Take out every arrived, waiting packet non-blockingly and return them.

        The consumer calls this after recv to push whatever is left in the socket queue
        into the jitter buffer in one go (exposing the reordering to the buffer). A
        transport that cannot peek at its queue does nothing by default.
        """
        return []

    def close(self) -> None:
        """Release the transport's resources. Does nothing by default (InProcessTransport
        and the like)."""


class InProcessTransport(Transport):
    """The same-process asyncio.Queue implementation (ADR-0051 tier-0)."""

    def __init__(self, max_queued: int) -> None:
        self._q: Queue[StreamPacket] = Queue(maxsize=max_queued)
        self.dropped = 0

    async def send(self, packet: StreamPacket) -> bool:
        ok = drop_oldest_put(self._q, packet)
        if not ok:
            self.dropped += 1
        return ok

    async def recv(self) -> StreamPacket:
        return await self._q.get()

    def drain_to_latest(self, keep: int = 1) -> list[StreamPacket]:
        """Take waiting packets out non-blockingly, keeping the newest `keep`.

        The consumer (playback) consumes on the output device clock, i.e. in real time,
        so a backlog built up by a transient stall does not shrink on its own even at
        RTF<1 and sticks around as permanent delay. Calling this after recv discards
        already-arrived old packets and keeps playback near-live. What was discarded is
        returned so the caller can observe it through seq/telemetry. Assumes a single
        loop: there is no await, so no other coroutine interleaves.
        """
        dropped: list[StreamPacket] = []
        while self._q.qsize() > keep:
            try:
                dropped.append(self._q.get_nowait())
            except QueueEmpty:  # impossible on a single loop; bail out defensively
                break
        return dropped
