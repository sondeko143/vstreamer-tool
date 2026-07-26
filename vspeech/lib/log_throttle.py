"""Time-based thinning of repeating warnings (ADR-0062).

Warnings from the real-time loops (underflow / drop / gap / transient GPU errors / UDP
protocol errors) fire on every block or every packet once they start. Emitting them
unconditionally buries the log in warnings and destroys their diagnostic value, so they
are rate-limited by time, per episode.
"""

from collections.abc import Callable
from time import perf_counter

# Output ceiling while a fault persists. It cuts on time rather than event rate, so
# changing block_ms or the transport does not change the line rate (ADR-0062).
DEFAULT_MIN_INTERVAL_S = 5.0
# Once occurrences stop for this long, the next one starts over as "another incident".
DEFAULT_QUIET_S = 10.0


class LogThrottle:
    """Rate-limit repeated warnings of the same condition by time, per episode.

    `hit()` records one occurrence and returns "the running count within this episode"
    when the caller should log, or None when it should not. Callers keep no counter of
    their own:

        if (n := self._underflow.hit()) is not None:
            logger.warning("... (total %d)", n)

    The end of an episode is decided lazily inside the next `hit()` -- so that the
    success path never has to call in to say "it settled down" (nothing is added to the
    real-time hot path). The cost is that there is no explicit "recovered" line; the
    first line of the next episode is the boundary.

    Assumes a single event loop and holds no lock (every intended caller, including the
    UDP protocol callbacks, runs on that loop).
    """

    def __init__(
        self,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        quiet_s: float = DEFAULT_QUIET_S,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._quiet_s = quiet_s
        self._clock = clock
        self._count = 0
        self._last_hit: float | None = None
        self._last_log = 0.0

    def hit(self) -> int | None:
        """Record one occurrence. Returns the running count to log, or None to stay
        quiet."""
        now = self._clock()
        # First occurrence, or a recurrence after a quiet_s gap = a new episode. Reset
        # the count and always emit one line (with a cumulative counter "the first one"
        # could only ever happen once in the process lifetime).
        if self._last_hit is None or now - self._last_hit > self._quiet_s:
            self._count = 1
            self._last_hit = now
            self._last_log = now
            return 1
        # The episode continues. Always advance the count; only the output is throttled
        # by min_interval_s. The episode boundary is measured from "when it last
        # occurred", not "when it was last logged" -- otherwise a boundary would fall in
        # the middle of a stretch that is being thinned.
        self._count += 1
        self._last_hit = now
        if now - self._last_log >= self._min_interval_s:
            self._last_log = now
            return self._count
        return None
