"""The self-healing device reconnect loop of streaming VC (ADR-0050).

Absorbs runtime device faults that occur in capture/playback steady state (OSError /
PortAudioError = the mic being unplugged, a format change, the output sink disappearing,
...) inside the subsystem, without dragging in sibling tasks or the utterance pipeline.

- **The first open is fail-loud** (worker_startup turns it into WorkerStartupError =
  ADR-0038). A missing model or device is a config problem and must not be hidden behind
  infinite retries.
- **Only steady-state device faults** are caught, then close -> backoff -> reopen. This
  follows the `(OSError, sd.PortAudioError)` retry pattern of the utterance-path
  recording worker (vspeech/worker/recording.py), which is reused unmodified.
- **CancelledError is never swallowed.** No bare except / except Exception; only
  DEVICE_ERRORS is caught (cancellation always propagates -> shutdown_worker).

This module is lazily imported only from capture/playback (subsystem.py does not import
it), so pulling in sounddevice does not cost the subsystem its CPU-light property.
"""

from __future__ import annotations

from asyncio import CancelledError
from asyncio import sleep as _async_sleep
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Protocol

import sounddevice as sd

from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger

# Device faults we recover from on our own in steady state. CUDA exceptions (the
# RuntimeError family) and CancelledError are **excluded** -- the runner handles the
# former separately and the latter must propagate.
DEVICE_ERRORS = (OSError, sd.PortAudioError)

# Reconnect backoff (seconds). Grows from start by factor and saturates at MAX.
BACKOFF_START = 0.5
BACKOFF_MAX = 5.0
BACKOFF_FACTOR = 2.0


def next_backoff(prev: float) -> float:
    """The next exponential-backoff value (clamped at BACKOFF_MAX). Pure (CPU-testable)."""
    return min(prev * BACKOFF_FACTOR, BACKOFF_MAX)


class _Closable(Protocol):
    def close(self) -> None: ...


def close_quietly(stream: _Closable) -> None:
    """Swallow device exceptions raised by stream.close().

    There is a path that double-closes an already broken/closed device
    (fault -> close -> close again in finally), so DEVICE_ERRORS from close itself are
    logged and ignored.
    """
    try:
        stream.close()
    except DEVICE_ERRORS as e:
        logger.debug("stream_vc ignore error while closing stream: %r", e)


async def _reopen_with_backoff[T: _Closable](
    open_stream: Callable[[], T],
    sleep: Callable[[float], Awaitable[None]],
    label: str,
) -> T:
    """Retry open_stream() with backoff in between until it succeeds, then return it.

    A reopen is a **runtime retry**, so it is not wrapped in worker_startup (not made
    fail-loud). Even when the open itself fails with DEVICE_ERRORS we extend the backoff
    and keep going. CancelledError propagates straight through the sleep.
    """
    backoff = BACKOFF_START
    while True:
        await sleep(backoff)
        try:
            stream = open_stream()
        except DEVICE_ERRORS as e:
            backoff = next_backoff(backoff)
            logger.warning(
                "%s reopen failed for %r; next backoff %.1fs", label, e, backoff
            )
            continue
        logger.info("%s reopened", label)
        return stream


async def run_with_device_retry[T: _Closable](
    *,
    open_stream: Callable[[], T],
    run: Callable[[T], Awaitable[None]],
    worker: str,
    label: str,
    on_reopen: Callable[[], None] | None = None,
    reopen_metric: str | None = None,
    sleep: Callable[[float], Awaitable[None]] = _async_sleep,
) -> None:
    """The device loop: first open -> steady state -> reconnect on a device fault.

    - `open_stream`: open and return the stream. Only the first call is wrapped in
      worker_startup and made fail-loud (every reopen after that is a runtime retry).
    - `run`: return a coroutine that runs steady state on the given stream. When it comes
      back having raised a device fault (DEVICE_ERRORS), close -> backoff -> reopen.
    - `on_reopen`: hook called just before a reopen (to reset per-connection state, etc).
    - `reopen_metric`: when given, records 1.0 to telemetry on every reconnect.

    CancelledError is not caught; it is wrapped by shutdown_worker and raised.
    """
    with worker_startup(worker):
        stream = open_stream()  # fail-loud on the first open only (ADR-0038)
    logger.info("%s started", label)
    try:
        while True:
            try:
                await run(stream)
            except DEVICE_ERRORS as e:
                # A runtime device fault. Absorb it inside the subsystem without dragging
                # in the utterance path or sibling tasks (ADR-0050).
                logger.warning("%s device fault; retry for %r", label, e)
                if reopen_metric:
                    telemetry.record(reopen_metric, 1.0)
                close_quietly(stream)
                if on_reopen is not None:
                    on_reopen()
                stream = await _reopen_with_backoff(open_stream, sleep, label)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        close_quietly(stream)
