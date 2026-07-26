"""The self-healing device reconnect loop of streaming VC (vspeech/stream_vc/retry.py).

Verifies the pure next_backoff plus run_with_device_retry's reconnect / fail-loud /
cancellation behaviour on CPU, using a fake stream and no real device.
"""

import asyncio

import pytest
import sounddevice as sd

from vspeech.exceptions import WorkerShutdown
from vspeech.exceptions import WorkerStartupError
from vspeech.stream_vc.retry import BACKOFF_MAX
from vspeech.stream_vc.retry import BACKOFF_START
from vspeech.stream_vc.retry import next_backoff
from vspeech.stream_vc.retry import run_with_device_retry


def test_next_backoff_grows_then_caps():
    b = BACKOFF_START
    seen = [b]
    for _ in range(10):
        b = next_backoff(b)
        seen.append(b)
    assert seen[1] == BACKOFF_START * 2  # doubles each time
    assert max(seen) == BACKOFF_MAX  # eventually saturates
    assert next_backoff(BACKOFF_MAX) == BACKOFF_MAX  # clamped at the ceiling


class _FakeStream:
    """A device stand-in whose read consumes a scripted sequence (exceptions or values).
    Once empty, it blocks forever."""

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    async def read(self):
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        await asyncio.Event().wait()  # once the script runs out, wait to be cancelled


async def _nosleep(_: float) -> None:
    return


async def _read_loop(stream: _FakeStream) -> None:
    while True:
        await stream.read()


@pytest.mark.parametrize(
    "fault", [OSError("mic unplugged"), sd.PortAudioError("format changed")]
)
async def test_reopens_on_device_error_and_cancellation_propagates(fault):
    """A steady-state (OSError, PortAudioError) recovers through close -> reopen, while
    CancelledError is not swallowed and propagates as WorkerShutdown."""
    opened: list[_FakeStream] = []

    def open_stream() -> _FakeStream:
        # The first read faults immediately; every stream after that blocks forever.
        s = _FakeStream([fault] if not opened else [])
        opened.append(s)
        return s

    task = asyncio.create_task(
        run_with_device_retry(
            open_stream=open_stream,
            run=_read_loop,
            worker="stream_vc",
            label="test",
            sleep=_nosleep,
        )
    )
    for _ in range(100):
        await asyncio.sleep(0)
        if len(opened) >= 2:
            break
    assert len(opened) == 2  # the first open plus the reopen after the fault
    assert opened[0].closed >= 1  # closed on the fault
    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_first_open_failure_is_fail_loud():
    """A failure on the first open becomes a WorkerStartupError through worker_startup (it
    does not retry forever)."""

    def open_stream() -> _FakeStream:
        raise OSError("no such device")

    async def run(_stream: _FakeStream) -> None:
        raise AssertionError("run should not be reached")

    with pytest.raises(WorkerStartupError):
        await run_with_device_retry(
            open_stream=open_stream,
            run=run,
            worker="stream_vc",
            label="test",
            sleep=_nosleep,
        )


async def test_non_device_error_propagates_without_retry():
    """Anything outside DEVICE_ERRORS (here a ValueError) is not caught and triggers no
    reopen."""
    opened: list[_FakeStream] = []

    def open_stream() -> _FakeStream:
        s = _FakeStream([])
        opened.append(s)
        return s

    async def run(_stream: _FakeStream) -> None:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        await run_with_device_retry(
            open_stream=open_stream,
            run=run,
            worker="stream_vc",
            label="test",
            sleep=_nosleep,
        )
    assert len(opened) == 1  # no reopen happened
