"""Closing a device stream while a blocking call is still inside it (ADR-0077).

Pa_CloseStream frees the stream and its host buffers; PortAudio synchronises nothing
against a Pa_ReadStream that another thread is still executing. The crash it produces is
a Windows access violation (0xC0000005, exit code -1073741819) that kills the process
with no Python exception to catch, and it only fires when the freed block happens to be
reused in time -- measured at ~5% of teardowns, so an end-to-end "does it crash" test
would be a coin flip. These tests pin the *ordering* invariant that makes the crash
impossible instead: close never overlaps a read, and the stream is still closed.
"""

import asyncio
import threading

import pytest

from vspeech.config import RecordingConfig
from vspeech.lib.audio import DeviceStreamThread
from vspeech.worker import recording as recording_mod
from vspeech.worker.recording import sd_recording_worker


class _BlockingStream:
    """A device whose read() sits inside "PortAudio" until it is released.

    Records whether close() landed while a read was still in there -- the exact overlap
    that frees the stream under the reader on a real device.
    """

    def __init__(self, frames_per_read: int = 1600) -> None:
        self.active = True
        self.frames_per_read = frames_per_read
        self.entered_read = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.inside_read = False
        self.closed_during_read = False

    def read(self, frames: int) -> tuple[bytes, bool]:
        self.inside_read = True
        self.entered_read.set()
        # Bounded so a regression fails the test instead of hanging the suite.
        self.release.wait(timeout=10)
        self.inside_read = False
        return b"\x00" * (frames * 2), False

    def close(self) -> None:
        self.closed_during_read = self.inside_read
        self.active = False
        self.closed.set()


# --- DeviceStreamThread --------------------------------------------------------------


async def test_close_waits_for_a_call_that_is_still_inside_the_device() -> None:
    """The whole point: a close asked for mid-read runs only once the read is out."""
    stream = _BlockingStream()
    thread = DeviceStreamThread("test_dev")
    call = asyncio.create_task(thread.call(stream.read, stream.frames_per_read))
    assert await asyncio.to_thread(stream.entered_read.wait, 10)

    thread.close(stream)

    # Still inside read() -> the close must not have happened.
    assert not stream.closed.is_set()
    stream.release.set()
    # ... but it must happen as soon as the read is out (no leaked native handle).
    assert await asyncio.to_thread(stream.closed.wait, 10)
    assert stream.closed_during_read is False
    await call


async def test_close_is_immediate_when_no_call_is_in_flight() -> None:
    """An idle thread means nothing is inside PortAudio, so the close happens on the
    caller's thread -- synchronously, the way `finally: stream.close()` always did."""
    stream = _BlockingStream()
    stream.release.set()  # reads return at once
    thread = DeviceStreamThread("test_dev")
    await thread.call(stream.read, stream.frames_per_read)

    thread.close(stream)

    assert stream.closed.is_set()
    assert stream.closed_during_read is False


async def test_close_still_closes_a_stream_whose_call_raised() -> None:
    """A device fault (the read raising) leaves nothing in flight, so the close is the
    same immediate one -- the fault path must not start deferring its close."""
    stream = _BlockingStream()

    def _boom(_frames: int) -> tuple[bytes, bool]:
        raise OSError("device gone")

    thread = DeviceStreamThread("test_dev")
    with pytest.raises(OSError):
        await thread.call(_boom, stream.frames_per_read)

    thread.close(stream)

    assert stream.closed.is_set()


async def test_closing_twice_is_harmless() -> None:
    """The second close must not raise (the executor is already retired by then)."""
    stream = _BlockingStream()
    thread = DeviceStreamThread("test_dev")
    thread.close(stream)
    thread.close(stream)


# --- sd_recording_worker: the path that actually crashed -----------------------------


async def test_cancelling_the_recorder_mid_read_never_closes_under_the_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation is what makes the overlap normal rather than exotic.

    `await <read>` hands control back the instant the task is cancelled, but the thread
    stays inside PortAudio, so the `finally: stream.close()` right behind it used to free
    the stream under the reader. Every teardown that cancels this worker mid-read hits it
    -- a failed worker startup (a taken gRPC port), a sibling worker dying, a TaskGroup
    unwinding -- which is how a config mistake turned into an access violation instead of
    the intended `exit 1`.
    """
    stream = _BlockingStream()
    monkeypatch.setattr(
        recording_mod, "open_input_stream", lambda config: (stream, 16000)
    )
    cfg = RecordingConfig(rate=16000, chunk=stream.frames_per_read)

    async def _drive() -> None:
        async for _ in sd_recording_worker(cfg):
            pass

    task = asyncio.create_task(_drive())
    assert await asyncio.to_thread(stream.entered_read.wait, 10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not stream.closed.is_set()  # the read is still inside the device
    stream.release.set()
    assert await asyncio.to_thread(stream.closed.wait, 10)
    assert stream.closed_during_read is False
