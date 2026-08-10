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

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import SampleFormat
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import DeviceStreamThread
from vspeech.shared_context import SoundInput
from vspeech.stream_vc.capture import InputTap
from vspeech.stream_vc.playback import OutputSink
from vspeech.worker import playback as playback_mod
from vspeech.worker import recording as recording_mod
from vspeech.worker.playback import OutputStream
from vspeech.worker.recording import sd_recording_worker

_OUTPUT_DEVICE = DeviceInfo(
    index=0,
    name="Speakers",
    host_api=0,
    max_input_channels=0,
    max_output_channels=2,
)


class _BlockingStream:
    """A device whose read()/write() sits inside "PortAudio" until it is released.

    Records whether close() landed while a call was still in there -- the exact overlap
    that frees the stream under the caller on a real device.
    """

    def __init__(self, frames_per_read: int = 1600) -> None:
        self.active = True
        self.frames_per_read = frames_per_read
        self.samplerate = 48000.0
        self.entered_call = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.inside_call = False
        self.closed_during_call = False

    def _block(self) -> None:
        self.inside_call = True
        self.entered_call.set()
        # Bounded so a regression fails the test instead of hanging the suite.
        self.release.wait(timeout=10)
        self.inside_call = False

    def read(self, frames: int) -> tuple[bytes, bool]:
        self._block()
        return b"\x00" * (frames * 2), False

    def write(self, data: bytes) -> bool:
        self._block()
        return False

    def close(self) -> None:
        self.closed_during_call = self.inside_call
        self.active = False
        self.closed.set()


# --- DeviceStreamThread --------------------------------------------------------------


async def test_close_waits_for_a_call_that_is_still_inside_the_device() -> None:
    """The whole point: a close asked for mid-read runs only once the read is out."""
    stream = _BlockingStream()
    thread = DeviceStreamThread("test_dev")
    call = asyncio.create_task(thread.call(stream.read, stream.frames_per_read))
    assert await asyncio.to_thread(stream.entered_call.wait, 10)

    thread.close(stream.close)

    # Still inside read() -> the close must not have happened.
    assert not stream.closed.is_set()
    stream.release.set()
    # ... but it must happen as soon as the read is out (no leaked native handle).
    assert await asyncio.to_thread(stream.closed.wait, 10)
    assert stream.closed_during_call is False
    await call


async def test_close_is_immediate_when_no_call_is_in_flight() -> None:
    """An idle thread means nothing is inside PortAudio, so the close happens on the
    caller's thread -- synchronously, the way `finally: stream.close()` always did."""
    stream = _BlockingStream()
    stream.release.set()  # reads return at once
    thread = DeviceStreamThread("test_dev")
    await thread.call(stream.read, stream.frames_per_read)

    thread.close(stream.close)

    assert stream.closed.is_set()
    assert stream.closed_during_call is False


async def test_close_still_closes_a_stream_whose_call_raised() -> None:
    """A device fault (the read raising) leaves nothing in flight, so the close is the
    same immediate one -- the fault path must not start deferring its close."""
    stream = _BlockingStream()

    def _boom(_frames: int) -> tuple[bytes, bool]:
        raise OSError("device gone")

    thread = DeviceStreamThread("test_dev")
    with pytest.raises(OSError):
        await thread.call(_boom, stream.frames_per_read)

    thread.close(stream.close)

    assert stream.closed.is_set()


async def test_closing_twice_is_harmless() -> None:
    """The second close must not raise (the executor is already retired by then)."""
    stream = _BlockingStream()
    thread = DeviceStreamThread("test_dev")
    thread.close(stream.close)
    thread.close(stream.close)


# --- the four device boundaries ------------------------------------------------------
#
# One test per boundary, all making the same statement: with a call still inside the
# device, that boundary's own close defers, and once the call is out the device really is
# closed. The recorder's is driven through its whole worker because that is the path that
# was seen to crash; the other three are driven through the object that owns the stream,
# which is where each of their closes lives (retry.py's close_quietly for the two stream_vc
# ones, sd_playback_worker's finally for the utterance one).


async def _await_entered(stream: _BlockingStream) -> None:
    assert await asyncio.to_thread(stream.entered_call.wait, 10)


async def _assert_deferred_then_closed(stream: _BlockingStream) -> None:
    assert not stream.closed.is_set()  # the call is still inside the device
    stream.release.set()
    assert await asyncio.to_thread(stream.closed.wait, 10)
    assert stream.closed_during_call is False


async def test_stream_vc_capture_close_waits_for_the_read_in_flight() -> None:
    """The mic tap of streaming VC. Its close runs on every reconnect (ADR-0050), not
    only at teardown, so this is the boundary with the most exposure of the four."""
    stream = _BlockingStream()
    tap = InputTap(stream, 48000)  # ty: ignore[invalid-argument-type]
    read = asyncio.create_task(tap.read(stream.frames_per_read))
    await _await_entered(stream)

    tap.close()

    await _assert_deferred_then_closed(stream)
    await read


async def test_stream_vc_output_sink_close_waits_for_the_write_in_flight() -> None:
    """The speaker sink of streaming VC, shared by the local (playback.py) and the
    consumer (consumer.py) loops -- both close it on every reconnect."""
    stream = _BlockingStream()
    sink = OutputSink(stream, 48000)  # ty: ignore[invalid-argument-type]
    write = asyncio.create_task(sink.play(b"\x00\x00" * 160, 48000))
    await _await_entered(stream)

    sink.close()

    await _assert_deferred_then_closed(stream)
    await write


async def test_utterance_playback_close_waits_for_the_write_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The utterance playback device. One utterance is one whole write, so the window
    here is as long as the audio being played."""
    stream = _BlockingStream()
    monkeypatch.setattr(
        playback_mod, "get_output_device", lambda config: _OUTPUT_DEVICE
    )
    monkeypatch.setattr(
        playback_mod, "open_device_stream", lambda **kwargs: (stream, 48000)
    )
    output = OutputStream(PlaybackConfig())
    output.update_stream_if_changed(format=SampleFormat.INT16, channels=1)
    sound = SoundInput(
        data=b"\x00\x00" * 160, rate=48000, format=SampleFormat.INT16, channels=1
    )
    write = asyncio.create_task(output.playback(volume=100, sound=sound))
    await _await_entered(stream)

    output.close_stream()

    await _assert_deferred_then_closed(stream)
    await write


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
    assert await asyncio.to_thread(stream.entered_call.wait, 10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not stream.closed.is_set()  # the read is still inside the device
    stream.release.set()
    assert await asyncio.to_thread(stream.closed.wait, 10)
    assert stream.closed_during_call is False
