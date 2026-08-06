import logging
from asyncio import Event
from asyncio import Queue

import numpy as np
import pytest

from vspeech.lib.telemetry import telemetry
from vspeech.stream_vc.capture import CaptureItem
from vspeech.stream_vc.capture import _capture_read_loop
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.capture import pcm16_to_float32


def test_ms_to_samples():
    assert ms_to_samples(80.0) == 1280  # 80ms @ 16k
    assert ms_to_samples(10.0) == 160
    assert ms_to_samples(0.0) == 0


def test_pcm16_to_float32_range():
    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    out = pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert abs(out[1] - 1.0) < 1e-3
    assert abs(out[2] + 1.0) < 1e-3


def test_pcm16_to_float32_empty():
    out = pcm16_to_float32(b"")
    assert out.shape == (0,)


class _FakeStream:
    """Returns hop samples n_blocks times, then raises OSError to mimic a device fault.

    `_capture_read_loop` is a `while True` that only exits on a device fault (delegating to
    run_with_device_retry), so the test stops at that same exit.
    """

    def __init__(self, n_blocks: int, overflowed: bool = False) -> None:
        self.remaining = n_blocks
        self.overflowed = overflowed

    def read(self, frames: int) -> tuple[bytes, bool]:
        if self.remaining <= 0:
            raise OSError("device gone")
        self.remaining -= 1
        return (b"\x00\x00" * frames, self.overflowed)


class _PausingStream(_FakeStream):
    """A mic that drops into the paused state **during** the `pause_on_read`-th read.

    It reproduces the real-hardware ordering (warn on backpressure, then a pause arrives
    and it goes quiet) within a single loop. The gate closes before that block reaches the
    queue, so the `pause_on_read`-th block is already on the paused side, and the drops on
    the running side number `pause_on_read - 1`.
    """

    def __init__(self, n_blocks: int, running: Event, pause_on_read: int) -> None:
        super().__init__(n_blocks)
        self._running = running
        self._pause_on_read = pause_on_read
        self._read_count = 0

    def read(self, frames: int) -> tuple[bytes, bool]:
        self._read_count += 1
        if self._read_count == self._pause_on_read:
            # Event.clear() wakes no waiters, so touching it from a to_thread worker
            # thread does not touch the loop (unlike set(), it uses no call_soon).
            self._running.clear()
        return super().read(frames)


@pytest.fixture
def enabled_telemetry():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=1000)
    yield telemetry
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


def _full_queue(hop: int) -> Queue[CaptureItem]:
    """Full = every subsequent put discards the oldest (i.e. a drop on every block)."""
    q: Queue[CaptureItem] = Queue(maxsize=1)
    q.put_nowait(np.zeros(hop, dtype=np.float32))
    return q


async def test_capture_drop_while_paused_does_not_warn(caplog, enabled_telemetry):
    """Drops during a pause are by design (ADR-0050), so no warning is emitted.

    While paused, vc_loop stops consuming, so capture_queue stays full and every
    subsequent block is dropped. Warning every time here would emit about 6 lines a second
    at block_ms=160 for the whole pause (the symptom reported on real hardware).
    """
    hop = 4
    running = Event()  # clear = paused
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _FakeStream(5),  # ty: ignore[invalid-argument-type]
                hop,
                _full_queue(hop),
                running,
            )
    assert not [r for r in caplog.records if "capture queue full" in r.getMessage()]
    # Rather than discarding silently, keep them observable under a pause-specific stage.
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop_paused"]["count"] == 5
    # The backpressure metric (used to assess RTF) is not polluted by pause drops.
    assert "stream_vc_capture_drop" not in summary


async def test_capture_drop_while_running_warns_once_per_episode(
    caplog, enabled_telemetry
):
    """Drops while running are real backpressure. Only one line at the head of the
    episode."""
    hop = 4
    running = Event()
    running.set()
    n = 51
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _FakeStream(n),  # ty: ignore[invalid-argument-type]
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 1  # a tight loop = all within min_interval_s
    assert "(total 1)" in warnings[0].getMessage()
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == n  # telemetry every time
    assert "stream_vc_capture_drop_paused" not in summary


async def test_capture_overflow_warns_once_per_episode_and_is_metered(
    caplog, enabled_telemetry
):
    """An input overflow persists once it starts (the reader is late), so it fires on
    every block.

    Warning unconditionally would emit about 6 lines a second at block_ms=160 and bury the
    log -- the symptom ADR-0062 exists to remove. The queue is left un-full here so the
    drop path stays quiet and only the overflow path is under test.
    """
    hop = 4
    running = Event()
    running.set()
    n = 40
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _FakeStream(n, overflowed=True),  # ty: ignore[invalid-argument-type]
                hop,
                Queue(),
                running,
            )
    warnings = [r for r in caplog.records if "input overflow" in r.getMessage()]
    assert len(warnings) == 1  # a tight loop = all within min_interval_s
    assert "(total 1)" in warnings[0].getMessage()
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_overflow"]["count"] == n  # telemetry every time
    assert "stream_vc_capture_drop" not in summary  # the queue never filled


async def test_capture_drop_switches_side_when_pause_arrives(caplog, enabled_telemetry):
    """Across a running -> pause transition, each block's drop is still attributed to the
    right side.

    Exactly the ordering that happens on real hardware (warn on backpressure, then go
    quiet on pause). This checks that no further warnings appear after the transition and
    that the drops from before the pause remain in the backpressure metric.
    """
    hop = 4
    running = Event()
    running.set()
    total_blocks = 10
    # pause happens during this read -> the 4th block is already on the paused side
    pause_on_read = 4
    running_drops = pause_on_read - 1
    stream = _PausingStream(total_blocks, running, pause_on_read=pause_on_read)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                stream,  # ty: ignore[invalid-argument-type]
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 1  # only the head of the episode on the running side
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == running_drops
    assert summary["stream_vc_capture_drop_paused"]["count"] == (
        total_blocks - running_drops
    )
