import logging
from asyncio import Event
from asyncio import Queue

import numpy as np
import pytest

from vspeech.lib.telemetry import telemetry
from vspeech.stream_vc.capture import CAPTURE_DROP_LOG_EVERY
from vspeech.stream_vc.capture import CaptureItem
from vspeech.stream_vc.capture import _capture_read_loop
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.capture import pcm16_to_float32
from vspeech.stream_vc.capture import should_log_capture_drop


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


def test_capture_drop_logs_first():
    assert should_log_capture_drop(1)


def test_capture_drop_log_is_rate_limited():
    # 持続 drop (block_ms=160 なら ~6 回/秒) でログを溢れさせない。
    assert not should_log_capture_drop(2)
    assert not should_log_capture_drop(CAPTURE_DROP_LOG_EVERY - 1)


def test_capture_drop_logs_every_nth():
    assert should_log_capture_drop(CAPTURE_DROP_LOG_EVERY)
    assert should_log_capture_drop(CAPTURE_DROP_LOG_EVERY * 3)


class _FakeStream:
    """hop サンプルを n_blocks 回返し、以降は device fault を模して OSError。

    `_capture_read_loop` は `while True` で device fault のときだけ抜ける
    (run_with_device_retry へ委ねる)ので、テストもその出口で止める。
    """

    def __init__(self, n_blocks: int, overflowed: bool = False) -> None:
        self.remaining = n_blocks
        self.overflowed = overflowed

    def read(self, frames: int) -> tuple[bytes, bool]:
        if self.remaining <= 0:
            raise OSError("device gone")
        self.remaining -= 1
        return (b"\x00\x00" * frames, self.overflowed)


@pytest.fixture
def enabled_telemetry():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=1000)
    yield telemetry
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


def _full_queue(hop: int) -> Queue[CaptureItem]:
    """満杯 = 以降の put は必ず最古を捨てる(= 毎ブロック drop)。"""
    q: Queue[CaptureItem] = Queue(maxsize=1)
    q.put_nowait(np.zeros(hop, dtype=np.float32))
    return q


async def test_capture_drop_while_paused_does_not_warn(caplog, enabled_telemetry):
    """pause 中の drop は設計どおり(ADR-0050)なので警告を出さない。

    pause 中は vc_loop が消費を止めるため capture_queue は満杯のままで、以降の
    ブロックは 100% drop する。ここで毎回警告すると block_ms=160 で ~6 行/秒が
    pause の間ずっと出続ける(実機で報告された症状)。
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
    # 黙って捨てるのではなく、pause 専用の stage で観測可能にしておく。
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop_paused"]["count"] == 5
    # バックプレッシャ指標(RTF 評価に使う)は pause の drop で汚さない。
    assert "stream_vc_capture_drop" not in summary


async def test_capture_drop_while_running_warns_and_throttles(
    caplog, enabled_telemetry
):
    """running 中の drop は本物のバックプレッシャ。1 回目と N 回ごとだけ出す。"""
    hop = 4
    running = Event()
    running.set()
    n = CAPTURE_DROP_LOG_EVERY + 1
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _FakeStream(n),  # ty: ignore[invalid-argument-type]
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 2  # 1 回目と CAPTURE_DROP_LOG_EVERY 回目
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == n
    assert "stream_vc_capture_drop_paused" not in summary
