"""streaming VC の device 自力再接続ループ(vspeech/stream_vc/retry.py)。

pure な next_backoff と、run_with_device_retry の再接続/fail-loud/cancellation の
振る舞いを、実デバイス無しの fake stream で CPU 検証する。
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
    assert seen[1] == BACKOFF_START * 2  # 倍々で増える
    assert max(seen) == BACKOFF_MAX  # 最終的に頭打ち
    assert next_backoff(BACKOFF_MAX) == BACKOFF_MAX  # 上限で clamp


class _FakeStream:
    """read が仕込んだ列(例外 or 値)を消化する device 代役。空になると永久ブロック。"""

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
        await asyncio.Event().wait()  # 列を使い切ったら cancel されるまで待つ


async def _nosleep(_: float) -> None:
    return


async def _read_loop(stream: _FakeStream) -> None:
    while True:
        await stream.read()


@pytest.mark.parametrize(
    "fault", [OSError("mic unplugged"), sd.PortAudioError("format changed")]
)
async def test_reopens_on_device_error_and_cancellation_propagates(fault):
    """steady-state の (OSError, PortAudioError) は close→再 open で回復し、
    CancelledError は握らず WorkerShutdown で propagate する。"""
    opened: list[_FakeStream] = []

    def open_stream() -> _FakeStream:
        # 1つ目の read は即 device fault、2つ目以降は永久ブロック。
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
    assert len(opened) == 2  # 初回 open + fault 後の再 open
    assert opened[0].closed >= 1  # fault 時に閉じている
    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_first_open_failure_is_fail_loud():
    """初回 open の失敗は worker_startup で WorkerStartupError 化(無限 retry しない)。"""

    def open_stream() -> _FakeStream:
        raise OSError("no such device")

    async def run(stream: _FakeStream) -> None:
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
    """DEVICE_ERRORS 以外(ここでは ValueError)は捕えず、再 open もしない。"""
    opened: list[_FakeStream] = []

    def open_stream() -> _FakeStream:
        s = _FakeStream([])
        opened.append(s)
        return s

    async def run(stream: _FakeStream) -> None:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        await run_with_device_retry(
            open_stream=open_stream,
            run=run,
            worker="stream_vc",
            label="test",
            sleep=_nosleep,
        )
    assert len(opened) == 1  # 再 open していない
