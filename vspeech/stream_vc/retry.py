"""streaming VC の device 自力再接続ループ(ADR-0050)。

capture/playback の steady-state で起きる runtime device fault
(OSError / PortAudioError = マイク抜け・フォーマット変更・出力先消失など)を、
兄弟タスクや発話系パイプラインを巻き込まずにサブシステム内で吸収する。

- **初回 open は fail-loud**(worker_startup で WorkerStartupError 化 = ADR-0038)。
  モデル/デバイス不在は設定不備であって、無限 retry で隠すべきではない。
- **steady-state の device fault だけ** catch して close→backoff→再 open する。
  発話系 recording worker(vspeech/worker/recording.py)の
  `(OSError, sd.PortAudioError)` retry パターンを踏襲する(あちらは無改変で再利用)。
- **CancelledError は握らない**。bare except / except Exception は使わず
  DEVICE_ERRORS のみ捕える(cancellation は必ず propagate → shutdown_worker)。

このモジュールは capture/playback からのみ lazy import される(subsystem.py は
import しない)ので、sounddevice を引いても subsystem の CPU-light 性は保たれる。
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

# steady-state で自力回復する device fault。CUDA 例外(RuntimeError 系)や
# CancelledError は **含めない** — 前者は runner が別に扱い、後者は propagate する。
DEVICE_ERRORS = (OSError, sd.PortAudioError)

# 再接続バックオフ(秒)。start から factor 倍で増え MAX で頭打ち。
BACKOFF_START = 0.5
BACKOFF_MAX = 5.0
BACKOFF_FACTOR = 2.0


def next_backoff(prev: float) -> float:
    """指数バックオフの次値(BACKOFF_MAX で clamp)。pure(CPU テスト対象)。"""
    return min(prev * BACKOFF_FACTOR, BACKOFF_MAX)


class _Closable(Protocol):
    def close(self) -> None: ...


def close_quietly(stream: _Closable) -> None:
    """stream.close() の device 例外を握り潰す。

    既に壊れた/閉じたデバイスを二重 close する経路(fault→close→finally 再close)が
    あるので、close 自体の DEVICE_ERRORS はログだけ残して無視する。
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
    """backoff を挟みながら open_stream() が成功するまで再試行して返す。

    再 open は **runtime retry** なので worker_startup で包まない(fail-loud に
    しない)。open 自体が DEVICE_ERRORS で失敗しても backoff を伸ばして粘る。
    CancelledError は sleep から素通しで propagate する。
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
    """初回 open → steady-state → device fault で再接続、を回す device ループ。

    - `open_stream`: ストリームを開いて返す。初回だけ worker_startup で包んで
      fail-loud にする(それ以降の再 open は runtime retry)。
    - `run`: 与えたストリームで steady-state を回す coroutine を返す。device
      fault(DEVICE_ERRORS)を投げて戻ってきたら close→backoff→再 open する。
    - `on_reopen`: 再 open の直前に呼ぶフック(per-connection state のリセット等)。
    - `reopen_metric`: 与えると再接続ごとに telemetry へ 1.0 を記録する。

    CancelledError は捕らえず shutdown_worker で包んで送出する。
    """
    with worker_startup(worker):
        stream = open_stream()  # 初回のみ fail-loud (ADR-0038)
    logger.info("%s started", label)
    try:
        while True:
            try:
                await run(stream)
            except DEVICE_ERRORS as e:
                # runtime device fault。サブシステム内で吸収し、発話系や兄弟タスクを
                # 巻き込まない(ADR-0050)。
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
