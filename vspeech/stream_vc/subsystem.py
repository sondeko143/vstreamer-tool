"""streaming VC サブシステムの配線(ADR-0050)。

Command/routing の外の自己完結サブシステム。capture(独立マイク)→ 変換 →
transport → 連続再生 を内側 TaskGroup で束ね、1 タスクとして起動する。
`context.add_worker`/`sender_queue` は使わない(発話系 routing に一切載らない)。
重い import(sounddevice/torch を引く capture/runner/playback)は起動時に遅延
させ、このモジュール自体は CPU から import できるようにする。
"""

from asyncio import CancelledError
from asyncio import Event
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from typing import Any
from uuid import uuid4

from vspeech.exceptions import WorkerShutdown
from vspeech.exceptions import shutdown_worker
from vspeech.logger import logger
from vspeech.shared_context import SharedContext


def _iter_leaves(exc: BaseException):
    """(ネストした)例外グループを葉の例外へ平坦化する。"""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _iter_leaves(sub)
    else:
        yield exc


async def _stream_vc_subsystem(context: SharedContext) -> None:
    from vspeech.stream_vc.capture import capture_loop
    from vspeech.stream_vc.capture import ms_to_samples
    from vspeech.stream_vc.playback import playback_loop
    from vspeech.stream_vc.runner import vc_loop
    from vspeech.stream_vc.transport import InProcessTransport

    sv_config = context.config.stream_vc
    hop = ms_to_samples(sv_config.block_ms)
    session_id = uuid4().hex
    capture_queue: Queue[Any] = Queue(maxsize=sv_config.max_queued_blocks)
    transport = InProcessTransport(max_queued=sv_config.max_queued_blocks)
    vc_ready = Event()
    try:
        async with TaskGroup() as tg:
            tg.create_task(
                capture_loop(sv_config, capture_queue, hop, vc_ready),
                name="stream_vc_capture",
            )
            tg.create_task(
                vc_loop(
                    context, sv_config, capture_queue, transport, session_id, vc_ready
                ),
                name="stream_vc_runner",
            )
            tg.create_task(
                playback_loop(sv_config, transport), name="stream_vc_playback"
            )
    except CancelledError as e:
        raise shutdown_worker(e)
    except BaseExceptionGroup as eg:
        # 子タスク(capture/vc/playback)の失敗で内側 TaskGroup が abort した。
        # ストリーミングは opt-in で有効化する機能なので、その unrecoverable な
        # 障害は握らずプロセスごと落とす(fail-loud → daemon 再起動; ADR-0050)。
        # ただし TaskGroup は原因を BaseExceptionGroup に畳み、そこには cancel され
        # た兄弟の WorkerShutdown も混じる。真の原因(RuntimeError / ORT の Fail /
        # WorkerStartupError 等)が集約ノイズに埋もれないよう、ここで一度だけ明示
        # ログしてから **そのまま再送出** する(swallow も restart もしない)。
        # 兄弟の WorkerShutdown は cancel の産物であって原因ではないので除外する。
        # 純粋な shutdown(全部 WorkerShutdown)なら causes は空 = 追加ログ無しで、
        # main の except* WorkerShutdown が通常どおり処理する。
        causes = [e for e in _iter_leaves(eg) if not isinstance(e, WorkerShutdown)]
        for cause in causes:
            logger.error(
                "stream_vc subsystem: unrecoverable fault in an explicitly-enabled "
                "feature — failing the whole process (fail-loud, a daemon restarts "
                "it): %r",
                cause,
            )
        raise


def create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task[None]:
    return tg.create_task(_stream_vc_subsystem(context), name="stream_vc")
