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

from vspeech.config import StreamVcConfig
from vspeech.config import StreamVcRole
from vspeech.exceptions import WorkerShutdown
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.stream_vc.transport import Transport


def _iter_leaves(exc: BaseException):
    """(ネストした)例外グループを葉の例外へ平坦化する。"""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _iter_leaves(sub)
    else:
        yield exc


def loops_for_role(role: StreamVcRole) -> frozenset[str]:
    """role が起動するループ名の集合(純関数=分岐の唯一の権威, ADR-0055)。"""
    if role is StreamVcRole.producer:
        return frozenset({"capture", "vc"})
    if role is StreamVcRole.consumer:
        return frozenset({"playback"})
    return frozenset({"capture", "vc", "playback"})  # local


async def _build_transport(sv_config: StreamVcConfig) -> Transport:
    """role から transport を作る。UDP endpoint 生成は async。

    bind/接続失敗は worker_startup で fail-loud(設定不備を隠さない, ADR-0038)。
    role=producer/consumer で transport_type が udp でない設定は preflight で弾く
    (role≠local ⇒ udp 必須)。2 つ目の網 transport(TCP/bidi)が来たら、下の
    producer/consumer の中で transport_type を見て分岐する。
    """
    role = sv_config.role
    if role is StreamVcRole.local:
        from vspeech.stream_vc.transport import InProcessTransport

        return InProcessTransport(max_queued=sv_config.max_queued_blocks)
    with worker_startup("stream_vc"):
        if role is StreamVcRole.producer:
            from vspeech.stream_vc.udp import create_udp_producer_transport

            peer_host = sv_config.peer_host
            peer_port = sv_config.peer_port
            if peer_host is None or peer_port is None:
                raise ValueError(
                    "stream_vc.role=producer requires peer_host and peer_port"
                )
            return await create_udp_producer_transport(peer_host, peer_port)
        from vspeech.stream_vc.udp import create_udp_consumer_transport

        bind_port = sv_config.bind_port
        if bind_port is None:
            raise ValueError("stream_vc.role=consumer requires bind_port")
        return await create_udp_consumer_transport(
            sv_config.bind_host, bind_port, sv_config.max_queued_blocks
        )


async def _stream_vc_subsystem(context: SharedContext) -> None:
    sv_config = context.config.stream_vc
    role = sv_config.role
    runs = loops_for_role(role)
    session_id = uuid4().hex
    transport = await _build_transport(sv_config)
    try:
        async with TaskGroup() as tg:
            if "capture" in runs or "vc" in runs:
                from vspeech.stream_vc.capture import capture_loop
                from vspeech.stream_vc.capture import ms_to_samples
                from vspeech.stream_vc.runner import vc_loop

                hop = ms_to_samples(sv_config.block_ms)
                capture_queue: Queue[Any] = Queue(maxsize=sv_config.max_queued_blocks)
                vc_ready = Event()
                tg.create_task(
                    capture_loop(sv_config, capture_queue, hop, vc_ready),
                    name="stream_vc_capture",
                )
                tg.create_task(
                    vc_loop(
                        context,
                        sv_config,
                        capture_queue,
                        transport,
                        session_id,
                        vc_ready,
                    ),
                    name="stream_vc_runner",
                )
            if role is StreamVcRole.local:
                from vspeech.stream_vc.playback import playback_loop

                tg.create_task(
                    playback_loop(sv_config, transport), name="stream_vc_playback"
                )
            elif role is StreamVcRole.consumer:
                from vspeech.stream_vc.consumer import network_playback_loop

                tg.create_task(
                    network_playback_loop(sv_config, transport),
                    name="stream_vc_playback",
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
    finally:
        transport.close()


def create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task[None]:
    return tg.create_task(_stream_vc_subsystem(context), name="stream_vc")
