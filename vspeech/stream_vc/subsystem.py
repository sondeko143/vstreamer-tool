"""streaming VC サブシステムの配線(ADR-0050)。

Command/routing の外の自己完結サブシステム。capture(独立マイク)→ 変換 →
transport → 連続再生 を内側 TaskGroup で束ね、1 タスクとして起動する。
`context.add_worker`/`sender_queue` は使わない(発話系 routing に一切載らない)。
重い import(sounddevice/torch を引く capture/runner/playback)は起動時に遅延
させ、このモジュール自体は CPU から import できるようにする。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from typing import Any
from uuid import uuid4

from vspeech.exceptions import shutdown_worker
from vspeech.shared_context import SharedContext


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
    try:
        async with TaskGroup() as tg:
            tg.create_task(
                capture_loop(sv_config, capture_queue, hop), name="stream_vc_capture"
            )
            tg.create_task(
                vc_loop(sv_config, capture_queue, transport, session_id),
                name="stream_vc_runner",
            )
            tg.create_task(
                playback_loop(sv_config, transport), name="stream_vc_playback"
            )
    except CancelledError as e:
        raise shutdown_worker(e)


def create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task[None]:
    return tg.create_task(_stream_vc_subsystem(context), name="stream_vc")
