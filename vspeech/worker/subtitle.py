"""subtitle worker のディスパッチャ (ADR-0040)。

transcription / tts と同じく `worker_type` でバックエンドへ振る。tkinter は
TK バックエンドの中だけに閉じ込めるので、このモジュールは import しない
(ヘッドレス構成で tkinter を要求しないため)。
"""

from asyncio import Queue
from asyncio import TaskGroup
from typing import assert_never

from vspeech.config import SubtitleWorkerType
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


async def subtitle_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    worker_type = context.config.subtitle.worker_type
    if worker_type == SubtitleWorkerType.TK:
        from vspeech.worker.subtitle_tk import subtitle_tk_worker

        await subtitle_tk_worker(context, in_queue=in_queue)
    elif worker_type == SubtitleWorkerType.OBS:
        # subtitle_obs.py lands in the follow-up ADR-0040 task; this branch is
        # unreachable until then (worker_type defaults to TK). ty can't resolve
        # the not-yet-created module statically; remove this ignore once it exists.
        from vspeech.worker.subtitle_obs import (  # ty: ignore[unresolved-import]
            subtitle_obs_worker,
        )

        await subtitle_obs_worker(context, in_queue=in_queue)
    else:
        assert_never(worker_type)


def create_subtitle_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    return tg.create_task(
        subtitle_worker(context, in_queue=worker.in_queue),
        name=worker.event.name,
    )
