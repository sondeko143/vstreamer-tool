"""The subtitle worker's dispatcher (ADR-0040).

Dispatches to a backend by `worker_type`, just like transcription / tts. tkinter is
confined to the TK backend, so this module does not import it (so a headless
configuration does not require tkinter).
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
        from vspeech.worker.subtitle_obs import subtitle_obs_worker

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
