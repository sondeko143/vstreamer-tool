from asyncio import CancelledError
from asyncio import Task
from asyncio import TaskGroup
from asyncio import get_event_loop
from traceback import format_exception
from typing import IO
from typing import Any
from typing import List
from typing import Optional

import click

from vspeech.config import Config
from vspeech.exceptions import WorkerShutdown
from vspeech.logger import configure_logger
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.worker.receiver import create_receiver_task
from vspeech.worker.sender import create_sender_task


async def cancel_tasks(tasks: List[Task[Any]]):
    task_name = ", ".join([task.get_name() for task in tasks if task.done()])
    logger.warning(f"{task_name} error")
    for task in tasks:
        if not task.done():
            task.cancel()
            logger.info("awaiting end %s", task.get_name())
            try:
                await task
            except CancelledError:
                logger.info("cancelled %s", task.get_name())


async def vspeech_coro(config: Config):
    context = SharedContext(config=config)
    try:
        async with TaskGroup() as tg:
            create_sender_task(tg=tg, context=context)
            create_receiver_task(tg=tg, context=context)
            if config.recording.enable:
                from vspeech.worker.recording import create_recording_task

                create_recording_task(tg=tg, context=context)
            if config.tts.enable:
                from vspeech.worker.tts import create_tts_task

                create_tts_task(tg=tg, context=context)
            if config.transcription.enable:
                from vspeech.worker.transcription import create_transcription_task

                create_transcription_task(tg=tg, context=context)
            if config.subtitle.enable:
                from vspeech.worker.subtitle import create_subtitle_task

                create_subtitle_task(tg=tg, context=context)
            if config.translation.enable:
                from vspeech.worker.translation import create_translation_task

                create_translation_task(tg=tg, context=context)
            if config.playback.enable:
                from vspeech.worker.playback import create_playback_task

                create_playback_task(tg=tg, context=context)
            if config.vc.enable:
                from vspeech.worker.vc import create_vc_task

                create_vc_task(tg=tg, context=context)
    except* WorkerShutdown as eg:
        for e in eg.exceptions:
            logger.warning("workers shutdown: %s", e.args)
            logger.debug("".join(format_exception(e)))


@click.command()
@click.option(
    "--config",
    "--json-config",
    "config_file",
    type=click.File("rb"),
)
def cmd(config_file: Optional[IO[bytes]]):
    if config_file:
        config = Config.read_config_from_file(config_file)
        config_file.close()
    else:
        # from environment variables
        config = Config()
    configure_logger(config)
    loop = get_event_loop()
    try:
        loop.run_until_complete(vspeech_coro(config=config))
        loop.stop()
        loop.close()
        exit(1)
    except (KeyboardInterrupt, CancelledError) as e:
        logger.exception(e)
        logger.debug("catch keyboard interrupt")
        loop.stop()
        loop.close()
        exit(0)
    except Exception as e:
        logger.exception(e)
        exit(1)
