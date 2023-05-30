from asyncio import FIRST_COMPLETED
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Task
from asyncio import get_event_loop
from asyncio import wait
from typing import IO
from typing import Any
from typing import List
from typing import Optional

import click

from vspeech.config import Config
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


async def vspeech_coro(loop: AbstractEventLoop, config: Config):
    context = SharedContext(config=config)
    tasks = [
        create_sender_task(loop=loop, context=context),
        create_receiver_task(loop=loop, context=context),
    ]
    if config.recording.enable:
        from vspeech.worker.recording import create_recording_task

        tasks.append(create_recording_task(loop=loop, context=context))
    if config.tts.enable:
        from vspeech.worker.tts import create_tts_task

        tasks.append(create_tts_task(loop=loop, context=context))
    if config.transcription.enable:
        from vspeech.worker.transcription import create_transcription_task

        tasks.append(create_transcription_task(loop=loop, context=context))
    if config.subtitle.enable:
        from vspeech.worker.subtitle import create_subtitle_task

        tasks.append(create_subtitle_task(loop=loop, context=context))
    if config.translation.enable:
        from vspeech.worker.translation import create_translation_task

        tasks.append(create_translation_task(loop=loop, context=context))
    if config.playback.enable:
        from vspeech.worker.playback import create_playback_task

        tasks.append(create_playback_task(loop=loop, context=context))
    if config.vc.enable:
        from vspeech.worker.vc import create_vc_task

        tasks.append(create_vc_task(loop=loop, context=context))
    tasks = [task for task in tasks if task]
    try:
        await wait(tasks, return_when=FIRST_COMPLETED)
    except CancelledError:
        logger.info("task cancelled")
    except Exception as e:
        logger.exception(e)
    finally:
        await cancel_tasks(tasks)


@click.command()
@click.option(
    "--config",
    "--json-config",
    "config_file",
    type=click.File("rb"),
)
def cmd(config_file: Optional[IO[bytes]]) -> int:
    if config_file:
        config = Config.read_config_from_file(config_file)
        config_file.close()
    else:
        # from environment variables
        config = Config()
    configure_logger(config)
    loop = get_event_loop()
    try:
        loop.run_until_complete(vspeech_coro(loop=loop, config=config))
        logger.debug("coro ended.")
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
    finally:
        logger.debug("main loop finally")
