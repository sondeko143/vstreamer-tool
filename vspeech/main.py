from asyncio import FIRST_COMPLETED
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Task
from asyncio import get_event_loop
from asyncio import wait
from typing import Any
from typing import List

from vspeech.broker import create_broker_worker
from vspeech.comm import create_comm_worker
from vspeech.config import Config
from vspeech.logger import configure_logger
from vspeech.logger import logger
from vspeech.recording import create_recording_task
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.speech import create_speech_task
from vspeech.subtitle import create_subtitle_task
from vspeech.transcription import create_transcription_worker
from vspeech.translation import create_translation_task


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
    context = SharedContext(
        config=config,
        event_routing={
            EventType.recording: [EventType.transcription],
            EventType.translation: [EventType.subtitle_translated],
            EventType.transcription: [
                EventType.speech,
                EventType.translation,
                EventType.subtitle,
            ],
        },
    )
    tasks = [
        create_recording_task(loop=loop, context=context),
        create_transcription_worker(loop=loop, context=context),
        create_speech_task(loop=loop, context=context),
        create_translation_task(loop=loop, context=context),
        create_subtitle_task(loop=loop, context=context),
        create_comm_worker(loop=loop, context=context),
        create_broker_worker(loop=loop, context=context),
    ]
    tasks = [task for task in tasks if task]
    try:
        await wait(tasks, return_when=FIRST_COMPLETED)
    except CancelledError:
        logger.info("task cancelled")
    except Exception as e:
        logger.exception(e)
    finally:
        context.audio.terminate()
        await cancel_tasks(tasks)


def cmd(config: Config) -> int:
    logger.debug(config)
    configure_logger(config)
    loop = get_event_loop()
    try:
        loop.run_until_complete(vspeech_coro(loop=loop, config=config))
        logger.debug("coro ended.")
        loop.stop()
        loop.close()
        return 1
    except (KeyboardInterrupt, CancelledError) as e:
        logger.exception(e)
        logger.debug("catch keyboard interrupt")
        loop.stop()
        loop.close()
        return 0
    except Exception as e:
        logger.exception(e)
        return 1
    finally:
        logger.debug("main loop finally")
