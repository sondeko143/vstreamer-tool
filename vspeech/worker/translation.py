from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import sleep
from html import unescape
from typing import AsyncGenerator

from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient

from vspeech.lib.gcp import get_credentials
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def translate_request(
    client: TranslationServiceAsyncClient,
    request: TranslateTextRequest,
    timeout: float,
    max_retry_count: int,
    retry_delay_sec: float,
):
    num_retries = 0
    while True:
        try:
            return await client.translate_text(request=request, timeout=timeout)
        except GoogleCloudError as e:
            logger.exception(e)
            if max_retry_count <= num_retries:
                raise e
            num_retries += 1
            await sleep(retry_delay_sec)


async def translation_worker_google(
    context: SharedContext, in_queue: Queue[WorkerInput]
) -> AsyncGenerator[WorkerOutput, None]:
    credentials = get_credentials(context.config.gcp)
    client = TranslationServiceAsyncClient(credentials=credentials)
    logger.info("translation worker [google] started")
    while True:
        config = context.config.translation
        gcp = context.config.gcp
        transcribed = await in_queue.get()
        request = TranslateTextRequest(
            contents=[transcribed.text],
            source_language_code=config.source_language_code,
            target_language_code=config.target_language_code,
            parent=f"projects/{gcp.project_id}",
        )
        try:
            logger.info("translating... %s", transcribed.text)
            response = await translate_request(
                client=client,
                request=request,
                timeout=gcp.request_timeout,
                max_retry_count=gcp.max_retry_count,
                retry_delay_sec=gcp.retry_delay_sec,
            )
            translated = unescape(
                "".join(
                    [
                        translation.translated_text
                        for translation in response.translations
                    ]
                )
            )
            logger.info("translatedOutput: %s -> %s", transcribed.text, translated)
            worker_output = WorkerOutput.from_input(transcribed)
            worker_output.text = translated
            yield worker_output
        except GoogleCloudError as e:
            logger.exception(e)


async def translation_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
    out_queue: Queue[WorkerOutput],
):
    while True:
        context.reset_need_reload()
        generator = translation_worker_google
        try:
            async for translated in generator(context=context, in_queue=in_queue):
                out_queue.put_nowait(translated)
                if context.need_reload:
                    break
        except CancelledError:
            logger.debug("transcription worker cancelled")
            raise


def create_translation_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.translation
    context.input_queues[event] = in_queue
    task = loop.create_task(
        translation_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
