from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import sleep
from html import unescape
from typing import AsyncGenerator

from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient
from google.oauth2.service_account import Credentials

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
) -> AsyncGenerator[str, None]:
    credentials = Credentials.from_service_account_file(
        context.config.gcp.gcp_credentials_file_path
    )
    client = TranslationServiceAsyncClient(credentials=credentials)
    logger.info("translation worker started")
    while True:
        config = context.config.translation
        gcp = context.config.gcp
        transcribed = await in_queue.get()
        request = TranslateTextRequest(
            contents=[transcribed.text],
            source_language_code=config.source_language_code,
            target_language_code=config.target_language_code,
            parent=f"projects/{gcp.gcp_project_id}",
        )
        try:
            logger.info("translating... %s", transcribed.text)
            response = await translate_request(
                client=client,
                request=request,
                timeout=gcp.gcp_request_timeout,
                max_retry_count=gcp.gcp_max_retry_count,
                retry_delay_sec=gcp.gcp_retry_delay_sec,
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
            yield translated
        except GoogleCloudError as e:
            logger.exception(e)


async def translation_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
    out_queue: Queue[WorkerOutput],
):
    try:
        generator = translation_worker_google
        async for translated in generator(context=context, in_queue=in_queue):
            out_queue.put_nowait(
                WorkerOutput(source=EventType.translation, text=translated, sound=None)
            )
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
    return loop.create_task(
        translation_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
