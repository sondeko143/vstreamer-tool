from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import sleep
from functools import partial
from html import unescape
from typing import AsyncGenerator

from google.api_core.exceptions import BadRequest
from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient

from vspeech.config import GcpConfig
from vspeech.exceptions import shutdown_worker
from vspeech.lib.gcp import get_credentials
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerMeta
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
        except BadRequest:
            raise
        except GoogleCloudError as e:
            logger.exception(e)
            if max_retry_count <= num_retries:
                raise e
            num_retries += 1
            await sleep(retry_delay_sec)


async def translation_worker_google(
    gcp_config: GcpConfig, in_queue: Queue[WorkerInput]
) -> AsyncGenerator[WorkerOutput, None]:
    credentials = get_credentials(gcp_config)
    client = TranslationServiceAsyncClient(credentials=credentials)
    logger.info("translation worker [google] started")
    while True:
        transcribed = await in_queue.get()
        request = TranslateTextRequest(
            contents=[transcribed.text],
            source_language_code=transcribed.current_event.params.source_language_code,
            target_language_code=transcribed.current_event.params.target_language_code,
            parent=f"projects/{credentials.project_id}",  # type: ignore
        )
        try:
            logger.debug("translating... %s", transcribed.text)
            response = await translate_request(
                client=client,
                request=request,
                timeout=gcp_config.request_timeout,
                max_retry_count=gcp_config.max_retry_count,
                retry_delay_sec=gcp_config.retry_delay_sec,
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
    try:
        while True:
            context.reset_need_reload()
            generator = partial(
                translation_worker_google, gcp_config=context.config.gcp
            )
            async for translated in generator(in_queue=in_queue):
                out_queue.put_nowait(translated)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError as e:
        raise shutdown_worker(e)


def create_translation_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(
        event=EventType.translation, configs_depends_on=["translation", "gcp"]
    )
    task = tg.create_task(
        translation_worker(
            context, in_queue=worker.in_queue, out_queue=context.sender_queue
        ),
        name=worker.event.name,
    )
    return task
