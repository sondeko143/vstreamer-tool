from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import sleep
from asyncio import timeout
from dataclasses import dataclass
from functools import partial
from html import unescape
from time import time_ns
from typing import AsyncGenerator

from google.api_core.exceptions import BadRequest
from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient

from vspeech.config import GcpConfig
from vspeech.config import TranslationConfig
from vspeech.exceptions import shutdown_worker
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
        except BadRequest:
            raise
        except GoogleCloudError as e:
            logger.exception(e)
            if max_retry_count <= num_retries:
                raise e
            num_retries += 1
            await sleep(retry_delay_sec)


@dataclass
class TranslationBlock:
    text: str
    source_language_code: str | None
    target_language_code: str | None
    original: WorkerInput

    @staticmethod
    def from_transcribed_input(input: WorkerInput):
        return TranslationBlock(
            text=input.text,
            source_language_code=input.current_event.params.source_language_code,
            target_language_code=input.current_event.params.target_language_code,
            original=input,
        )

    def samey(self, other: "TranslationBlock"):
        return (
            self.source_language_code == other.source_language_code
            and self.target_language_code == other.target_language_code
        )


async def translation_worker_google(
    config: TranslationConfig, gcp_config: GcpConfig, in_queue: Queue[WorkerInput]
) -> AsyncGenerator[WorkerOutput, None]:
    credentials = get_credentials(gcp_config)
    client = TranslationServiceAsyncClient(credentials=credentials)
    logger.info("translation worker [google] started")
    while True:
        transcribed = await in_queue.get()
        logger.info("Got a chunk %s.", transcribed.input_id)
        blocks: list[TranslationBlock] = []
        blocks.append(TranslationBlock.from_transcribed_input(transcribed))
        total_awaiting_time_sec = 0
        total_n_chunks = 0
        while (
            total_awaiting_time_sec <= config.max_sec_await_total
            or total_n_chunks <= config.max_n_chunk_await_total
        ):
            try:
                await_start_time_ns = time_ns()
                async with timeout(config.sec_await_next_text):
                    next_transc = await in_queue.get()
                    next_block = TranslationBlock.from_transcribed_input(next_transc)
                    try:
                        previous = next(b for b in blocks if b.samey(next_block))
                        logger.info("Got a next translation chunk.")
                        previous.text += next_block.text
                    except StopIteration:
                        blocks.append(
                            TranslationBlock.from_transcribed_input(next_transc)
                        )
                awaited_time_ns = time_ns()
                total_awaiting_time_sec += (
                    awaited_time_ns - await_start_time_ns
                ) // 1000000000
                total_n_chunks += 1
            except TimeoutError:
                logger.info("Timeout awaiting a next translation chunk.")
                break

        requests = [
            TranslateTextRequest(
                contents=[b.text],
                source_language_code=b.source_language_code,
                target_language_code=b.target_language_code,
                parent=f"projects/{credentials.project_id}",  # type: ignore
            )
            for b in blocks
        ]
        try:
            for request, block in zip(requests, blocks):
                logger.debug("translating... %s", block.text)
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
                logger.info("translatedOutput: %s -> %s", block.text, translated)
                worker_output = WorkerOutput.from_input(block.original)
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
                translation_worker_google,
                config=context.config.translation,
                gcp_config=context.config.gcp,
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
