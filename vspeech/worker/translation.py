from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import sleep
from asyncio import timeout
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from html import unescape
from time import time_ns

from google.api_core.exceptions import BadRequest
from google.auth.credentials import Credentials as BaseCredentials
from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient
from google.cloud.translate_v3.services.translation_service.transports import (
    TranslationServiceGrpcAsyncIOTransport,
)

from vspeech.config import GcpConfig
from vspeech.config import TranslationConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.gcp import GAPIC_DEFAULT_CHANNEL_OPTIONS
from vspeech.lib.gcp import create_auth_channel
from vspeech.lib.gcp import get_credentials
from vspeech.lib.telemetry import telemetry
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
            if max_retry_count <= num_retries:
                raise
            num_retries += 1
            # Transient (e.g. a 503 when an idle gap resets the auth-metadata
            # connection) — we retry, so log ONE concise WARNING line instead of
            # a full ERROR traceback that floods the log on every idle reconnect.
            # A genuine failure (retries exhausted) still surfaces via the
            # caller's `except GoogleCloudError` with a full traceback.
            logger.warning(
                "translation transient error, retrying (%d/%d): %s",
                num_retries,
                max_retry_count,
                e,
            )
            await sleep(retry_delay_sec)


def create_translation_client(
    credentials: BaseCredentials,
) -> TranslationServiceAsyncClient:
    """翻訳クライアントを、トークン更新が retry される認証チャネルの上に作る。

    `TranslationServiceAsyncClient(credentials=...)` に任せると api_core が
    `Request()` を引数無しで作り、トークン更新が retry 無しの素の
    `requests.Session` で走る。約 1 時間 idle した接続をプールから掴んで
    ConnectionReset で落ちる窓がそこにあるので (`vspeech.lib.gcp` の
    `_AUTH_RETRY` 参照)、チャネルだけこちらで組んで注入する。

    transport に `aio.Channel` の実体を渡すと、ライブラリ側は credentials を
    無視してそのチャネルを使う (`_ignore_credentials`) -- つまり認証経路は
    完全にこちらの持ち物になる。
    """
    transport = TranslationServiceGrpcAsyncIOTransport
    channel = create_auth_channel(
        credentials,
        host=transport.DEFAULT_HOST,
        scopes=transport.AUTH_SCOPES,
        options=GAPIC_DEFAULT_CHANNEL_OPTIONS,
    )
    return TranslationServiceAsyncClient(transport=transport(channel=channel))


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

    def samey(self, other: TranslationBlock):
        return (
            self.source_language_code == other.source_language_code
            and self.target_language_code == other.target_language_code
        )


async def translation_worker_google(
    config: TranslationConfig, gcp_config: GcpConfig, in_queue: Queue[WorkerInput]
) -> AsyncGenerator[WorkerOutput]:
    with worker_startup("translation"):
        credentials, project_id = get_credentials(gcp_config)
        client = create_translation_client(credentials)
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
                parent=f"projects/{project_id}",
            )
            for b in blocks
        ]
        try:
            for request, block in zip(requests, blocks):
                logger.debug("translating... %s", block.text)
                with telemetry.timer("translation", trace_id=block.original.trace_id):
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
