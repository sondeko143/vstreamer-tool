from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from dataclasses import dataclass
from html import unescape

from google.cloud.exceptions import GoogleCloudError
from google.cloud.translate_v3 import TranslateTextRequest
from google.cloud.translate_v3 import TranslationServiceAsyncClient
from google.oauth2.service_account import Credentials

from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import Message
from vspeech.shared_context import SharedContext
from vspeech.transcription import Transcription


@dataclass
class Translation:
    id: str
    translated: str


async def translation_worker_google(
    context: SharedContext, in_queue: Queue[Transcription]
):
    credentials = Credentials.from_service_account_file(
        context.config.gcp_credentials_file_path
    )
    client = TranslationServiceAsyncClient(credentials=credentials)
    logger.info("translation worker started")
    while True:
        transcribed = await in_queue.get()
        request = TranslateTextRequest(
            contents=[transcribed.text],
            source_language_code="ja",
            target_language_code="en",
            parent=f"projects/{context.config.gcp_project_id}",
        )
        try:
            logger.info("translating...")
            response = await client.translate_text(
                request=request, timeout=context.config.gcp_request_timeout
            )
            translated = unescape("".join([translation.translated_text for translation in response.translations]))  # type: ignore
            logger.info("translated: %s", translated)
            yield Translation(
                id=transcribed.id,
                translated=translated,
            )
        except GoogleCloudError as e:
            logger.warning(e)


async def translation_worker(
    context: SharedContext,
    in_queue: Queue[Transcription],
    out_queue: Queue[Message[Translation]],
):
    try:
        generator = translation_worker_google
        async for translation in generator(context=context, in_queue=in_queue):
            out_queue.put_nowait(
                Message(source=EventType.translation, content=translation)
            )
    except CancelledError:
        logger.debug("transcription worker cancelled")
        raise


def create_translation_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[Transcription]()
    event = EventType.translation
    context.input_queues[event] = in_queue
    config = context.config
    if config.gcp_credentials_file_path and config.enable_translation:
        return loop.create_task(
            translation_worker(
                context, in_queue=in_queue, out_queue=context.broker_queue
            ),
            name=event.name,
        )
