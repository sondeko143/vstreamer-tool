from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import sleep
from asyncio import to_thread
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator
from wave import Error as WavError
from wave import open as wav_open

from aiofiles import open as aio_open
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.speech import RecognitionAudio
from google.cloud.speech import RecognitionConfig
from google.cloud.speech import RecognizeRequest
from google.cloud.speech import SpeechAsyncClient
from grpc.aio import AioRpcError
from httpx import AsyncClient
from httpx import HTTPError
from pydantic import ValidationError

from vspeech.ami import AmiResponse
from vspeech.ami import text_removed_filler_symbol
from vspeech.config import TranscriptionWorkerType
from vspeech.config import get_sample_size
from vspeech.gcp import get_credentials
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

try:
    from vspeech.transliterate import get_transliterated_text
except ModuleNotFoundError:
    logger.info("mozc not found")

try:
    from whisper import load_model
    from whisper import transcribe
except ModuleNotFoundError:
    logger.info("whisper not found")


async def log_transcribed(log_dir_parent: Path, wav_file: BytesIO, text: str):
    now = datetime.now()
    log_dir = Path(log_dir_parent / now.strftime("%Y%m%d"))
    log_dir.mkdir(exist_ok=True, parents=True)
    log_wav_name = now.strftime("%Y%m%d%H%M%S.wav")
    log_txt_name = now.strftime("%Y%m%d%H%M%S.txt")
    async with aio_open(log_dir / log_wav_name, "wb") as log:
        wav_file.seek(0)
        await log.write(wav_file.read())
    if not text:
        return
    async with aio_open(
        log_dir / log_txt_name, "w", encoding="utf-8", errors="backslashreplace"
    ) as log:
        await log.write(text)


def wav(sound: SoundInput, sample_size: int):
    temp_wav = BytesIO()
    with wav_open(temp_wav, "wb") as waveFile:
        waveFile.setnchannels(sound.channels)
        waveFile.setsampwidth(sample_size)
        waveFile.setframerate(sound.rate)
        waveFile.writeframes(sound.data)
    temp_wav.seek(0)
    return temp_wav


async def transcribe_request_google(
    client: SpeechAsyncClient,
    request: RecognizeRequest,
    timeout: float,
    max_retry_count: int,
    retry_delay_sec: float,
):
    num_retries = 0
    while True:
        try:
            return await client.recognize(request=request, timeout=timeout)
        except (AioRpcError, GoogleAPICallError) as e:
            logger.exception(e)
            if max_retry_count <= num_retries:
                raise e
            num_retries += 1
            await sleep(retry_delay_sec)


async def transcript_worker_whisper(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[str, None]:
    model = load_model(context.config.whisper.whisper_model)
    while True:
        config = context.config.whisper
        recorded = await in_queue.get()
        try:
            sample_size = get_sample_size(recorded.sound.format)
            with wav(recorded.sound, sample_size=sample_size) as wav_file:
                async with aio_open("./recorded.wav", mode="wb") as out:
                    await out.write(wav_file.read())
                    await out.flush()
                wav_file.close()
                logger.info("transcribing...")
                result = await to_thread(
                    transcribe,
                    audio="./recorded.wav",
                    model=model,
                    language="ja",
                    no_speech_threshold=config.whisper_no_speech_prob_threshold,
                    logprob_threshold=config.whisper_logprob_threshold,
                )
                logger.info("transcribed: %s", result)
                transcribed = "".join(
                    [
                        segment["text"]
                        for segment in result["segments"]
                        if segment["no_speech_prob"]
                        < config.whisper_no_speech_prob_threshold
                        and segment["avg_logprob"] > config.whisper_logprob_threshold
                        and segment["temperature"] < 1.0
                    ]
                )
                if transcribed:
                    yield transcribed
                if context.config.recording_log:
                    await log_transcribed(
                        context.config.recording_log_dir,
                        wav_file=wav_file,
                        text=transcribed,
                    )
        except WavError as e:
            logger.warning(e)


async def transcript_worker_google(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[str, None]:
    credentials = get_credentials(context.config.gcp)
    client = SpeechAsyncClient(credentials=credentials)
    logger.info("transcript worker [google] started")
    while True:
        gcp = context.config.gcp
        recorded = await in_queue.get()
        try:
            sample_size = get_sample_size(recorded.sound.format)
            with wav(recorded.sound, sample_size=sample_size) as wav_file:
                rec_audio = RecognitionAudio(content=wav_file.read())
                rec_config = RecognitionConfig(
                    encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=context.config.recording.rate,
                    language_code="ja-JP",
                )
                logger.info("transcribing...")
                request = RecognizeRequest(config=rec_config, audio=rec_audio)
                r = await transcribe_request_google(
                    client=client,
                    request=request,
                    timeout=gcp.gcp_request_timeout,
                    max_retry_count=gcp.gcp_max_retry_count,
                    retry_delay_sec=gcp.gcp_retry_delay_sec,
                )
                transcribed = "".join([result.alternatives[0].transcript for result in r.results])  # type: ignore
                logger.info("transcribed: %s", r)
                if transcribed:
                    yield transcribed
                if context.config.recording_log:
                    await log_transcribed(
                        context.config.recording_log_dir,
                        wav_file=wav_file,
                        text=transcribed,
                    )
        except (HTTPError, AioRpcError, GoogleAPICallError) as e:
            logger.warning("transcription request error: %s", e)
        except WavError as e:
            logger.warning(e)


async def transcript_worker_ami(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[str, None]:
    logger.info("transcript worker [ami] started")
    while True:
        ami_config = context.config.ami
        async with AsyncClient(timeout=ami_config.ami_request_timeout) as client:
            recorded = await in_queue.get()
            try:
                sample_size = get_sample_size(recorded.sound.format)
                with wav(recorded.sound, sample_size=sample_size) as wav_file:
                    data = {
                        "d": f"grammarFileNames={ami_config.ami_engine_name} profileId={ami_config.ami_service_id} {ami_config.ami_extra_parameters}",
                        "u": ami_config.ami_appkey,
                    }
                    files = {"a": wav_file}
                    logger.info("transcribing...")
                    r = await client.post(
                        ami_config.ami_engine_uri, data=data, files=files
                    )
                    res_json = r.json()
                    logger.info("transcribed: %s", res_json)
                    res_body = AmiResponse.parse_obj(res_json)
                    if context.config.transcription.transliterate_with_mozc:
                        text = get_transliterated_text(res_body=res_body)
                    else:
                        text = text_removed_filler_symbol(res_body)
                    spoken = "".join(
                        [
                            "".join([token.spoken for token in result.tokens])
                            for result in res_body.results
                        ]
                    )
                    if text:
                        logger.info("transliterate: %s -> %s", spoken, text)
                        yield text
                    if context.config.recording_log:
                        await log_transcribed(
                            context.config.recording_log_dir,
                            wav_file=wav_file,
                            text=str(res_json),
                        )
            except (HTTPError, ValidationError) as e:
                logger.warning("transcription request error: %s", e)
                logger.exception(e)
            except WavError as e:
                logger.warning(e)


async def transcription_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
    out_queue: Queue[WorkerOutput],
):
    config = context.config.transcription
    if config.transcription_worker_type == TranscriptionWorkerType.ACP:
        generator = transcript_worker_ami
    elif config.transcription_worker_type == TranscriptionWorkerType.GCP:
        generator = transcript_worker_google
    elif config.transcription_worker_type == TranscriptionWorkerType.WHISPER:
        generator = transcript_worker_whisper
    else:
        raise ValueError("transcription worker type unknown.")
    try:
        async for transcription in generator(context=context, in_queue=in_queue):
            out_queue.put_nowait(
                WorkerOutput(
                    source=EventType.transcription, text=transcription, sound=None
                )
            )
    except CancelledError:
        logger.debug("transcription worker cancelled")
        raise


def create_transcription_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.transcription
    context.input_queues[event] = in_queue
    return loop.create_task(
        transcription_worker(
            context, in_queue=in_queue, out_queue=context.sender_queue
        ),
        name=event.name,
    )
