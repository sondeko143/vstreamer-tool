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

from vspeech.config import Config
from vspeech.config import TranscriptionWorkerType
from vspeech.config import get_sample_size
from vspeech.lib.ami import parse_response
from vspeech.lib.gcp import get_credentials
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def log_transcribed(log_dir_parent: Path, wav_file: BytesIO, text: str):
    now = datetime.now()
    log_dir = Path(log_dir_parent.expanduser() / now.strftime("%Y%m%d"))
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
    config: Config,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    from whisper import load_model
    from whisper import transcribe

    whisper_config = config.whisper
    model = load_model(whisper_config.model)
    while True:
        recorded = await in_queue.get()
        try:
            sample_size = get_sample_size(recorded.sound.format)
            with wav(recorded.sound, sample_size=sample_size) as wav_file:
                async with aio_open("./recorded.wav", mode="wb") as out:
                    await out.write(wav_file.read())
                    await out.flush()
                logger.info("transcribing...")
                result = await to_thread(
                    transcribe,
                    audio="./recorded.wav",
                    model=model,
                    language="ja",
                    no_speech_threshold=whisper_config.no_speech_prob_threshold,
                    logprob_threshold=whisper_config.logprob_threshold,
                )
                logger.info("transcribed: %s", result)
                transcribed = "".join(
                    [
                        segment["text"]
                        for segment in result["segments"]
                        if segment["no_speech_prob"]
                        < whisper_config.no_speech_prob_threshold
                        and segment["avg_logprob"] > whisper_config.logprob_threshold
                        and segment["temperature"] < 1.0
                    ]
                )
                if transcribed:
                    worker_output = WorkerOutput.from_input(recorded)
                    worker_output.sound = SoundOutput(
                        data=recorded.sound.data,
                        rate=recorded.sound.rate,
                        format=recorded.sound.format,
                        channels=recorded.sound.channels,
                    )
                    worker_output.text = transcribed
                    yield worker_output
                if config.recording_log:
                    await log_transcribed(
                        config.recording_log_dir,
                        wav_file=wav_file,
                        text=transcribed,
                    )
        except WavError as e:
            logger.warning(e)


async def transcript_worker_google(
    config: Config,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    credentials = get_credentials(config.gcp)
    client = SpeechAsyncClient(credentials=credentials)
    logger.info("transcript worker [google] started")
    gcp = config.gcp
    while True:
        recorded = await in_queue.get()
        try:
            sample_size = get_sample_size(recorded.sound.format)
            with wav(recorded.sound, sample_size=sample_size) as wav_file:
                rec_audio = RecognitionAudio(content=wav_file.read())
                rec_config = RecognitionConfig(
                    encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=config.recording.rate,
                    language_code="ja-JP",
                )
                logger.info("transcribing...")
                request = RecognizeRequest(config=rec_config, audio=rec_audio)
                r = await transcribe_request_google(
                    client=client,
                    request=request,
                    timeout=gcp.request_timeout,
                    max_retry_count=gcp.max_retry_count,
                    retry_delay_sec=gcp.retry_delay_sec,
                )
                transcribed = "".join(
                    [result.alternatives[0].transcript for result in r.results]
                )
                logger.info("transcribed: %s", r)
                if transcribed:
                    worker_output = WorkerOutput.from_input(recorded)
                    worker_output.sound = SoundOutput(
                        data=recorded.sound.data,
                        rate=recorded.sound.rate,
                        format=recorded.sound.format,
                        channels=recorded.sound.channels,
                    )
                    worker_output.text = transcribed
                    yield worker_output
                if config.recording_log:
                    await log_transcribed(
                        config.recording_log_dir,
                        wav_file=wav_file,
                        text=transcribed,
                    )
        except (HTTPError, AioRpcError, GoogleAPICallError) as e:
            logger.warning("transcription request error: %s", e)
        except WavError as e:
            logger.warning(e)


async def transcript_worker_ami(
    config: Config,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    logger.info("transcript worker [ami] started")
    ami_config = config.ami
    while True:
        async with AsyncClient(timeout=ami_config.request_timeout) as client:
            recorded = await in_queue.get()
            try:
                sample_size = get_sample_size(recorded.sound.format)
                with wav(recorded.sound, sample_size=sample_size) as wav_file:
                    data = {
                        "d": f"grammarFileNames={ami_config.engine_name} "
                        f"profileId={ami_config.service_id} "
                        f"{ami_config.extra_parameters}",
                        "u": ami_config.appkey.get_secret_value(),
                    }
                    files = {"a": wav_file}
                    logger.info("transcribing...")
                    r = await client.post(ami_config.engine_uri, data=data, files=files)
                    res_json = r.json()
                    logger.info("transcribed: %s", res_json)
                    text, spoken = parse_response(
                        res_json,
                        use_mozc=config.transcription.transliterate_with_mozc,
                    )
                    if text:
                        logger.info("transliterate: %s -> %s", spoken, text)
                        worker_output = WorkerOutput.from_input(recorded)
                        worker_output.sound = SoundOutput(
                            data=recorded.sound.data,
                            rate=recorded.sound.rate,
                            format=recorded.sound.format,
                            channels=recorded.sound.channels,
                        )
                        worker_output.text = text
                        yield worker_output
                    if config.recording_log:
                        await log_transcribed(
                            config.recording_log_dir,
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
    try:
        while True:
            context.reset_need_reload()
            config = context.config.transcription
            if config.worker_type == TranscriptionWorkerType.ACP:
                generator = transcript_worker_ami
            elif config.worker_type == TranscriptionWorkerType.GCP:
                generator = transcript_worker_google
            elif config.worker_type == TranscriptionWorkerType.WHISPER:
                generator = transcript_worker_whisper
            else:
                raise ValueError("transcription worker type unknown.")
            async for transcribed in generator(
                config=context.config, in_queue=in_queue
            ):
                out_queue.put_nowait(transcribed)
                if context.need_reload:
                    break
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
    task = loop.create_task(
        transcription_worker(
            context, in_queue=in_queue, out_queue=context.sender_queue
        ),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
