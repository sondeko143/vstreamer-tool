from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4
from wave import open as wav_open

from aiofiles import open as aio_open
from google.cloud.speech import RecognitionAudio
from google.cloud.speech import RecognitionConfig
from google.cloud.speech import SpeechAsyncClient
from google.oauth2.service_account import Credentials
from httpx import AsyncClient
from httpx import HTTPError
from whisper import load_model
from whisper import transcribe

from vspeech.config import Config
from vspeech.config import TranscriptionWorkerType
from vspeech.logger import logger
from vspeech.recording import RecordingOutput
from vspeech.shared_context import EventType
from vspeech.shared_context import Message
from vspeech.shared_context import SharedContext


@dataclass
class Transcription:
    id: str
    text: str
    spoken: str


def wav(frames: bytes, config: Config, sample_size: int):
    temp_wav = BytesIO()
    with wav_open(temp_wav, "wb") as waveFile:
        waveFile.setnchannels(config.channels)
        waveFile.setsampwidth(sample_size)
        waveFile.setframerate(config.rate)
        waveFile.writeframes(frames)
    temp_wav.seek(0)
    return temp_wav


async def transcript_worker_whisper(
    context: SharedContext,
    in_queue: Queue[RecordingOutput],
):
    sample_size = context.audio.get_sample_size(context.config.format)
    model = load_model(context.config.whisper_model)
    while True:
        recorded_frames = await in_queue.get()
        with wav(recorded_frames, context.config, sample_size=sample_size) as wav_file:
            async with aio_open("./recorded.wav", mode="wb") as out:
                await out.write(wav_file.read())
                await out.flush()
            wav_file.close()
            logger.info("transcribing...")
            result = await to_thread(
                transcribe, audio="./recorded.wav", model=model, language="ja"
            )
            logger.info("transcribed: %s", result)
            transcribed = "".join(
                [
                    segment["text"]
                    for segment in result["segments"]
                    if segment["no_speech_prob"] < 0.8
                ]
            )
            if transcribed:
                transcription_id = str(uuid4())
                transcription = Transcription(
                    id=transcription_id, text=transcribed, spoken=transcribed
                )
                yield transcription


async def transcript_worker_google(
    context: SharedContext,
    in_queue: Queue[RecordingOutput],
):
    credentials = Credentials.from_service_account_file(
        context.config.gcp_credentials_file_path
    )
    client = SpeechAsyncClient(credentials=credentials)
    sample_size = context.audio.get_sample_size(context.config.format)
    logger.info("transcript worker [google] started")
    while True:
        recorded_frames = await in_queue.get()
        with wav(recorded_frames, context.config, sample_size=sample_size) as wav_file:
            rec_audio = RecognitionAudio(content=wav_file.read())
            rec_config = RecognitionConfig(
                encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=context.config.rate,
                language_code="ja-JP",
            )
            try:
                logger.info("transcribing...")
                r = await client.recognize(config=rec_config, audio=rec_audio)
                transcribed = "".join([result.alternatives[0].transcript for result in r.results])  # type: ignore
                logger.info("transcribed: %s", r)
                if transcribed:
                    transcription_id = str(uuid4())
                    transcription = Transcription(
                        id=transcription_id, text=transcribed, spoken=transcribed
                    )
                    yield transcription
            except HTTPError as e:
                logger.warning("transcription request error: %s", e)


async def transcript_worker_ami(
    context: SharedContext,
    in_queue: Queue[RecordingOutput],
):
    sample_size = context.audio.get_sample_size(context.config.format)
    logger.info("transcript worker [ami] started")
    while True:
        async with AsyncClient(timeout=context.config.ami_request_timeout) as client:
            recorded_frames = await in_queue.get()
            with wav(
                recorded_frames, context.config, sample_size=sample_size
            ) as wav_file:
                data = {
                    "d": f"grammarFileNames={context.config.ami_engine_name} profileId={context.config.ami_service_id} keepFillerToken=1",
                    "u": context.config.ami_appkey,
                }
                files = {"a": wav_file}
                try:
                    logger.info("transcribing...")
                    r = await client.post(
                        context.config.ami_engine_uri, data=data, files=files
                    )
                    res_body = r.json()
                    logger.info("transcribed: %s", res_body)
                    text: str = res_body["text"].replace("%", "")
                    spoken = "".join(
                        [
                            "".join([token["spoken"] for token in result["tokens"]])
                            for result in res_body["results"]
                        ]
                    )
                    if text:
                        transcription_id = str(uuid4())
                        transcription = Transcription(
                            id=transcription_id, text=text, spoken=spoken
                        )
                        yield transcription
                except HTTPError as e:
                    logger.warning("transcription request error: %s", e)
                    logger.exception(e)


async def transcription_worker(
    context: SharedContext,
    in_queue: Queue[RecordingOutput],
    out_queue: Queue[Message[Transcription]],
):
    if context.config.transcription_worker_type == TranscriptionWorkerType.ACP:
        generator = transcript_worker_ami
    elif context.config.transcription_worker_type == TranscriptionWorkerType.GCP:
        generator = transcript_worker_google
    elif context.config.transcription_worker_type == TranscriptionWorkerType.WHISPER:
        generator = transcript_worker_whisper
    else:
        raise ValueError("transcription worker type unknown.")
    try:
        async for transcription in generator(context=context, in_queue=in_queue):
            out_queue.put_nowait(
                Message(source=EventType.transcription, content=transcription)
            )
    except CancelledError:
        logger.debug("transcription worker cancelled")
        raise


def create_transcription_worker(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[RecordingOutput]()
    event = EventType.transcription
    context.input_queues[event] = in_queue
    return loop.create_task(
        transcription_worker(
            context, in_queue=in_queue, out_queue=context.broker_queue
        ),
        name=event.name,
    )
