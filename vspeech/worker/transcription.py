from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import sleep
from asyncio import to_thread
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import partial
from io import BytesIO
from logging import DEBUG
from pathlib import Path
from typing import TYPE_CHECKING
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

from vspeech.config import AmiConfig
from vspeech.config import GcpConfig
from vspeech.config import SampleFormat
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.config import WhisperConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.ami import parse_response
from vspeech.lib.gcp import get_credentials
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

if TYPE_CHECKING:
    import numpy as np


def pcm_to_waveform(sound: SoundInput) -> "np.ndarray":
    """Decode INT16 PCM bytes into a mono float32 waveform in [-1, 1].

    faster-whisper accepts a float32 ndarray directly, so this lets the
    transcription worker hand audio to the model without a temporary WAV
    file on disk.
    """
    import numpy as np

    samples = np.frombuffer(sound.data, dtype=np.int16).astype(np.float32) / 32768.0
    if sound.channels > 1:
        samples = samples.reshape(-1, sound.channels).mean(axis=1)
    return samples.astype(np.float32)


def join_transcribed_segments(segments: list, whisper_config: WhisperConfig) -> str:
    """Join the text of segments that clear whisper's confidence thresholds."""
    return "".join(
        segment.text
        for segment in segments
        if segment.no_speech_prob < whisper_config.no_speech_prob_threshold
        and segment.avg_logprob > whisper_config.logprob_threshold
        and segment.temperature is not None
        and segment.temperature < 1.0
    )


def _run_whisper(model, waveform, whisper_config: WhisperConfig) -> list:
    """Transcribe and fully materialize segments. Runs in a worker thread so the
    heavy decode (which happens while iterating the generator) never blocks the
    event loop."""
    segments, _ = model.transcribe(
        waveform,
        language="ja",
        no_speech_threshold=whisper_config.no_speech_prob_threshold,
        log_prob_threshold=whisper_config.logprob_threshold,
    )
    return list(segments)


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
    config: TranscriptionConfig,
    whisper_config: WhisperConfig,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    from faster_whisper import WhisperModel

    from vspeech.lib.cuda_util import get_device

    device, device_name = get_device(whisper_config.gpu_id, whisper_config.gpu_name)
    logger.info("transcript worker device: %s, %s", device, device_name)

    model = WhisperModel(
        whisper_config.model,
        device="cuda",
        compute_type="float16",
        device_index=device.index,
    )
    logger.info("transcript worker [whisper] started")
    # Warm up: pay the one-off cold-start cost (model graph build / kernel
    # compilation that happens on the first decode) at startup instead of
    # penalizing the first real utterance.
    try:
        silence = SoundInput(
            data=b"\x00\x00" * 16000,
            rate=16000,
            format=SampleFormat.INT16,
            channels=1,
        )
        await to_thread(_run_whisper, model, pcm_to_waveform(silence), whisper_config)
        logger.info("transcript worker [whisper] warmed up")
    except Exception as e:
        logger.warning("whisper warmup failed: %s", e)
    while True:
        recorded = await in_queue.get()
        try:
            logger.debug("transcribing...")
            waveform = pcm_to_waveform(recorded.sound)
            with telemetry.timer("transcription", trace_id=recorded.trace_id):
                segments = await to_thread(
                    _run_whisper, model, waveform, whisper_config
                )
                transcribed = join_transcribed_segments(segments, whisper_config)
            if logger.isEnabledFor(DEBUG):
                for segment in segments:
                    logger.debug(
                        "segment: %s, log: %s, no_speech: %s",
                        segment.text,
                        segment.avg_logprob,
                        segment.no_speech_prob,
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
                sample_size = get_sample_size(recorded.sound.format)
                with wav(recorded.sound, sample_size=sample_size) as wav_file:
                    await log_transcribed(
                        config.recording_log_dir,
                        wav_file=wav_file,
                        text=transcribed,
                    )
        except WavError as e:
            logger.warning("%s", e)
        except ValueError as e:
            logger.warning("%s", e)


async def transcript_worker_google(
    config: TranscriptionConfig,
    gcp_config: GcpConfig,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    credentials, _ = get_credentials(gcp_config)
    client = SpeechAsyncClient(credentials=credentials)
    logger.info("transcript worker [google] started")
    while True:
        recorded = await in_queue.get()
        try:
            sample_size = get_sample_size(recorded.sound.format)
            with wav(recorded.sound, sample_size=sample_size) as wav_file:
                rec_audio = RecognitionAudio(content=wav_file.read())
                rec_config = RecognitionConfig(
                    encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=recorded.sound.rate,
                    language_code="ja-JP",
                )
                logger.debug("transcribing...")
                request = RecognizeRequest(config=rec_config, audio=rec_audio)
                with telemetry.timer("transcription", trace_id=recorded.trace_id):
                    r = await transcribe_request_google(
                        client=client,
                        request=request,
                        timeout=gcp_config.request_timeout,
                        max_retry_count=gcp_config.max_retry_count,
                        retry_delay_sec=gcp_config.retry_delay_sec,
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
            logger.warning("%s", e)
        except ValueError as e:
            logger.warning("%s", e)


async def transcript_worker_ami(
    config: TranscriptionConfig,
    ami_config: AmiConfig,
    in_queue: Queue[WorkerInput],
) -> AsyncGenerator[WorkerOutput, None]:
    logger.info("transcript worker [ami] started")
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
                    logger.debug("transcribing...")
                    with telemetry.timer("transcription", trace_id=recorded.trace_id):
                        r = await client.post(
                            ami_config.engine_uri, data=data, files=files
                        )
                    res_json = r.json()
                    logger.info("transcribed: %s", res_json)
                    text, spoken = parse_response(
                        res_json,
                        use_mozc=config.transliterate_with_mozc,
                    )
                    if text:
                        logger.debug("transliterate: %s -> %s", spoken, text)
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
                logger.warning("%s", e)
            except ValueError as e:
                logger.warning("%s", e)


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
                generator = partial(
                    transcript_worker_ami, ami_config=context.config.ami
                )
            elif config.worker_type == TranscriptionWorkerType.GCP:
                generator = partial(
                    transcript_worker_google, gcp_config=context.config.gcp
                )
            elif config.worker_type == TranscriptionWorkerType.WHISPER:
                generator = partial(
                    transcript_worker_whisper, whisper_config=context.config.whisper
                )
            else:
                raise ValueError("transcription worker type unknown.")
            async for transcribed in generator(
                config=context.config.transcription, in_queue=in_queue
            ):
                out_queue.put_nowait(transcribed)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError as e:
        raise shutdown_worker(e)


def create_transcription_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(
        event=EventType.transcription,
        configs_depends_on=["transcription", "ami", "gcp", "whisper"],
    )
    task = tg.create_task(
        transcription_worker(
            context, in_queue=worker.in_queue, out_queue=context.sender_queue
        ),
        name=worker.event.name,
    )
    return task
