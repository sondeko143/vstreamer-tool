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
from typing import assert_never
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
from vspeech.exceptions import worker_startup
from vspeech.lib.ami import parse_response
from vspeech.lib.gcp import get_credentials
from vspeech.lib.telemetry import telemetry
from vspeech.lib.vad import create_vad_session
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_probs
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession


WHISPER_SAMPLE_RATE = 16000


def _pcm_to_float32_mono(sound: SoundInput) -> np.ndarray:
    """Decode PCM bytes into a mono float32 signal in [-1, 1] at sound.rate.

    Dispatch is keyed on ``sound.format`` (NOT byte width): UINT8 and INT8
    share a width but differ in sign/bias, so a width-keyed table would decode
    unsigned-8 as signed and skip its 128 offset (silence -> full-scale DC).
    """
    import numpy as np

    fmt = sound.format
    if fmt == SampleFormat.FLOAT32:
        samples = np.frombuffer(sound.data, dtype=np.float32).astype(np.float32)
    elif fmt == SampleFormat.UINT8:
        # unsigned 8-bit PCM is biased by 128 (128 == silence).
        samples = (
            np.frombuffer(sound.data, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif fmt == SampleFormat.INT8:
        samples = np.frombuffer(sound.data, dtype=np.int8).astype(np.float32) / 128.0
    elif fmt == SampleFormat.INT16:
        samples = np.frombuffer(sound.data, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt == SampleFormat.INT24:
        # 3-byte little-endian signed PCM -> sign-extended int32 -> [-1, 1).
        b = np.frombuffer(sound.data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        as32 = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        as32 = (as32 ^ 0x800000) - 0x800000
        samples = as32.astype(np.float32) / float(1 << 23)
    else:
        raise ValueError(f"unsupported PCM format for transcription: {fmt!r}")
    if sound.channels > 1:
        samples = samples.reshape(-1, sound.channels).mean(axis=1)
    return samples.astype(np.float32)


def _resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample a mono float32 signal to 16 kHz with PyAV (libswresample).

    faster-whisper resamples file/bytes input but never a raw ndarray, so a
    non-16 kHz recording would otherwise reach the model at the wrong speed
    and be transcribed as garbage. PyAV ships with faster-whisper, so this
    needs no extra dependency (torchaudio/scipy are not in the `whisper`
    extra, and the extra's `torch` is win32-only).
    """
    import av
    import numpy as np

    # An empty buffer would make PyAV push a 0-sample frame into swresample,
    # which raises av.error.MemoryError (not a ValueError) and would escape the
    # worker's catch and kill the TaskGroup. Return empty before touching PyAV.
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)

    resampler = av.AudioResampler(format="flt", layout="mono", rate=WHISPER_SAMPLE_RATE)
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(samples.reshape(1, -1)), format="flt", layout="mono"
    )
    frame.sample_rate = src_rate
    out_frames = resampler.resample(frame) + resampler.resample(None)
    if not out_frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([f.to_ndarray().reshape(-1) for f in out_frames]).astype(
        np.float32
    )


def pcm_to_waveform(sound: SoundInput) -> np.ndarray:
    """Decode PCM into a mono float32 waveform at 16 kHz for faster-whisper.

    faster-whisper consumes a float32 ndarray as-is *at 16 kHz* (it only
    resamples file/bytes input, never a raw array). We decode with the dtype
    that matches sound.format, downmix to mono, and resample sound.rate ->
    16 kHz when they differ -- without the resample a non-16 kHz recording
    reaches the model at the wrong speed and is transcribed as garbage.
    """
    samples = _pcm_to_float32_mono(sound)
    if sound.rate != WHISPER_SAMPLE_RATE:
        samples = _resample_to_16k(samples, sound.rate)
    return samples


def create_transcription_vad_session(
    config: TranscriptionConfig,
) -> InferenceSession | None:
    """Build the Silero VAD session for the transcription skip gate, or None.

    Returns None when the gate is disabled so the hot loop can cheaply skip it.
    When enabled, a missing or malformed model raises inside create_vad_session
    (fail loudly at startup, ADR-0019/0037) rather than silently passing every
    chunk through. CPU-fixed session (ADR-0024 の意図的例外).
    """
    if not config.vad_gate:
        return None
    session = create_vad_session(config.vad_model_file)
    logger.info("transcription vad gate enabled: %s", config.vad_model_file)
    return session


# Fail-open gate errors are logged, but a *persistent* failure (e.g. non-16kHz
# input on an ACP/GCP install without PyAV, which lives in the whisper extra)
# would warn on every chunk. Warn once per distinct error type; drop the rest to
# debug. Mutated only on the event loop (the except runs there, not in-thread).
_vad_gate_warned: set[str] = set()


def _vad_probs(vad_session: InferenceSession, sound: SoundInput) -> NDArray[np.float64]:
    """Decode to 16kHz mono then run Silero VAD.

    Both steps run inside the caller's ``to_thread`` so the bounded PyAV
    decode/resample (``pcm_to_waveform``) never blocks the event loop -- for the
    ACP/GCP backends this is the only audio decode on the request path.
    """
    return speech_probs(vad_session, pcm_to_waveform(sound))


async def vad_should_skip(
    vad_session: InferenceSession | None,
    sound: SoundInput,
    config: TranscriptionConfig,
    trace_id: str,
) -> bool:
    """Return True if this chunk has too little speech to transcribe.

    Reuses the vc gate's pure decision (should_skip_vc) on 16kHz probs from the
    shared Silero model. Skip-only: transcription emits no audio, so there is no
    output duck (unlike the vc path). Error handling is asymmetric (ADR-0019):
    an inference failure passes the chunk through ungated (returns False) rather
    than dropping speech. Returns False immediately when the gate is disabled.
    """
    if vad_session is None:
        return False
    # Deferred edge case: the gate mono-downmixes (pcm_to_waveform) before VAD,
    # but GCP/AMI upload the raw multi-channel WAV. Anti-phase stereo (L approx
    # -R, some virtual-audio setups) downmixes to approx 0, so the gate can skip
    # a chunk the cloud would have transcribed. Narrow: needs channels>1
    # (default 1) AND anti-phase; whisper decodes via pcm_to_waveform too.
    try:
        probs = await to_thread(_vad_probs, vad_session, sound)
        skip, ratio = should_skip_vc(
            probs, config.vad_threshold, config.vad_min_speech_ratio
        )
        if skip:
            telemetry.record("transc_skip", ratio, trace_id=trace_id)
            logger.info(
                "transcription skipped: speech ratio %.3f < %.3f",
                ratio,
                config.vad_min_speech_ratio,
            )
            return True
        return False
    except Exception as e:  # noqa: BLE001 - gate failure must not drop speech
        key = type(e).__name__
        if key in _vad_gate_warned:
            logger.debug("transcription vad gate failed; passing ungated: %s", e)
        else:
            _vad_gate_warned.add(key)
            logger.warning(
                "transcription vad gate failed; passing chunk ungated"
                " (further %s warnings suppressed): %s",
                key,
                e,
            )
        return False


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


_rec_log_warned: set[str] = set()


async def log_transcribed(log_dir_parent: Path, wav_file: BytesIO, text: str):
    try:
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
    except OSError as e:
        # 録音ログは補助機能: 保存先が書込不可でもパイプラインは止めない (DEGRADE)。
        key = str(log_dir_parent)
        if key in _rec_log_warned:
            logger.debug("recording_log 保存失敗 (継続): %s", e)
        else:
            _rec_log_warned.add(key)
            logger.warning(
                "recording_log を保存できません (%s) — ログ保存のみ無効化して継続: %s",
                log_dir_parent,
                e,
            )


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
) -> AsyncGenerator[WorkerOutput]:
    from faster_whisper import WhisperModel

    from vspeech.lib.cuda_util import get_device

    with worker_startup("transcription"):
        device, device_name = get_device(whisper_config.gpu_id, whisper_config.gpu_name)
        logger.info("transcript worker device: %s, %s", device, device_name)

        model = WhisperModel(
            whisper_config.model,
            device="cuda",
            compute_type="float16",
            device_index=device.index,
        )
        logger.info("transcript worker [whisper] started")
        # Created before warmup (like GCP/AMI, right after "started") so a missing
        # VAD model fails loud before whisper's cold start, not after it.
        vad_session = create_transcription_vad_session(config)
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
        if await vad_should_skip(
            vad_session, recorded.sound, config, recorded.trace_id
        ):
            continue
        try:
            logger.debug("transcribing...")
            # Deferred: vad_should_skip already decoded this chunk to 16kHz and
            # discarded it; gate-passing chunks re-decode here. Only whisper
            # pays it (ACP/GCP never call pcm_to_waveform), only on gate-pass,
            # ~a few ms against downstream GPU inference. Returning the waveform
            # from the gate ((skip, waveform)) isn't worth the signature churn.
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
) -> AsyncGenerator[WorkerOutput]:
    with worker_startup("transcription"):
        credentials, _ = get_credentials(gcp_config)
        client = SpeechAsyncClient(credentials=credentials)
        logger.info("transcript worker [google] started")
        vad_session = create_transcription_vad_session(config)
    while True:
        recorded = await in_queue.get()
        if await vad_should_skip(
            vad_session, recorded.sound, config, recorded.trace_id
        ):
            continue
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
) -> AsyncGenerator[WorkerOutput]:
    logger.info("transcript worker [ami] started")
    with worker_startup("transcription"):
        vad_session = create_transcription_vad_session(config)
    while True:
        # Deferred: the skip check runs inside this `async with`, so skipped
        # chunks still build/tear down a client. AsyncClient opens no connection
        # until the first request, so the cost is object creation only; hoist
        # the client above the gate only if skip rates get high.
        async with AsyncClient(timeout=ami_config.request_timeout) as client:
            recorded = await in_queue.get()
            if await vad_should_skip(
                vad_session, recorded.sound, config, recorded.trace_id
            ):
                continue
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
                assert_never(config.worker_type)
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
