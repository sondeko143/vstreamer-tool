import json
import time
from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import to_thread
from audioop import mul
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vspeech.config import EventType
from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig
from vspeech.config import SampleFormat
from vspeech.config import VcConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.telemetry import telemetry
from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import apply_vad_gate
from vspeech.lib.vad import create_vad_session
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_gate_mask
from vspeech.lib.vad import speech_probs
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def record_vc_elapsed(seconds: float, trace_id: str = "") -> None:
    telemetry.record("vc", seconds, trace_id=trace_id)
    logger.info("rvc elapsed time: %s", seconds)


def check_cuda_provider(providers: Sequence[str]) -> None:
    """Fail loudly if the RVC onnxruntime session fell back to CPU.

    Without onnxruntime-gpu the CUDAExecutionProvider is silently dropped and
    every inference raises a cryptic io-binding error while RVC runs unusably
    slow on CPU. Surface that at startup instead.
    """
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "RVC onnxruntime session has no CUDAExecutionProvider "
            f"(active providers: {providers}). Install the GPU runtime with "
            "`uv sync --extra rvc` (onnxruntime-gpu); running RVC on CPU is "
            "unusably slow."
        )


def _dtype_for_width(width: int) -> np.dtype[Any]:
    """Signed-integer numpy dtype for a PCM byte-width.

    Matches the integer-PCM assumption of the old audioop path; vc input is
    INT16 in practice. Note: width==1 is treated as signed int8; unsigned
    8-bit WAV (bias 128) is not handled (never exercised here).
    """
    if width == 1:
        return np.dtype(np.int8)
    if width == 2:
        return np.dtype(np.int16)
    if width == 4:
        return np.dtype(np.int32)
    raise ValueError(f"unsupported input sample width: {width}")


def _framewise_rms(samples: NDArray[np.float32], n_frames: int) -> NDArray[np.float64]:
    """RMS of each of n_frames near-equal contiguous segments of samples."""
    bounds = np.linspace(0, samples.shape[0], n_frames + 1).astype(np.int64)
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = samples[bounds[i] : bounds[i + 1]]
        if seg.shape[0]:
            rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
    return rms


def apply_input_envelope(
    output_i16: NDArray[np.int16],
    input_pcm: bytes,
    input_sample_width: int,
    input_rate: int,
    window_ms: float,
    strength: float,
    min_gain: float,
    max_gain: float,
) -> NDArray[np.int16]:
    """Modulate the RVC output by the input voice's relative loudness envelope.

    The input PCM is reduced to a per-frame RMS envelope normalized by its own
    mean, so only the *relative* shape survives (mean 1, independent of mic
    gain). That shape is linearly interpolated to sample resolution and
    multiplied onto the RVC output as a clamped gain -- overlaying how the
    speaker's volume rose and fell. It deliberately does NOT divide by the
    output's own envelope: doing so saturates the gain wherever the output is
    quiet (pumping the noise floor into "silent" sections) and inverse-weights
    by output loudness (an audible compressor when max_gain > 1).

    The RVC output is already a full-level int16 signal, so the gain acts as a
    downward *duck*: with max_gain <= 1 (the default) it can only attenuate,
    which is clip-free (|out * gain| <= |out|). A max_gain > 1 boosts the loud
    parts past int16 range and hard-clips them (audible distortion) -- only use
    it if the RVC output has headroom. Returns the RVC output unchanged when
    disabled (strength <= 0), when either side is empty, or when the input is
    effectively silent.
    """
    out_len = int(output_i16.shape[0])
    if out_len == 0 or not input_pcm or strength <= 0.0:
        return output_i16

    in_i = np.frombuffer(input_pcm, dtype=_dtype_for_width(input_sample_width))
    if in_i.shape[0] == 0:
        return output_i16
    # Absolute scale is irrelevant -- it cancels in the mean-normalization below.
    in_f = in_i.astype(np.float32)

    frame_len = max(1, round(window_ms * input_rate / 1000.0))
    n_frames = max(1, in_f.shape[0] // frame_len)

    rms_in = _framewise_rms(in_f, n_frames)
    mean_in = float(rms_in.mean())
    if mean_in < 1e-8:
        return output_i16

    # Input's relative loudness envelope (mean 1), stretched to the output
    # sample grid by linear interpolation (smooth gain, no per-frame steps).
    src_x = (np.arange(n_frames) + 0.5) / n_frames
    dst_x = (np.arange(out_len) + 0.5) / out_len
    shape_in = np.interp(dst_x, src_x, rms_in / mean_in)

    gain = np.clip(np.power(shape_in, strength), min_gain, max_gain)
    out_f = output_i16.astype(np.float32)
    return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)


def _input_as_float32_16k(
    data: bytes, sample_width: int, rate: int
) -> NDArray[np.float32]:
    """Decode integer PCM to float32 in [-1, 1] at the VAD rate.

    torch/torchaudio are imported only on the resample path so this stays
    usable (and testable) in environments without the rvc extra when the
    input is already 16kHz.
    """
    scale = float(2 ** (8 * sample_width - 1))
    audio = (
        np.frombuffer(data, dtype=_dtype_for_width(sample_width)).astype(np.float32)
        / scale
    )
    if rate == VAD_SAMPLE_RATE:
        return audio
    import torch

    from vspeech.lib.rvc import get_resampler

    resampler = get_resampler(rate, VAD_SAMPLE_RATE, torch.device("cpu"))
    return resampler(torch.from_numpy(audio)).numpy()


async def rvc_worker(
    rvc_config: RvcConfig,
    vc_config: VcConfig,
    in_queue: Queue[WorkerInput],
):
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.pitch_extract import create_rmvpe_session
    from vspeech.lib.rvc import change_voice
    from vspeech.lib.rvc import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    device, device_name = get_device(rvc_config.gpu_id, rvc_config.gpu_name)
    logger.info("vc worker device: %s, %s", device, device_name)
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file,
        device=device,
        is_half=half_available,
    )
    session = create_session(rvc_config.model_file, device)
    check_cuda_provider(session.get_providers())
    if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
        rmvpe_session = create_rmvpe_session(rvc_config.rmvpe_model_file, device.index)
    else:
        rmvpe_session = None
    if vc_config.vad_gate:
        vad_session = create_vad_session(vc_config.vad_model_file)
        logger.info("vad gate enabled: %s", vc_config.vad_model_file)
    else:
        vad_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    target_sample_rate = metadata["samplingRate"]
    f0_enabled = metadata["f0"]
    # Warm up: pay the onnxruntime graph-build / CUDA kernel autotune cost at
    # startup. The first real inference would otherwise stall for seconds
    # (observed up to ~145s) while these build lazily.
    try:
        await to_thread(
            change_voice,
            voice_frames=b"\x00\x00" * 16000,
            voice_sample_rate=16000,
            rvc_config=rvc_config,
            target_sample_rate=target_sample_rate,
            device=device,
            emb_output_layer=metadata.get("embOutputLayer", 9),
            use_final_proj=metadata.get("useFinalProj", True),
            hubert_model=hubert_model,
            session=session,
            f0_enabled=f0_enabled,
            rmvpe_session=rmvpe_session,
        )
        logger.info("vc worker warmed up")
    except Exception as e:
        logger.warning("vc warmup failed: %s", e)
    logger.info("vc worker started")
    while True:
        speech = await in_queue.get()
        try:
            logger.debug("voice changing...")
            input_sample_width = get_sample_size(speech.sound.format)
            vad_gains: NDArray[np.float64] | None = None
            if vad_session is not None:
                try:
                    audio_16k = _input_as_float32_16k(
                        speech.sound.data, input_sample_width, speech.sound.rate
                    )
                    probs = await to_thread(speech_probs, vad_session, audio_16k)
                    skip, speech_ratio = should_skip_vc(
                        probs,
                        vc_config.vad_threshold,
                        vc_config.vad_min_speech_ratio,
                    )
                    if skip:
                        telemetry.record(
                            "vc_skip", speech_ratio, trace_id=speech.trace_id
                        )
                        logger.info(
                            "vc skipped: speech ratio %.3f < %.3f",
                            speech_ratio,
                            vc_config.vad_min_speech_ratio,
                        )
                        continue
                    vad_gains = speech_gate_mask(
                        probs,
                        vc_config.vad_threshold,
                        vc_config.vad_speech_pad_ms,
                        vc_config.vad_min_gain,
                    )
                except Exception as e:
                    logger.warning("vad gate failed; passing chunk ungated: %s", e)
                    vad_gains = None
            vc_start_time = time.perf_counter()
            audio = await to_thread(
                change_voice,
                voice_frames=mul(
                    speech.sound.data,
                    input_sample_width,
                    rvc_config.input_boost,
                ),
                voice_sample_rate=speech.sound.rate,
                rvc_config=rvc_config,
                target_sample_rate=target_sample_rate,
                device=device,
                emb_output_layer=metadata.get("embOutputLayer", 9),
                use_final_proj=metadata.get("useFinalProj", True),
                hubert_model=hubert_model,
                session=session,
                f0_enabled=f0_enabled,
                rmvpe_session=rmvpe_session,
            )
            if vc_config.adjust_output_vol_to_input_voice:
                audio = apply_input_envelope(
                    audio,
                    speech.sound.data,
                    input_sample_width,
                    input_rate=speech.sound.rate,
                    window_ms=vc_config.volume_adjust_window_ms,
                    strength=vc_config.envelope_strength,
                    min_gain=vc_config.min_gain,
                    max_gain=vc_config.max_gain,
                )
            if vad_gains is not None:
                audio = apply_vad_gate(audio, vad_gains)
            output_data = audio.tobytes()
            vc_end_time = time.perf_counter()
            record_vc_elapsed(vc_end_time - vc_start_time, trace_id=speech.trace_id)
            worker_output = WorkerOutput.from_input(speech)
            worker_output.sound = SoundOutput(
                data=output_data,
                rate=target_sample_rate,
                channels=1,
                format=SampleFormat.INT16,
            )
            worker_output.text = speech.text
            yield worker_output
        except Exception as e:
            logger.exception(e)


async def vc_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    try:
        while True:
            context.reset_need_reload()
            worker = partial(rvc_worker, rvc_config=context.config.rvc)
            async for output in worker(vc_config=context.config.vc, in_queue=in_queue):
                out_queue.put_nowait(output)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError as e:
        raise shutdown_worker(e)


def create_vc_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(event=EventType.vc, configs_depends_on=["vc", "rvc"])
    task = tg.create_task(
        vc_worker(context, in_queue=worker.in_queue, out_queue=context.sender_queue),
        name=worker.event.name,
    )
    return task
