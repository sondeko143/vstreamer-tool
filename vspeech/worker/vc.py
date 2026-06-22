import audioop
import json
import time
from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import to_thread
from audioop import mul
from collections.abc import Generator
from collections.abc import Sequence
from functools import partial
from math import floor
from math import sqrt
from typing import Any

from vspeech.config import EventType
from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig
from vspeech.config import SampleFormat
from vspeech.config import VcConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def record_vc_elapsed(seconds: float) -> None:
    telemetry.record("vc", seconds)
    logger.info("rvc elapsed time: %s", seconds)


def chunks(data: bytes, chunk_size: int) -> Generator[bytes, bytes, None]:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


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
    session = create_session(rvc_config.model_file, gpu_id=device.index)
    check_cuda_provider(session.get_providers())
    if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
        rmvpe_session = create_rmvpe_session(rvc_config.rmvpe_model_file, device.index)
    else:
        rmvpe_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    target_sample_rate = metadata["samplingRate"]
    f0_enabled = metadata["f0"]
    # Warm up: pay the torch.compile (max-autotune) and onnxruntime graph-build
    # cost at startup. The first real inference would otherwise stall for
    # seconds (observed up to ~145s) while these compile lazily.
    try:
        await to_thread(
            change_voice,
            voice_frames=b"\x00\x00" * 16000,
            voice_sample_rate=16000,
            rvc_config=rvc_config,
            half_available=half_available,
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
            if vc_config.adjust_output_vol_to_input_voice:
                max_possible_val = (2 ** (input_sample_width * 8)) / 2
                input_vols = [
                    min(
                        max(
                            sqrt(
                                audioop.rms(chunk, input_sample_width)
                                / max_possible_val
                            ),
                            vc_config.min_volume,
                        ),
                        vc_config.max_volume,
                    )
                    for chunk in chunks(
                        speech.sound.data,
                        chunk_size=floor(vc_config.volume_adjust_window),
                    )
                ]
            else:
                input_vols = []
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
                half_available=half_available,
                target_sample_rate=target_sample_rate,
                device=device,
                emb_output_layer=metadata.get("embOutputLayer", 9),
                use_final_proj=metadata.get("useFinalProj", True),
                hubert_model=hubert_model,
                session=session,
                f0_enabled=f0_enabled,
                rmvpe_session=rmvpe_session,
            )
            worker_output = WorkerOutput.from_input(speech)
            if vc_config.adjust_output_vol_to_input_voice and input_vols:
                raw_frames = audio.tobytes()
                output_data = b""
                chunk_size = floor(len(raw_frames) / len(input_vols))
                if chunk_size % 2 == 1:
                    chunk_size -= 1
                output_sample_size = get_sample_size(SampleFormat.INT16)
                for idx, chunk in enumerate(chunks(raw_frames, chunk_size=chunk_size)):
                    volume = (
                        input_vols[idx] if idx < len(input_vols) else input_vols[-1]
                    )
                    logger.debug("chunk size: %s x %s", len(chunk), volume)
                    output_data += audioop.mul(
                        chunk,
                        output_sample_size,
                        volume,
                    )
            else:
                output_data = audio.tobytes()
            vc_end_time = time.perf_counter()
            record_vc_elapsed(vc_end_time - vc_start_time)
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
