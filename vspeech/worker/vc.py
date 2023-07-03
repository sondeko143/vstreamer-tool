import audioop
import json
from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import to_thread
from audioop import mul
from functools import partial
from math import sqrt
from typing import Any

from vspeech.config import EventType
from vspeech.config import RvcConfig
from vspeech.config import SampleFormat
from vspeech.config import VcConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def rvc_worker(
    rvc_config: RvcConfig,
    vc_config: VcConfig,
    in_queue: Queue[WorkerInput],
):
    from vspeech.lib.rvc import change_voice
    from vspeech.lib.rvc import create_session
    from vspeech.lib.rvc import get_device
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    device = get_device()
    half_available = half_precision_available(rvc_config.gpu_id)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file,
        device=device,
        is_half=half_available,
    )
    session = create_session(rvc_config.model_file)
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    target_sample_rate = metadata["samplingRate"]
    f0_enabled = metadata["f0"]
    logger.info("vc worker started")
    while True:
        speech = await in_queue.get()
        try:
            logger.debug("voice changing...")
            input_sample_width = get_sample_size(speech.sound.format)
            if vc_config.adjust_output_vol_to_input_voice:
                max_possible_val = (2 ** (input_sample_width * 8)) / 2
                input_vol = sqrt(
                    audioop.rms(speech.sound.data, input_sample_width)
                    / max_possible_val
                )
                logger.debug("input_vol: %s", input_vol)
            else:
                input_vol = 1.0
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
            )
            logger.debug("voice changed")
            worker_output = WorkerOutput.from_input(speech)
            worker_output.sound = SoundOutput(
                data=audioop.mul(
                    audio.tobytes(),
                    get_sample_size(SampleFormat.INT16),
                    input_vol,
                ),
                rate=target_sample_rate,
                channels=1,
                format=SampleFormat.INT16,
            )
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
