import json
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from functools import partial
from typing import Any

from vspeech.config import EventType
from vspeech.config import RvcConfig
from vspeech.config import SampleFormat
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def rvc_worker(
    rvc_config: RvcConfig,
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
    metadata = json.loads(modelmeta.custom_metadata_map["metadata"])
    target_sample_rate = metadata["samplingRate"]
    f0_enabled = metadata["f0"]
    logger.info("vc worker started")
    while True:
        speech = await in_queue.get()
        try:
            logger.debug("voice changing...")
            audio = await to_thread(
                change_voice,
                voice_frames=speech.sound.data,
                voice_sample_rate=speech.sound.rate,
                rvc_config=rvc_config,
                half_available=half_available,
                target_sample_rate=target_sample_rate,
                device=device,
                emb_channels=metadata["embChannels"],
                hubert_model=hubert_model,
                session=session,
                f0_enabled=f0_enabled,
            )
            logger.debug("voice changed")
            worker_output = WorkerOutput.from_input(speech)
            worker_output.sound = SoundOutput(
                data=audio.tobytes(),
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
            async for output in worker(in_queue=in_queue):
                out_queue.put_nowait(output)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError:
        logger.debug("vc worker cancelled")
        raise


def create_vc_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.vc
    context.input_queues[event] = in_queue
    task = loop.create_task(
        vc_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
