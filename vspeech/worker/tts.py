from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from typing import List

from vspeech.config import SampleFormat
from vspeech.config import TtsWorkerType
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def vroid2_worker(context: SharedContext, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voiceroid import VR2
    from vspeech.lib.voiceroid import vr2_reload

    vr2 = VR2()
    with vr2.vc_roid2:
        lang_list: List[str] = vr2.list_languages()
        if "standard" in lang_list:
            vr2.load_language("standard")
        elif lang_list:
            vr2.load_language(lang_list[0])
        else:
            raise Exception("No language library")
        if context.config.vr2.voice_name:
            voice_name = context.config.vr2.voice_name
        else:
            voice_list: List[str] = vr2.list_voices()
            if voice_list:
                voice_name = voice_list[0]
            else:
                raise Exception("No voice library")
        logger.info("vr2 voice: %s", voice_name)
        logger.info("tts worker [vr2] started")
        vr2.load_voice(voice_name, context.config.vr2.params)
        while True:
            transcribed = await in_queue.get()
            vr2_reload(context, vr2=vr2)
            logger.info("voice generating...")
            try:
                speech, _ = await to_thread(
                    vr2.text_to_speech, transcribed.text, raw=True
                )
                logger.info("voice generated")
                worker_output = WorkerOutput.from_input(transcribed)
                worker_output.sound = SoundOutput(
                    data=speech, rate=44110, format=SampleFormat.INT16, channels=1
                )
                yield worker_output
            except Exception as e:
                logger.warning(e)


async def voicevox_worker(context: SharedContext, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voicevox import Voicevox
    from vspeech.lib.voicevox import voicevox_reload

    vvox = Voicevox(context.config.voicevox.openjtalk_dir)
    vvox.load_model(context.config.voicevox.speaker_id)
    logger.info("tts worker [voicevox] started")
    while True:
        transcribed = await in_queue.get()
        voicevox_reload(context=context, vvox=vvox)
        logger.info("voice generating...")
        try:
            speech = await to_thread(
                vvox.voicevox_tts,
                text=transcribed.text,
                speaker_id=context.config.voicevox.speaker_id,
                params=context.config.voicevox.params,
            )
            logger.info("voice generated")
            worker_output = WorkerOutput.from_input(transcribed)
            worker_output.sound = SoundOutput(
                data=speech[44:], rate=24000, format=SampleFormat.INT16, channels=1
            )
            yield worker_output
        except UnicodeEncodeError as e:
            logger.warning(e)


async def tts_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    while True:
        context.reset_need_reload()
        config = context.config.tts
        if config.worker_type == TtsWorkerType.VR2:
            worker = vroid2_worker
        elif config.worker_type == TtsWorkerType.VOICEVOX:
            worker = voicevox_worker
        else:
            raise ValueError("tts worker type unknown.")
        while True:
            try:
                async for output in worker(context=context, in_queue=in_queue):
                    out_queue.put_nowait(output)
                    if context.need_reload:
                        break
            except CancelledError:
                logger.info("tts worker cancelled")
                raise


def create_tts_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.tts
    context.input_queues[event] = in_queue
    task = loop.create_task(
        tts_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
