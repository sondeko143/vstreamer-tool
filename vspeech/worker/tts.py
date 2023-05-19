from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from functools import partial
from typing import List

from vspeech.config import SampleFormat
from vspeech.config import TtsWorkerType
from vspeech.config import VoicevoxConfig
from vspeech.config import Vr2Config
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def vroid2_worker(vr2_config: Vr2Config, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voiceroid import VR2

    vr2 = VR2()
    with vr2.vc_roid2:
        lang_list: List[str] = vr2.list_languages()
        if "standard" in lang_list:
            vr2.load_language("standard")
        elif lang_list:
            vr2.load_language(lang_list[0])
        else:
            raise Exception("No language library")
        if vr2_config.voice_name:
            voice_name = vr2_config.voice_name
        else:
            voice_list: List[str] = vr2.list_voices()
            if voice_list:
                voice_name = voice_list[0]
            else:
                raise Exception("No voice library")
        logger.info("vr2 voice: %s", voice_name)
        logger.info("tts worker [vr2] started")
        vr2.load_voice(voice_name, vr2_config.params)
        while True:
            transcribed = await in_queue.get()
            logger.debug("voice generating...")
            try:
                speech, _ = await to_thread(
                    vr2.text_to_speech, transcribed.text, raw=True
                )
                logger.debug("voice generated")
                worker_output = WorkerOutput.from_input(transcribed)
                worker_output.sound = SoundOutput(
                    data=speech, rate=44110, format=SampleFormat.INT16, channels=1
                )
                yield worker_output
            except Exception as e:
                logger.warning("%s", e)


async def voicevox_worker(vvox_config: VoicevoxConfig, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voicevox import Voicevox

    vvox = Voicevox(vvox_config.openjtalk_dir)
    vvox.load_model(vvox_config.speaker_id)
    logger.info("tts worker [voicevox] started")
    while True:
        transcribed = await in_queue.get()
        logger.debug("voice generating...")
        try:
            speech = await to_thread(
                vvox.voicevox_tts,
                text=transcribed.text,
                speaker_id=vvox_config.speaker_id,
                params=vvox_config.params,
            )
            logger.debug("voice generated")
            worker_output = WorkerOutput.from_input(transcribed)
            worker_output.sound = SoundOutput(
                data=speech[44:], rate=24000, format=SampleFormat.INT16, channels=1
            )
            yield worker_output
        except UnicodeEncodeError as e:
            logger.warning("%s", e)


async def tts_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    try:
        while True:
            context.reset_need_reload()
            config = context.config.tts
            if config.worker_type == TtsWorkerType.VR2:
                worker = partial(vroid2_worker, vr2_config=context.config.vr2)
            elif config.worker_type == TtsWorkerType.VOICEVOX:
                worker = partial(voicevox_worker, vvox_config=context.config.voicevox)
            else:
                raise ValueError("tts worker type unknown.")
            async for output in worker(in_queue=in_queue):
                out_queue.put_nowait(output)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
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
