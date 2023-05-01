from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import current_task
from asyncio import to_thread
from typing import List

from vspeech.config import SampleFormat
from vspeech.config import SpeechWorkerType
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

try:
    from vspeech.voicevox import Voicevox
except ImportError as e:
    logger.warning(e)

try:
    from vspeech.voiceroid import VR2
except ModuleNotFoundError:
    logger.info("pyvcroid2 not found")


def vr2_reload(context: SharedContext, vr2: "VR2"):
    config = context.config.vr2
    task = current_task()
    if not task:
        return
    task_name = task.get_name()
    if not context.reload.get(task_name):
        return
    if config.vr2_voice_name:
        logger.info("vr2 reload voice_name: %s", config.vr2_voice_name)
        vr2.load_voice(config.vr2_voice_name, config.vr2_params)
    context.reload[task_name] = False


def voicevox_reload(context: SharedContext, vvox: "Voicevox"):
    config = context.config.voicevox
    task = current_task()
    if not task:
        return
    task_name = task.get_name()
    if not context.reload.get(task_name):
        return
    if config.voicevox_speaker_id and not vvox.is_model_loaded(
        config.voicevox_speaker_id
    ):
        logger.info("vvox reload voice_name: %s", config.voicevox_speaker_id)
        vvox.load_model(config.voicevox_speaker_id)
    context.reload[task_name] = False


async def vroid2_worker(context: SharedContext, in_queue: Queue[WorkerInput]):
    vr2 = VR2()
    with vr2.vc_roid2:
        lang_list: List[str] = vr2.list_languages()
        if "standard" in lang_list:
            vr2.load_language("standard")
        elif lang_list:
            vr2.load_language(lang_list[0])
        else:
            raise Exception("No language library")
        if context.config.vr2.vr2_voice_name:
            voice_name = context.config.vr2.vr2_voice_name
        else:
            voice_list: List[str] = vr2.list_voices()
            if voice_list:
                voice_name = voice_list[0]
            else:
                raise Exception("No voice library")
        logger.info("vr2 voice: %s", voice_name)
        logger.info("speech worker [vr2] started")
        vr2.load_voice(voice_name, context.config.vr2.vr2_params)
        while True:
            transcribed = await in_queue.get()
            vr2_reload(context, vr2=vr2)
            logger.info("voice generating...")
            try:
                speech, _ = await to_thread(
                    vr2.text_to_speech, transcribed.text, raw=True
                )
                logger.info("voice generated")
                yield WorkerOutput(
                    source=EventType.speech,
                    sound=SoundOutput(
                        data=speech, rate=44110, format=SampleFormat.INT16, channels=1
                    ),
                    text=None,
                )
            except Exception as e:
                logger.warning(e)


async def voicevox_worker(context: SharedContext, in_queue: Queue[WorkerInput]):
    vvox = Voicevox(context.config.voicevox.openjtalk_dir)
    vvox.load_model(context.config.voicevox.voicevox_speaker_id)
    logger.info("speech worker [voicevox] started")
    while True:
        transcribed = await in_queue.get()
        voicevox_reload(context=context, vvox=vvox)
        logger.info("voice generating...")
        try:
            speech = await to_thread(
                vvox.voicevox_tts,
                text=transcribed.text,
                speaker_id=context.config.voicevox.voicevox_speaker_id,
                params=context.config.voicevox.voicevox_params,
            )
            logger.info("voice generated")
            yield WorkerOutput(
                source=EventType.speech,
                sound=SoundOutput(
                    data=speech[44:], rate=24000, format=SampleFormat.INT16, channels=1
                ),
                text=None,
            )
        except UnicodeEncodeError as e:
            logger.warning(e)


async def speech_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    config = context.config.speech
    if config.speech_worker_type == SpeechWorkerType.VR2:
        worker = vroid2_worker
    elif config.speech_worker_type == SpeechWorkerType.VOICEVOX:
        worker = voicevox_worker
    else:
        raise ValueError("speech worker type unknown.")
    while True:
        try:
            async for output in worker(context=context, in_queue=in_queue):
                out_queue.put_nowait(output)
        except CancelledError:
            logger.debug("speech worker cancelled")
            raise


def create_speech_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.speech
    context.input_queues[event] = in_queue
    return loop.create_task(
        speech_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
