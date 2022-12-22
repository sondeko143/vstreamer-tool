from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import current_task
from asyncio import to_thread
from typing import List
from typing import get_type_hints

from humps import camelize
from pyaudio import PyAudio
from pyaudio import Stream
from pyaudio import paInt16
from pyvcroid2 import VcRoid2

from vspeech.audio import get_device_name
from vspeech.audio import search_device
from vspeech.config import Config
from vspeech.config import SpeechWorkerType
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.transcription import Transcription
from vspeech.voicevox import VoiceVox


async def playback(stream: Stream, data: bytes):
    await to_thread(stream.write, data)


def get_output_stream(audio: PyAudio, config: Config, rate: int) -> Stream:
    output_device_index = config.output_device_index
    if not output_device_index:
        output_device = search_device(
            audio,
            host_api_type=config.output_host_api_name,
            name=config.output_device_name,
            output=True,
        )
        if not output_device:
            raise TypeError("not found output device")
        output_device_index = output_device.index
    output_device_name = get_device_name(audio, output_device_index)
    logger.info("use output device %s: %s", output_device_index, output_device_name)
    output_stream = audio.open(
        format=paInt16,
        channels=1,
        rate=rate,
        output=True,
        output_device_index=output_device_index,
    )
    return output_stream


def vr2_reload(context: SharedContext, vr2: VcRoid2):
    config = context.config
    task = current_task()
    if not task:
        return
    task_name = task.get_name()
    if not context.reload.get(task_name):
        return
    if config.vr2_voice_name:
        logger.info("vr2 reload voice_name: %s", config.vr2_voice_name)
        vr2.loadVoice(config.vr2_voice_name)
    if vr2.param:
        for name in get_type_hints(config.vr2_params):
            value = getattr(config.vr2_params, name)
            logger.info("vr2 %s: %s", name, value)
            setattr(vr2.param, camelize(name), value)
    context.reload[task_name] = False


def voicevox_reload(context: SharedContext, vvox: VoiceVox):
    config = context.config
    task = current_task()
    if not task:
        return
    task_name = task.get_name()
    if not context.reload.get(task_name):
        return
    if config.voicevox_speaker_id:
        logger.info("vvox reload voice_name: %s", config.voicevox_speaker_id)
        vvox.load_model(config.voicevox_speaker_id)
    context.reload[task_name] = False


async def vroid2_worker(
    context: SharedContext,
    in_queue: Queue[Transcription],
):
    output_stream = get_output_stream(
        audio=context.audio, config=context.config, rate=44110
    )
    with VcRoid2() as vr2:
        lang_list: List[str] = vr2.listLanguages()
        if "standard" in lang_list:
            vr2.loadLanguage("standard")
        elif lang_list:
            vr2.loadLanguage(lang_list[0])
        else:
            raise Exception("No language library")

        # Load Voice

        if context.config.vr2_voice_name:
            voice_name = context.config.vr2_voice_name
        else:
            voice_list: List[str] = vr2.listVoices()
            if voice_list:
                voice_name = voice_list[0]
            else:
                raise Exception("No voice library")
        logger.info("vr2 voice: %s", voice_name)
        vr2.loadVoice(voice_name)

        if vr2.param:
            for name in get_type_hints(context.config.vr2_params):
                value = getattr(context.config.vr2_params, name)
                logger.info("vr2 %s: %s", name, value)
                setattr(vr2.param, camelize(name), value)
        try:
            while True:
                transcribed = await in_queue.get()
                vr2_reload(context, vr2=vr2)
                logger.info("voice generating...")
                try:
                    speech, _ = await to_thread(
                        vr2.textToSpeech, transcribed.text, raw=True
                    )
                    logger.info("voice generated")
                    logger.info("playback...")
                    await playback(stream=output_stream, data=speech)
                    logger.info("playback end")
                except UnicodeEncodeError as e:
                    logger.warning(e)
        except CancelledError:
            logger.info("speech worker cancelled")
            raise


async def voicevox_worker(
    context: SharedContext,
    in_queue: Queue[Transcription],
):
    output_stream = get_output_stream(
        audio=context.audio, config=context.config, rate=24000
    )
    with VoiceVox(context.config.voicevox_core_dir) as vvox:
        vvox.voicevox_load_openjtalk_dict(str(context.config.openjtalk_dir))
        vvox.load_model(context.config.voicevox_speaker_id)
        try:
            while True:
                transcribed = await in_queue.get()
                voicevox_reload(context=context, vvox=vvox)
                logger.info("voice generating...")
                try:
                    speech = await to_thread(
                        vvox.voicevox_tts,
                        transcribed.text,
                        context.config.voicevox_speaker_id,
                    )
                    logger.info("voice generated")
                    logger.info("playback...")
                    await playback(stream=output_stream, data=speech[44:])
                    logger.info("playback end")
                except UnicodeEncodeError as e:
                    logger.warning(e)
        except CancelledError:
            logger.info("speech worker cancelled")
            raise


async def transcription_worker(
    context: SharedContext,
):
    config = context.config
    if config.speech_worker_type == SpeechWorkerType.VR2:
        return vroid2_worker
    elif config.speech_worker_type == SpeechWorkerType.VOICEVOX:
        return voicevox_worker
    else:
        raise ValueError("speech worker type unknown.")


def create_speech_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[Transcription]()
    event = EventType.speech
    context.input_queues[event] = in_queue
    config = context.config
    if config.speech_worker_type == SpeechWorkerType.VR2:
        worker = vroid2_worker
    elif config.speech_worker_type == SpeechWorkerType.VOICEVOX:
        worker = voicevox_worker
    else:
        raise ValueError("speech worker type unknown.")
    return loop.create_task(
        worker(context, in_queue=in_queue),
        name=event.name,
    )
