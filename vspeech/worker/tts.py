from asyncio import CancelledError
from asyncio import Queue
from asyncio import TaskGroup
from asyncio import to_thread
from functools import partial
from typing import Any
from typing import List
from typing import cast

from emoji import demojize

from vspeech.config import SampleFormat
from vspeech.config import TtsWorkerType
from vspeech.config import VoicevoxConfig
from vspeech.config import VoicevoxParam
from vspeech.config import Vr2Config
from vspeech.exceptions import shutdown_worker
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


async def vroid2_worker(vr2_: Any, vr2_config: Vr2Config, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voiceroid import VR2

    vr2: VR2 = cast(VR2, vr2_)
    logger.info("tts worker [vr2] started")
    while True:
        transcribed = await in_queue.get()
        logger.debug("voice generating...")
        try:
            demojized = demojize(transcribed.text)
            speech, _ = await to_thread(vr2.text_to_speech, demojized, raw=True)
            logger.debug("voice generated")
            worker_output = WorkerOutput.from_input(transcribed)
            worker_output.sound = SoundOutput(
                data=speech, rate=44110, format=SampleFormat.INT16, channels=1
            )
            yield worker_output
        except Exception as e:
            logger.exception(e)


async def voicevox_worker(vvox_config: VoicevoxConfig, in_queue: Queue[WorkerInput]):
    from vspeech.lib.voicevox import Voicevox

    vvox = Voicevox(vvox_config.openjtalk_dir)
    loaded_models: list[int] = []
    logger.info("tts worker [voicevox] started")
    while True:
        transcribed = await in_queue.get()
        logger.debug("voice generating...")
        try:
            given_speaker_id = transcribed.current_event.params.speaker_id
            speaker_id = (
                given_speaker_id
                if given_speaker_id is not None
                else vvox_config.speaker_id
            )
            if speaker_id not in loaded_models:
                vvox.load_model(speaker_id)
                loaded_models.append(speaker_id)
            given_speed = transcribed.current_event.params.speed
            given_pitch = transcribed.current_event.params.pitch
            audio_query = VoicevoxParam(
                **vvox_config.params.dict(exclude={"speed_scale", "pitch_scale"}),
                speed_scale=given_speed
                if given_speed is not None
                else vvox_config.params.speed_scale,
                pitch_scale=given_pitch
                if given_pitch is not None
                else vvox_config.params.pitch_scale,
            )
            demojized = demojize(transcribed.text)
            speech = await to_thread(
                vvox.voicevox_tts,
                text=demojized,
                speaker_id=speaker_id,
                params=audio_query,
            )
            logger.debug("voice generated")
            worker_output = WorkerOutput.from_input(transcribed)
            worker_output.sound = SoundOutput(
                data=speech[44:], rate=24000, format=SampleFormat.INT16, channels=1
            )
            worker_output.text = transcribed.text
            yield worker_output
        except UnicodeEncodeError as e:
            logger.exception("%s", e)
        except ValueError as e:
            logger.warning("%s", e)


async def tts_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    try:
        while True:
            context.reset_need_reload()
            config = context.config.tts
            if config.worker_type == TtsWorkerType.VR2:
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
                    if context.config.vr2.voice_name:
                        voice_name = context.config.vr2.voice_name
                    else:
                        voice_list: List[str] = vr2.list_voices()
                        if voice_list:
                            voice_name = voice_list[0]
                        else:
                            raise Exception("No voice library")
                    logger.info("vr2 voice: %s", voice_name)
                    vr2.load_voice(voice_name, context.config.vr2.params)
                    worker = partial(vroid2_worker, vr2_config=context.config.vr2)
                    async for output in worker(vr2_=vr2, in_queue=in_queue):
                        out_queue.put_nowait(output)
                        if context.need_reload:
                            break
                    if not context.running.is_set():
                        await context.running.wait()
            elif config.worker_type == TtsWorkerType.VOICEVOX:
                worker = partial(voicevox_worker, vvox_config=context.config.voicevox)
                async for output in worker(in_queue=in_queue):
                    out_queue.put_nowait(output)
                    if context.need_reload:
                        break
                if not context.running.is_set():
                    await context.running.wait()
            else:
                raise ValueError("tts worker type unknown.")
    except CancelledError as e:
        raise shutdown_worker(e)


def create_tts_task(
    tg: TaskGroup,
    context: SharedContext,
):
    worker = context.add_worker(
        event=EventType.tts, configs_depends_on=["tts", "vr2", "voicevox"]
    )
    task = tg.create_task(
        tts_worker(context, in_queue=worker.in_queue, out_queue=context.sender_queue),
        name=worker.event.name,
    )
    return task
