import audioop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from asyncio import to_thread
from math import log
from typing import Any
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Tuple

from pyaudio import PyAudio

from vspeech.config import EventType
from vspeech.config import RecordingConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.audio import get_device_name
from vspeech.lib.audio import get_pa_format
from vspeech.lib.audio import search_device
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerMeta
from vspeech.shared_context import WorkerOutput


def open_input_stream(
    audio: PyAudio,
    config: RecordingConfig,
    stream_callback: Optional[
        Callable[[Optional[bytes], int, Any, int], Tuple[Optional[bytes], int]]
    ] = None,
):
    input_device_index = config.input_device_index
    if input_device_index is None:
        input_device = search_device(
            audio,
            host_api_type=config.input_host_api_name,
            name=config.input_device_name,
            input=True,
        )
        if not input_device:
            raise TypeError("not found input device")
        input_device_index = input_device.index
    input_device_name = get_device_name(audio, input_device_index)
    logger.info("use input device %s: %s", input_device_index, input_device_name)
    return audio.open(
        input_device_index=input_device_index,
        format=get_pa_format(config.format),
        channels=config.channels,
        rate=config.rate,
        input=True,
        output=False,
        frames_per_buffer=config.chunk,
        stream_callback=stream_callback,
    )


def get_dbfs(interval_frames: bytes, sample_width: int):
    rms = audioop.rms(interval_frames, sample_width)
    max_possible_val = (2 ** (sample_width * 8)) / 2
    if rms == 0:
        return float("-inf")
    return 20 * log(rms / max_possible_val, 10)


async def pyaudio_recording_worker(config: RecordingConfig):
    interval_frame_count = 0
    interval_frames: bytes = b""
    speaking_frames: bytes = b""
    last_interval_frames: bytes = b""
    total_seconds_of_this_recording = 0
    status = "waiting"
    audio = PyAudio()
    stream = open_input_stream(audio, config)
    sample_width = get_sample_size(config.format)
    stopping_time = 0
    try:
        while stream.is_active():
            in_data = await to_thread(stream.read, config.chunk)
            interval_frame_count += config.chunk
            interval_frames += in_data
            approx_max_amp = get_dbfs(interval_frames, sample_width=sample_width)
            if interval_frame_count >= config.rate * config.interval_sec:
                speaking = approx_max_amp > config.silence_threshold
                if status == "waiting" and speaking:
                    logger.debug("voice recording...")
                    speaking_frames += last_interval_frames + interval_frames
                    status = "speaking"
                elif status == "speaking":
                    speaking_frames += interval_frames
                    total_seconds_of_this_recording += config.interval_sec
                    if (
                        not speaking
                        or config.max_recording_sec < total_seconds_of_this_recording
                    ):
                        logger.debug("voice stopped")
                        status = "stopped"
                elif status == "stopped":
                    speaking_frames += interval_frames
                    total_seconds_of_this_recording += config.interval_sec
                    if (
                        stopping_time > config.gradually_stopping_interval
                        or config.max_recording_sec < total_seconds_of_this_recording
                    ):
                        logger.debug(
                            "voice recorded %s %s",
                            stopping_time,
                            total_seconds_of_this_recording,
                        )
                        yield speaking_frames
                        status = "waiting"
                        speaking_frames = b""
                        interval_frames = b""
                        stopping_time = 0
                        total_seconds_of_this_recording = 0
                    elif not speaking:
                        stopping_time += 1
                        logger.debug("voice stopping")
                    elif speaking:
                        stopping_time = 0
                        status = "speaking"
                last_interval_frames = interval_frames
                interval_frame_count = 0
                interval_frames = b""
    finally:
        stream.close()
        audio.terminate()


async def recording_worker(context: SharedContext, out_queue: Queue[WorkerOutput]):
    try:
        while True:
            context.reset_need_reload()
            rec_config = context.config.recording
            async for frames in pyaudio_recording_worker(
                config=rec_config,
            ):
                if not context.running.is_set():
                    logger.info("recording have been paused")
                    break
                worker_output = WorkerOutput.from_routes_list(rec_config.routes_list)
                worker_output.sound = SoundOutput(
                    data=frames,
                    rate=rec_config.rate,
                    format=rec_config.format,
                    channels=rec_config.channels,
                )
                out_queue.put_nowait(worker_output)
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError as e:
        raise shutdown_worker(e)


def create_recording_task(tg: TaskGroup, context: SharedContext) -> Task[NoReturn]:
    worker = context.add_worker(
        event=EventType.recording,
        configs_depends_on=["recording"],
    )
    task = tg.create_task(
        recording_worker(context, out_queue=context.sender_queue),
        name=worker.event.name,
    )
    return task
