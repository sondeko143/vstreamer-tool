from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy
from pyaudio import PyAudio
from pyaudio import Stream
from pyaudio import paInt8
from pyaudio import paInt16
from pyaudio import paInt24
from pyaudio import paInt32
from pyaudio import paUInt8

from vspeech.audio import get_device_name
from vspeech.audio import search_device
from vspeech.config import Config
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import Message
from vspeech.shared_context import SharedContext

DTYPE_CONV: Dict[int, str] = {
    paInt8: "int8",
    paInt16: "int16",
    paInt24: "int24",
    paInt32: "int32",
    paUInt8: "uint8",
}

AMP_MAX: Dict[int, int] = {
    paInt8: pow(2, 7) - 1,
    paInt16: pow(2, 15) - 1,
    paInt24: pow(2, 23) - 1,
    paInt32: pow(2, 31) - 1,
    paUInt8: pow(2, 8) - 1,
}


def open_input_stream(
    audio: PyAudio,
    config: Config,
    stream_callback: Optional[
        Callable[[Optional[bytes], int, Any, int], Tuple[Optional[bytes], int]]
    ] = None,
):
    input_device_index = config.input_device_index
    if not input_device_index:
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
        format=config.format,
        channels=config.channels,
        rate=config.rate,
        input=True,
        output=False,
        frames_per_buffer=config.chunk,
        stream_callback=stream_callback,
    )


RecordingOutput = bytes


async def recording_task_loop(
    stream: Stream,
    config: Config,
):
    interval_frame_count = 0
    interval_frames: bytes = b""
    speaking_frames: bytes = b""
    last_interval_frames: bytes = b""
    total_speaking_seconds = 0
    status = "waiting"
    while stream.is_active():
        in_data = await to_thread(stream.read, config.chunk)
        interval_frame_count += config.chunk
        interval_frames += in_data
        if interval_frame_count >= config.rate * config.record_interval_sec:
            framebuffer = numpy.frombuffer(
                interval_frames, dtype=DTYPE_CONV[config.format]
            )
            in_data_min = framebuffer.min()
            in_data_max = framebuffer.max()
            approx_max_amp = 100 * (
                max(-in_data_min, in_data_max) / AMP_MAX[config.format]
            )
            logger.debug("amp: %s", approx_max_amp)
            speaking = approx_max_amp > config.silence_threshold
            if status == "waiting":
                if speaking:
                    logger.info("voice recording...")
                    speaking_frames += interval_frames + last_interval_frames
                    status = "speaking"
            elif status == "speaking":
                speaking_frames += interval_frames
                total_speaking_seconds += config.record_interval_sec
                logger.debug(
                    f"min: {in_data_min}, max: {in_data_max}, amp: {approx_max_amp} thr: {config.silence_threshold}"
                )
                if not speaking or config.max_recording_sec < total_speaking_seconds:
                    logger.info("voice recorded")
                    yield speaking_frames
                    status = "waiting"
                    speaking_frames = b""
            last_interval_frames = interval_frames
            interval_frame_count = 0
            interval_frames = b""


async def recording_worker(
    context: SharedContext, out_queue: Queue[Message[RecordingOutput]]
):
    while True:
        stream = open_input_stream(context.audio, context.config)
        try:
            async for frames in recording_task_loop(
                stream=stream, config=context.config
            ):
                if not context.resume.is_set():
                    logger.info("recording have been paused")
                    break
                out_queue.put_nowait(
                    Message[RecordingOutput](source=EventType.recording, content=frames)
                )
        except CancelledError:
            logger.debug("recording worker cancelled")
            stream.close()
            raise
        stream.close()
        try:
            await context.resume.wait()
        except CancelledError:
            logger.debug("recording worker cancelled")
            raise


def create_recording_task(loop: AbstractEventLoop, context: SharedContext):
    return loop.create_task(
        recording_worker(context, out_queue=context.broker_queue), name="recording_task"
    )
