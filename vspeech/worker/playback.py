import audioop
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import to_thread
from dataclasses import dataclass
from typing import NoReturn

from pyaudio import PyAudio
from pyaudio import Stream as PaStream

from vspeech.config import PlaybackConfig
from vspeech.config import SampleFormat
from vspeech.config import get_sample_size
from vspeech.lib.audio import get_device_name
from vspeech.lib.audio import get_pa_format
from vspeech.lib.audio import search_device
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


@dataclass
class OutputStream:
    device_index: int = 0
    rate: int = 0
    format: SampleFormat = SampleFormat.INVALID
    channels: int = 0
    stream: PaStream | None = None

    def update_stream_if_changed(
        self,
        audio: PyAudio,
        device_index: int,
        rate: int,
        format: SampleFormat,
        channels: int,
    ):
        if (
            self.device_index == device_index
            and self.rate == rate
            and self.format == format
            and self.channels == channels
        ):
            return
        self.close()
        output_device_name = get_device_name(audio, device_index)
        logger.info("use output device %s: %s", device_index, output_device_name)
        self.device_index = device_index
        self.rate = rate
        self.format = format
        self.channels = channels
        self.stream = audio.open(
            format=get_pa_format(format),
            channels=channels,
            rate=rate,
            output=True,
            output_device_index=device_index,
        )

    async def playback(self, volume: int, data: bytes):
        if not self.stream:
            return
        if volume != 100:
            _data = audioop.mul(data, get_sample_size(self.format), volume / 100.0)
        else:
            _data = data
        await to_thread(self.stream.write, _data)

    def close(self):
        if self.stream:
            self.stream.close()


def get_output_device_index(audio: PyAudio, config: PlaybackConfig):
    output_device_index = config.output_device_index
    if output_device_index is None:
        output_device = search_device(
            audio,
            host_api_type=config.output_host_api_name,
            name=config.output_device_name,
            output=True,
        )
        if not output_device:
            raise TypeError("not found output device")
        output_device_index = output_device.index
    return output_device_index


async def pyaudio_playback_worker(
    config: PlaybackConfig,
    in_queue: Queue[WorkerInput],
):
    audio = PyAudio()
    device = get_output_device_index(audio=audio, config=config)
    output_stream = OutputStream()
    try:
        logger.info("playback worker started.")
        while True:
            speech = await in_queue.get()
            try:
                output_stream.update_stream_if_changed(
                    audio=audio,
                    device_index=device,
                    rate=speech.sound.rate,
                    format=speech.sound.format,
                    channels=speech.sound.channels,
                )

                logger.debug("playback...")
                yield await output_stream.playback(
                    volume=config.volume,
                    data=speech.sound.data,
                )
                logger.debug("playback end")
            except Exception as e:
                logger.warning("%s", e)
    finally:
        output_stream.close()
        audio.terminate()


async def playback_worker(context: SharedContext, in_queue: Queue[WorkerInput]):
    try:
        while True:
            context.reset_need_reload()
            async for _ in pyaudio_playback_worker(
                config=context.config.playback, in_queue=in_queue
            ):
                if context.need_reload:
                    break
            if not context.running.is_set():
                await context.running.wait()
    except CancelledError:
        logger.info("playback worker cancelled")
        raise


def create_playback_task(
    loop: AbstractEventLoop,
    context: SharedContext,
) -> Task[NoReturn]:
    in_queue = Queue[WorkerInput]()
    event = EventType.playback
    context.input_queues[event] = in_queue
    task = loop.create_task(
        playback_worker(context, in_queue=in_queue),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
