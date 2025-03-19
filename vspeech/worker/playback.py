import audioop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from asyncio import to_thread
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from typing import NoReturn

from pyaudio import PyAudio
from pyaudio import Stream as PaStream

from vspeech.config import PlaybackConfig
from vspeech.config import SampleFormat
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import get_device_info
from vspeech.lib.audio import get_pa_format
from vspeech.lib.audio import search_device
from vspeech.lib.audio import search_device_by_name
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


@dataclass
class OutputStream:
    config: InitVar[PlaybackConfig]
    rate: int = 0
    format: SampleFormat = SampleFormat.INVALID
    channels: int = 0
    stream: PaStream | None = None
    audio: PyAudio = field(init=False)
    device: DeviceInfo = field(init=False)

    def __post_init__(self, config: PlaybackConfig):
        self.audio = PyAudio()
        self.device = get_output_device(audio=self.audio, config=config)
        logger.info("setting device %s: %s", self.device.index, self.device.name)

    def update_stream_if_changed(
        self,
        rate: int,
        format: SampleFormat,
        channels: int,
    ):
        output_device = get_device_info(self.audio, self.device.index)
        if (
            self.stream
            and self.rate == rate
            and self.format == format
            and self.channels == channels
            and output_device.name == self.device.name
        ):
            logger.debug("stream is reused.")
            return

        self.audio.terminate()
        del self.audio
        self.audio = PyAudio()
        self.device = self.search_appropriate_device()
        logger.info("use device %s: %s", self.device.index, self.device.name)
        self.rate = rate
        self.format = format
        self.channels = channels
        self.stream = self.audio.open(
            format=get_pa_format(format),
            channels=channels,
            rate=rate,
            output=True,
            output_device_index=self.device.index,
        )

    def search_appropriate_device(self):
        output_device = search_device_by_name(
            self.audio,
            host_api_index=self.device.host_api,
            name=self.device.name,
            output=True,
        )
        if not output_device:
            raise TypeError(f"not found output device {self.device.name}")
        return output_device

    async def playback(self, volume: int, data: bytes):
        if not self.stream:
            return
        if volume != 100:
            _data = audioop.mul(data, get_sample_size(self.format), volume / 100.0)
        else:
            _data = data
        await to_thread(self.stream.write, _data)


def get_output_device(audio: PyAudio, config: PlaybackConfig):
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
    return get_device_info(audio, output_device_index)


async def pyaudio_playback_worker(
    config: PlaybackConfig,
    in_queue: Queue[WorkerInput],
):
    output_stream = OutputStream(config)
    try:
        logger.info("playback worker started.")
        while True:
            speech = await in_queue.get()
            try:
                output_stream.update_stream_if_changed(
                    rate=speech.sound.rate,
                    format=speech.sound.format,
                    channels=speech.sound.channels,
                )
                given_volume = speech.current_event.params.volume
                logger.debug("playback...")
                await output_stream.playback(
                    volume=given_volume if given_volume is not None else config.volume,
                    data=speech.sound.data,
                )
                logger.debug("playback end")
                worker_output = WorkerOutput.from_input(speech)
                worker_output.sound = SoundOutput.from_input(speech.sound)
                worker_output.text = speech.text
                yield worker_output
            except Exception as e:
                logger.warning("%s", e)
    finally:
        output_stream.audio.terminate()
        del output_stream.audio


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
    except CancelledError as e:
        raise shutdown_worker(e)


def create_playback_task(
    tg: TaskGroup,
    context: SharedContext,
) -> Task[NoReturn]:
    worker = context.add_worker(
        event=EventType.playback,
        configs_depends_on=["playback"],
    )
    task = tg.create_task(
        playback_worker(context, in_queue=worker.in_queue),
        name=worker.event.name,
    )
    return task
