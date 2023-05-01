import audioop
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread

from pyaudio import PyAudio
from pyaudio import Stream

from vspeech.audio import get_device_name
from vspeech.audio import get_pa_format
from vspeech.audio import search_device
from vspeech.config import PlaybackConfig
from vspeech.config import SampleFormat
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


async def playback(volume: int, stream: Stream, data: bytes):
    if volume != 100:
        _data = audioop.mul(data, 2, volume / 100.0)
    else:
        _data = data
    await to_thread(stream.write, _data)


def get_output_stream(
    audio: PyAudio,
    config: PlaybackConfig,
    rate: int,
    format: SampleFormat,
    channels: int,
) -> Stream:
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
    output_device_name = get_device_name(audio, output_device_index)
    logger.info("use output device %s: %s", output_device_index, output_device_name)
    output_stream = audio.open(
        format=get_pa_format(format),
        channels=channels,
        rate=rate,
        output=True,
        output_device_index=output_device_index,
    )
    return output_stream


async def playback_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    audio = PyAudio()
    try:
        while True:
            speech = await in_queue.get()
            try:
                output_stream = get_output_stream(
                    audio=audio,
                    config=context.config.playback,
                    rate=speech.sound.rate,
                    format=speech.sound.format,
                    channels=speech.sound.channels,
                )
                logger.info("playback...")
                await playback(
                    volume=context.config.playback.speech_volume,
                    stream=output_stream,
                    data=speech.sound.data,
                )
                logger.info("playback end")
            except Exception as e:
                logger.warning(e)
    except CancelledError:
        logger.info("playback worker cancelled")
        raise
    finally:
        audio.terminate()


def create_playback_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.playback
    context.input_queues[event] = in_queue
    return loop.create_task(
        playback_worker(context, in_queue=in_queue),
        name=event.name,
    )
