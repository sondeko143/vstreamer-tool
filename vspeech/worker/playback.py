from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from asyncio import to_thread
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from time import time
from typing import NoReturn

import audioop
import sounddevice as sd

from vspeech.config import PlaybackConfig
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import get_device_info
from vspeech.lib.audio import get_sd_dtype
from vspeech.lib.audio import search_device
from vspeech.lib.audio import search_device_by_name
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def record_playback_e2e(
    speech: WorkerInput, now: float, cfg: TelemetryConfig
) -> float | None:
    if speech.origin_ts <= 0.0:
        return None
    e2e = now - speech.origin_ts
    if e2e < 0.0 or e2e > cfg.skew_hard_ceiling_sec:
        # Negative (clock went backwards) or implausibly large: genuine skew.
        logger.warning(
            "clock skew suspected: e2e=%.3fs trace=%s (NTP同期を確認)",
            e2e,
            speech.trace_id,
        )
        return None
    if e2e > cfg.skew_warn_threshold:
        # Above the warn threshold but plausible: a long utterance or playback
        # backlog tail, not skew. Warn, but still record so the telemetry tail
        # reflects real viewer latency.
        logger.warning(
            "high e2e (playback backlog/long utterance): e2e=%.3fs trace=%s",
            e2e,
            speech.trace_id,
        )
    telemetry.record_e2e(e2e, trace_id=speech.trace_id)
    if cfg.log_raw_e2e:
        logger.info("e2e trace=%s %.3fs", speech.trace_id, e2e)
    return e2e


@dataclass
class OutputStream:
    config: InitVar[PlaybackConfig]
    rate: int = 0
    format: SampleFormat = SampleFormat.INVALID
    channels: int = 0
    stream: sd.RawOutputStream | None = None
    device: DeviceInfo = field(init=False)

    def __post_init__(self, config: PlaybackConfig) -> None:
        self.device = get_output_device(config=config)
        logger.info("setting device %s: %s", self.device.index, self.device.name)

    def update_stream_if_changed(
        self,
        rate: int,
        format: SampleFormat,
        channels: int,
    ):
        output_device = get_device_info(self.device.index)
        if (
            self.stream
            and self.rate == rate
            and self.format == format
            and self.channels == channels
            and output_device.name == self.device.name
        ):
            logger.debug("stream is reused.")
            return

        if self.stream:
            self.stream.close()
        self.device = self.search_appropriate_device()
        logger.info("use device %s: %s", self.device.index, self.device.name)
        self.rate = rate
        self.format = format
        self.channels = channels
        self.stream = sd.RawOutputStream(
            samplerate=rate,
            channels=channels,
            device=self.device.index,
            dtype=get_sd_dtype(format),
        )
        self.stream.start()

    def search_appropriate_device(self):
        # Deferred: search_device_by_name reads sd.query_devices(), whose device
        # list is cached at PortAudio init, so a device hot-plugged/reconnected
        # after startup is not seen (the old PyAudio path re-created PyAudio() to
        # re-enumerate). Fixed-device setups (e.g. "Line 4") never hit this;
        # re-enumerating needs sd._terminate()/_initialize() (private API) whose
        # side effects aren't worth verifying for this edge case.
        output_device = search_device_by_name(
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


def get_output_device(config: PlaybackConfig):
    output_device_index = config.output_device_index
    if output_device_index is None:
        output_device = search_device(
            host_api_type=config.output_host_api_name,
            name=config.output_device_name,
            output=True,
        )
        if not output_device:
            raise TypeError("not found output device")
        output_device_index = output_device.index
    return get_device_info(output_device_index)


async def sd_playback_worker(
    config: PlaybackConfig,
    telemetry_config: TelemetryConfig,
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
                logger.debug("playback... %s", speech.text)
                with telemetry.timer("playback", trace_id=speech.trace_id):
                    await output_stream.playback(
                        volume=given_volume
                        if given_volume is not None
                        else config.volume,
                        data=speech.sound.data,
                    )
                logger.debug("playback end")
                record_playback_e2e(speech, now=time(), cfg=telemetry_config)
                worker_output = WorkerOutput.from_input(speech)
                worker_output.sound = SoundOutput.from_input(speech.sound)
                worker_output.text = speech.text
                yield worker_output
            except Exception as e:
                logger.warning("%s", e)
    finally:
        if output_stream.stream:
            output_stream.stream.close()


async def playback_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    try:
        while True:
            context.reset_need_reload()
            async for output in sd_playback_worker(
                config=context.config.playback,
                telemetry_config=context.config.telemetry,
                in_queue=in_queue,
            ):
                out_queue.put_nowait(output)
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
        playback_worker(
            context, in_queue=worker.in_queue, out_queue=context.sender_queue
        ),
        name=worker.event.name,
    )
    return task
