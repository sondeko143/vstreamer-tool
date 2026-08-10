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
from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import get_device_info
from vspeech.lib.audio import get_sd_dtype
from vspeech.lib.audio import open_device_stream
from vspeech.lib.audio import resolve_output_device
from vspeech.lib.audio import search_device_by_name
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

# How many source rates keep a built resampler around. The utterance path alternates
# between sources (TTS 24000Hz, VC 40000Hz, recording 16000Hz, whatever a remote sends),
# and a build costs 0.3-5.7 ms measured across those pairs against a 48000/44100 device,
# so keeping them beats rebuilding on every alternation. There is a cap because the key
# arrives with the audio (`WorkerInput.sound.rate` crosses gRPC from another machine with
# no validation on the way), and past it the least recently used entry is evicted.
#
# What a *bounded* table costs is bounded by resample.MAX_PROTOTYPE_TAPS, not by this
# number: that cap is what stops one entry from being a 563MB filter (ADR-0075). Under it,
# an entry is 1-180 KB of taps for the pairs these boundaries actually meet, and about
# 4 MB (one float32 per tap) for the very largest ratio the cap admits at all.
MAX_CACHED_RESAMPLERS = 8


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
    """The output device, held open at its own rate, plus the converters that feed it.

    The rate is decided when the device is opened and stays fixed for the life of that
    stream; an utterance arriving at another rate is converted into it here instead of the
    device being reopened at the source's rate (ADR-0073/0074). A source rate change is
    therefore no longer a reason to reopen -- only the sample format, the channel count or
    the device itself changing is. Asking the device for the source's rate would hand the
    conversion to the OS, whose filter we can neither test nor log, and WASAPI shared mode
    refuses any rate but its mix format.

    One utterance is one self-contained buffer, not a slice of a continuous stream, so the
    conversion goes through `resample_full`, which flushes the filter and removes the group
    delay and leaves no state behind. That is what makes the cached resamplers safe to
    share across utterances: nothing of one utterance can reach the next.
    """

    config: InitVar[PlaybackConfig]
    device_rate: int = 0
    format: SampleFormat = SampleFormat.INVALID
    channels: int = 0
    stream: sd.RawOutputStream | None = None
    device: DeviceInfo = field(init=False)
    rate_override: int | None = field(init=False)
    resamplers: dict[int, PolyphaseResampler | None] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self, config: PlaybackConfig) -> None:
        self.device = get_output_device(config=config)
        # The only part of the config that outlives the constructor: the device is
        # resolved here, and the rate it is opened at is decided per open, next to it.
        self.rate_override = config.output_device_rate
        logger.info("setting device %s: %s", self.device.index, self.device.name)

    def update_stream_if_changed(
        self,
        format: SampleFormat,
        channels: int,
    ):
        """Open the device, or reopen it when the sample format, the channel count or the
        device itself changed. **Never for a change of sample rate.**

        The rate used to be part of this check, so a 24000Hz TTS utterance followed by a
        40000Hz VC one closed and reopened the device every time. The stream now runs at
        the device's own rate and the source is converted into it.
        """
        output_device = get_device_info(self.device.index)
        if (
            self.stream
            and self.format == format
            and self.channels == channels
            and output_device.name == self.device.name
        ):
            logger.debug("stream is reused.")
            return

        if self.stream:
            self.stream.close()
            # Cleared before the open below gets a chance to fail: leaving a closed
            # stream in place would let playback() write to it.
            self.stream = None
        self.device = self.search_appropriate_device()
        self.format = format
        self.channels = channels
        # The cached resamplers were built against the rate of the stream just closed,
        # and this open may land on a different device with a different rate.
        self.resamplers.clear()
        self.stream, self.device_rate = open_device_stream(
            device=self.device,
            override=self.rate_override,
            input=False,
            config_key="playback.output_device_rate",
            opening="use output device",
            subject="playback",
            open_stream=lambda rate: sd.RawOutputStream(
                samplerate=rate,
                channels=channels,
                device=self.device.index,
                dtype=get_sd_dtype(format),
            ),
        )

    def resampler_for(self, rate: int) -> PolyphaseResampler | None:
        """The resampler from `rate` to the device rate, or None when they match.

        Built once per source rate and kept (see MAX_CACHED_RESAMPLERS): the utterance
        path alternates between sources, so rebuilding whenever the rate changes would pay
        the build cost on nearly every utterance. Keeping them carries nothing between
        utterances -- `resample_full` resets the filter on both sides of the call.

        The table is a plain dict used as an LRU (insertion order is the recency order).
        Evicting one entry rather than clearing the table keeps a source rotating through
        more rates than fit at one rebuild per miss instead of a whole table's worth.
        """
        if rate in self.resamplers:
            # Re-insert to move it to the most-recently-used end.
            self.resamplers[rate] = self.resamplers.pop(rate)
            return self.resamplers[rate]
        # The table is not touched at all until the build has succeeded -- neither the new
        # key recorded nor the oldest evicted. make_resampler rejects a rate it cannot
        # serve, and a table mutated first would either claim "no conversion needed" for a
        # rate that never resolved (playing the next such utterance unconverted, i.e.
        # silently at the wrong speed) or throw away a warm entry to make room for a build
        # that never happened. Building before evicting means one extra resampler is alive
        # for the length of this call; MAX_PROTOTYPE_TAPS bounds what that costs.
        resampler = make_resampler(rate, self.device_rate)
        if len(self.resamplers) >= MAX_CACHED_RESAMPLERS:
            del self.resamplers[next(iter(self.resamplers))]
        self.resamplers[rate] = resampler
        logger.info(
            "playback %dHz -> %dHz (%s)",
            rate,
            self.device_rate,
            "変換なし" if resampler is None else "プロセス内で変換",
        )
        return resampler

    def convert(
        self, data: bytes, rate: int, format: SampleFormat, channels: int
    ) -> bytes:
        """`data` (PCM at `rate`) as PCM at the device rate.

        Returns the input object untouched when the rates already match, which keeps that
        path bit-identical to the pre-ADR-0073 code -- decode+encode is not bit-exact at
        full scale (int16 -32768 comes back as -32767), so skipping the round trip
        matters, not just its cost.
        """
        resampler = self.resampler_for(rate)
        if resampler is None:
            return data
        samples = decode_pcm(data, format, channels)
        # resample_full, not process: an utterance is a self-contained buffer, and the
        # streaming entry point would leave the last `delay_samples` inside the filter,
        # clipping the tail off every utterance. encode_pcm saturates, because resampling
        # overshoots the original peak (Gibbs) and a wrapping cast would turn that
        # overshoot into a sign flip = an audible click.
        return encode_pcm(resampler.resample_full(samples), format)

    def search_appropriate_device(self):
        # Deferred: search_device_by_name reads sd.query_devices(), cached at
        # PortAudio init, so a device hot-plugged after startup is not seen.
        # Fixed-device setups (e.g. "Line 4") never hit this; re-enumerating
        # needs sd._terminate()/_initialize() (private API) not worth verifying
        # for this edge case.
        output_device = search_device_by_name(
            host_api_index=self.device.host_api,
            name=self.device.name,
            output=True,
        )
        if not output_device:
            raise TypeError(f"not found output device {self.device.name}")
        return output_device

    async def playback(self, volume: int, sound: SoundInput):
        stream = self.stream
        if stream is None:
            return
        # Volume, conversion and the blocking write all go to the same worker thread.
        # Resampling a whole utterance is real CPU work, and doing it on the event loop
        # would stall every other worker for its duration.
        await to_thread(self._write, stream, volume, sound)

    def _write(
        self, stream: sd.RawOutputStream, volume: int, sound: SoundInput
    ) -> None:
        """Apply the volume, convert to the device rate, and write. Off the event loop.

        The volume is applied first, to the source bytes, exactly as it was before
        ADR-0073: at a matching rate the bytes reaching the device are byte-for-byte the
        ones the old code wrote.
        """
        data = sound.data
        if volume != 100:
            data = audioop.mul(data, get_sample_size(sound.format), volume / 100.0)
        stream.write(self.convert(data, sound.rate, sound.format, sound.channels))


def get_output_device(config: PlaybackConfig):
    return resolve_output_device(config)


async def sd_playback_worker(
    config: PlaybackConfig,
    telemetry_config: TelemetryConfig,
    in_queue: Queue[WorkerInput],
):
    with worker_startup("playback"):
        output_stream = OutputStream(config)
    try:
        logger.info("playback worker started.")
        while True:
            speech = await in_queue.get()
            try:
                output_stream.update_stream_if_changed(
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
                        sound=speech.sound,
                    )
                logger.debug("playback end")
                record_playback_e2e(speech, now=time(), cfg=telemetry_config)
                worker_output = WorkerOutput.from_input(speech)
                # The ORIGINAL sound travels on, not the device-rate conversion: the
                # conversion exists for this device, and a following step (another host's
                # playback, a file) has its own boundary to convert at.
                worker_output.sound = SoundOutput.from_input(speech.sound)
                worker_output.text = speech.text
                yield worker_output
            except DeviceRateUnresolvedError:
                # A rate that cannot be decided is a config problem, not a device fault:
                # no retry fixes it, and swallowing it into the warning below would leave
                # the pipeline playing nothing at all, silently, for every utterance. Fail
                # loud like the three other device boundaries (ADR-0074).
                raise
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
