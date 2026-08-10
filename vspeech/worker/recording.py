from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from asyncio import to_thread
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from math import log
from time import perf_counter
from time import time
from typing import Literal
from typing import NoReturn
from uuid import uuid4

import audioop
import sounddevice as sd

from vspeech.config import EventType
from vspeech.config import RecordingConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import get_sd_dtype
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.audio import resolve_input_device
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerOutput


def device_frames_per_read(chunk: int, device_rate: int, config_rate: int) -> int:
    """How many device-rate frames make up one `chunk`-sized block at config_rate.

    Mirrors stream_vc/capture.py's device_frames_per_read, parameterized by the
    recording pipeline's own rate instead of a fixed CAPTURE_RATE (ADR-0070).
    """
    if device_rate == config_rate:
        return chunk
    return max(1, round(chunk * device_rate / config_rate))


def open_input_stream(config: RecordingConfig) -> tuple[sd.RawInputStream, int]:
    """Open the mic at its native rate; return the stream and that rate.

    Rate resolution sits next to the device resolution that was already here, so an
    open stays a single decision point (the same shape as stream_vc/capture.py's
    opener). Asking the device for `config.rate` directly would hand the conversion
    to the OS, whose filter we can neither test nor log, and WASAPI shared mode
    refuses any rate but its mix format (ADR-0070/0071).
    """
    device = resolve_input_device(config)
    rate, how = resolve_device_rate(
        device,
        config.input_device_rate,
        input=True,
        config_key="recording.input_device_rate",
    )
    # Logged before the open so a failing open still says what was attempted.
    logger.info(
        "use input device %s: %s @%dHz (%s) -> %dHz (%s)",
        device.index,
        device.name,
        rate,
        how,
        config.rate,
        "プロセス内で変換" if rate != config.rate else "変換なし",
    )
    stream = sd.RawInputStream(
        samplerate=rate,
        blocksize=device_frames_per_read(config.chunk, rate, config.rate),
        device=device.index,
        channels=config.channels,
        dtype=get_sd_dtype(config.format),
    )
    stream.start()
    # PortAudio may know the endpoint runs at a slightly different rate than the one
    # it accepted. We keep converting at the requested rate (the L/M ratio has to be
    # built from a sane number: 44099 -> 16000 would mean 16000 phases), so a delta
    # shows up only as a slow drift in the audio -- invisible unless said out loud.
    reported = float(stream.samplerate)
    if abs(reported - rate) > 0.5:
        logger.warning(
            "recording device reports %.4fHz for a requested %dHz; "
            "converting at the requested rate",
            reported,
            rate,
        )
    return stream, rate


def convert_chunk(
    data: bytes, resampler: PolyphaseResampler | None, config: RecordingConfig
) -> tuple[bytes, int]:
    """Convert one device-rate read into bytes at config.rate, plus the number of
    config.rate frames those bytes represent.

    Returns `data` untouched when `resampler` is None (the device already runs at
    config.rate), which keeps that path bit-identical to the pre-ADR-0070 code --
    decode+encode is not bit-exact at full-scale values (e.g. int16 -32768 round-trips
    to -32767 through decode_pcm/encode_pcm), so skipping the round trip matters, not
    just its cost.

    The returned frame count is measured from the actual conversion, not a config
    constant: under resampling, the polyphase filter's per-call output length is not
    fixed (it depends on the running phase), so a constant would drift from the real
    elapsed time -- exactly the trap this task exists to avoid, since interval_sec /
    max_recording_sec / silence timing all compare against this count.
    """
    if resampler is None:
        frame_size = get_sample_size(config.format) * config.channels
        return data, len(data) // frame_size
    samples = decode_pcm(data, config.format, config.channels)
    converted = resampler.process(samples)
    # encode_pcm saturates: resampling overshoots the original peak (Gibbs), and a
    # wrapping cast would turn that overshoot into a sign flip = an audible click.
    return encode_pcm(converted, config.format), converted.shape[0]


def get_dbfs(interval_frames: bytes, sample_width: int):
    rms = audioop.rms(interval_frames, sample_width)
    max_possible_val = (2 ** (sample_width * 8)) / 2
    if rms == 0:
        return float("-inf")
    return 20 * log(rms / max_possible_val, 10)


@dataclass
class RecordedUtterance:
    frames: bytes
    capture_sec: float
    silence_lag: float
    stop_reason: Literal["silence", "maxlen"]


def utterance_capture_sec(frames: bytes, config: RecordingConfig) -> float:
    denom = get_sample_size(config.format) * config.channels * config.rate
    return len(frames) / denom


def record_recording_metrics(
    capture_sec: float, silence_lag: float, stop_reason: str, trace_id: str = ""
) -> None:
    telemetry.record("rec_capture", capture_sec, trace_id=trace_id)
    if stop_reason == "silence":
        telemetry.record("rec_silence_lag", silence_lag, trace_id=trace_id)


async def sd_recording_worker(
    config: RecordingConfig,
) -> AsyncGenerator[RecordedUtterance]:
    while True:
        interval_frame_count = 0
        interval_frames: bytes = b""
        speaking_frames: bytes = b""
        last_interval_frames_buffer: deque[bytes] = deque(
            maxlen=config.last_interval_frames_buffer_size
        )
        total_seconds_of_this_recording = 0
        status = "waiting"
        last_voice_ts = perf_counter()
        with worker_startup("recording"):
            stream, device_rate = open_input_stream(config)
        sample_width = get_sample_size(config.format)
        n_move_avg_amp = config.gradually_stopping_interval
        approx_max_amps: list[float] = []
        # Built once per stream open, not inside the read loop: construction costs
        # 0.2-8ms measured across the rate pairs this boundary meets, and the read
        # cadence here (>= one chunk, 64ms by default) makes a rebuild cost invisible
        # either way, but a fresh resampler starts from a zeroed filter tail, so
        # rebuilding per read would put a transient at every chunk boundary. None on a
        # matching rate keeps the pass-through path bit-identical to the
        # pre-ADR-0070 code.
        resampler = make_resampler(device_rate, config.rate)
        frames_per_read = device_frames_per_read(config.chunk, device_rate, config.rate)
        try:
            while stream.active:
                chunk_data, overflowed = await to_thread(stream.read, frames_per_read)
                if overflowed:
                    # sounddevice reports an overflow with a flag rather than an
                    # exception, so at least leave a log line.
                    logger.warning("recording input overflow: samples were dropped")
                in_data, frame_count = convert_chunk(
                    bytes(chunk_data), resampler, config
                )
                interval_frame_count += frame_count
                interval_frames += in_data
                if interval_frame_count >= config.rate * config.interval_sec:
                    approx_max_amp = get_dbfs(
                        interval_frames, sample_width=sample_width
                    )
                    speaking = approx_max_amp >= config.silence_threshold
                    if status == "waiting" and speaking:
                        logger.info("record start ")
                        speaking_frames += (
                            b"".join(last_interval_frames_buffer) + interval_frames
                        )
                        status = "speaking"
                        last_voice_ts = perf_counter()
                        approx_max_amps = []
                    elif status == "speaking":
                        speaking_frames += interval_frames
                        # [Open, deferred 2026-08-11] This adds the constant
                        # config.interval_sec per tick, but the `>=` check above (not `==`)
                        # lets interval_frame_count -- and therefore the real audio inside
                        # interval_frames -- overshoot the threshold by up to one read's
                        # worth before this branch fires. Each tick therefore represents MORE
                        # real audio than it is credited with, so max_recording_sec caps LESS
                        # real time than configured. At the defaults (chunk=1024,
                        # interval_sec=0.1, rate=16000: threshold=1600 frames, but reads land
                        # on 1024/2048/... so a tick fires every 2048 frames = 0.128s of real
                        # audio credited as only 0.1s), a configured 0.25s cap is closer to
                        # ~0.32s in practice. Pre-existing (unchanged by ADR-0070's
                        # device-rate read -- at a matching rate the read size and the
                        # overshoot geometry are identical to the pre-ADR-0070 code); out of
                        # this task's scope. A fix would need to measure real elapsed frames
                        # per tick instead of crediting a constant.
                        total_seconds_of_this_recording += config.interval_sec
                        if speaking:
                            last_voice_ts = perf_counter()
                        approx_max_amps.append(approx_max_amp)
                        if len(approx_max_amps) > n_move_avg_amp:
                            approx_max_amps.pop(0)
                        avg_amp = sum(approx_max_amps) / len(approx_max_amps)
                        silent = avg_amp < config.silence_threshold
                        if (
                            silent
                            or config.max_recording_sec
                            < total_seconds_of_this_recording
                        ):
                            stop_reason = "silence" if silent else "maxlen"
                            silence_lag = (
                                perf_counter() - last_voice_ts if silent else 0.0
                            )
                            logger.info(
                                "record stop %s reason=%s lag=%.3f",
                                avg_amp,
                                stop_reason,
                                silence_lag,
                            )
                            yield RecordedUtterance(
                                frames=speaking_frames,
                                capture_sec=utterance_capture_sec(
                                    speaking_frames, config
                                ),
                                silence_lag=silence_lag,
                                stop_reason=stop_reason,
                            )
                            status = "waiting"
                            speaking_frames = b""
                            interval_frames = b""
                            last_interval_frames_buffer.clear()
                            total_seconds_of_this_recording = 0
                    last_interval_frames_buffer.append(interval_frames)
                    interval_frame_count = 0
                    interval_frames = b""
        except (OSError, sd.PortAudioError) as e:
            logger.warning("retry for %r", e)
        finally:
            stream.close()


def build_recording_output(
    config: RecordingConfig, frames: bytes, silence_lag: float = 0.0
) -> WorkerOutput:
    worker_output = WorkerOutput.from_routes_list(config.routes_list)
    worker_output.trace_id = uuid4().hex
    worker_output.origin_ts = time() - silence_lag
    worker_output.sound = SoundOutput(
        data=frames,
        rate=config.rate,
        format=config.format,
        channels=config.channels,
    )
    return worker_output


async def recording_worker(context: SharedContext, out_queue: Queue[WorkerOutput]):
    try:
        while True:
            context.reset_need_reload()
            rec_config = context.config.recording
            async for utterance in sd_recording_worker(
                config=rec_config,
            ):
                if not context.running.is_set():
                    logger.info("recording have been paused")
                    break
                worker_output = build_recording_output(
                    rec_config, utterance.frames, silence_lag=utterance.silence_lag
                )
                record_recording_metrics(
                    capture_sec=utterance.capture_sec,
                    silence_lag=utterance.silence_lag,
                    stop_reason=utterance.stop_reason,
                    trace_id=worker_output.trace_id,
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
