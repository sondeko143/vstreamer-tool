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
from typing import NoReturn
from uuid import uuid4

import audioop
import sounddevice as sd

from vspeech.config import EventType
from vspeech.config import RecordingConfig
from vspeech.config import get_sample_size
from vspeech.exceptions import shutdown_worker
from vspeech.lib.audio import get_device_name
from vspeech.lib.audio import get_sd_dtype
from vspeech.lib.audio import search_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerOutput


def open_input_stream(config: RecordingConfig) -> sd.RawInputStream:
    input_device_index = config.input_device_index
    if input_device_index is None:
        input_device = search_device(
            host_api_type=config.input_host_api_name,
            name=config.input_device_name,
            input=True,
        )
        if not input_device:
            raise TypeError("not found input device")
        input_device_index = input_device.index
    input_device_name = get_device_name(input_device_index)
    logger.info("use input device %s: %s", input_device_index, input_device_name)
    stream = sd.RawInputStream(
        samplerate=config.rate,
        blocksize=config.chunk,
        device=input_device_index,
        channels=config.channels,
        dtype=get_sd_dtype(config.format),
    )
    stream.start()
    return stream


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
    stop_reason: str  # "silence" | "maxlen"


def utterance_capture_sec(frames: bytes, config: RecordingConfig) -> float:
    denom = get_sample_size(config.format) * config.channels * config.rate
    if denom <= 0:
        return 0.0
    return len(frames) / denom


def record_recording_metrics(
    capture_sec: float, silence_lag: float, stop_reason: str, trace_id: str = ""
) -> None:
    telemetry.record("rec_capture", capture_sec, trace_id=trace_id)
    if stop_reason == "silence":
        telemetry.record("rec_silence_lag", silence_lag, trace_id=trace_id)


async def pyaudio_recording_worker(
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
        stream = open_input_stream(config)
        sample_width = get_sample_size(config.format)
        n_move_avg_amp = config.gradually_stopping_interval
        approx_max_amps: list[float] = []
        try:
            while stream.active:
                chunk_data, overflowed = await to_thread(stream.read, config.chunk)
                if overflowed:
                    # sounddevice は overflow で例外を投げず flag で返す。旧 PyAudio は
                    # exception_on_overflow=True で送出していたので、最低限ログは残す。
                    logger.warning("recording input overflow: samples were dropped")
                in_data = bytes(chunk_data)
                interval_frame_count += config.chunk
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
            logger.warning("retry for %e", e)
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
            async for utterance in pyaudio_recording_worker(
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
