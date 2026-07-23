"""streaming VC の独立マイクキャプチャ(ADR-0052)。

発話系 recording は無改変のまま、別途 16k mono でマイクを開いて固定 hop の
float32 ブロックを出す。排他デバイスで二重 open が失敗する環境向けの fan-out
フォールバックは M2 スコープ外(ADR-0052 に設計として残す)。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_input_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.transport import drop_oldest_put

CAPTURE_RATE = 16000


def ms_to_samples(ms: float, rate: int = 16000) -> int:
    """ms を rate のサンプル数へ(round)。"""
    return round(ms * rate / 1000.0)


def pcm16_to_float32(data: bytes) -> NDArray[np.float32]:
    """int16 PCM バイト列を [-1, 1] の float32 にする。"""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream:
    device = resolve_stream_vc_input_device(config)
    logger.info("stream_vc input device %s: %s", device.index, device.name)
    stream = sd.RawInputStream(
        samplerate=CAPTURE_RATE,
        blocksize=hop,
        device=device.index,
        channels=1,
        dtype="int16",
    )
    stream.start()
    return stream


async def capture_loop(config: StreamVcConfig, out_queue: Queue, hop: int) -> None:
    """マイクから hop サンプルずつ読み、float32 ブロックを out_queue へ。"""
    with worker_startup("stream_vc"):
        stream = open_stream_vc_input_stream(config, hop)
    logger.info("stream vc capture started")
    try:
        while stream.active:
            data, overflowed = await to_thread(stream.read, hop)
            if overflowed:
                logger.warning("stream_vc capture input overflow")
            block = pcm16_to_float32(bytes(data))
            if not drop_oldest_put(out_queue, block):
                telemetry.record("stream_vc_capture_drop", 1.0)
                logger.warning("stream_vc capture queue full; dropped oldest block")
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        stream.close()
