"""streaming VC の独立マイクキャプチャ(ADR-0052)。

発話系 recording は無改変のまま、別途 16k mono でマイクを開いて固定 hop の
float32 ブロックを出す。排他デバイスで二重 open が失敗する環境向けの fan-out
フォールバックは未実装(ADR-0052 に設計として残す)。
"""

from asyncio import Event
from asyncio import Queue
from asyncio import to_thread
from enum import Enum

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.lib.audio import resolve_stream_vc_input_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.retry import run_with_device_retry
from vspeech.stream_vc.transport import drop_oldest_put

CAPTURE_RATE = 16000


class CaptureSignal(Enum):
    """capture → runner の帯域内シグナル(音声ブロックに混ぜて capture_queue へ流す番兵)。

    capture と runner(vc_loop)は別タスクなので、capture の device 再 open から
    runner の StreamingVc へ直接触れない。単一メンバの Enum を番兵にすることで、
    runner 側は `block is CaptureSignal.REOPEN` の identity 判定で音声ブロックと
    区別でき、型も `NDArray | CaptureSignal` に正直に絞れる。
    """

    REOPEN = 0  # device 再 open の境界。runner は文脈/VAD ゲートを reset する。


# capture_queue が運ぶ要素型:音声ブロック、または帯域内シグナルの番兵。
type CaptureItem = NDArray[np.float32] | CaptureSignal


def ms_to_samples(ms: float, rate: int = CAPTURE_RATE) -> int:
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
        latency="low",
    )
    stream.start()
    return stream


async def _capture_read_loop(
    stream: sd.RawInputStream, hop: int, out_queue: Queue[CaptureItem]
) -> None:
    """steady-state: device fault が起きるまで hop サンプルずつ読み続ける。

    device loss は stream.read() が (OSError, sd.PortAudioError) を raise する。
    ここでは捕えず run_with_device_retry へ抜けさせ、close→backoff→再 open で
    サブシステム内だけで回復する(兄弟 vc/playback や発話系は巻き込まない,
    ADR-0050)。`while stream.active` だと deactivate が黙って返り get()/recv() で
    待つ兄弟を無言で stall させうるので `while True` にする。
    """
    while True:
        data, overflowed = await to_thread(stream.read, hop)
        if overflowed:
            logger.warning("stream_vc capture input overflow")
        block = pcm16_to_float32(bytes(data))
        if not drop_oldest_put(out_queue, block):
            telemetry.record("stream_vc_capture_drop", 1.0)
            logger.warning("stream_vc capture queue full; dropped oldest block")


async def capture_loop(
    config: StreamVcConfig,
    out_queue: Queue[CaptureItem],
    hop: int,
    ready: Event,
) -> None:
    """マイクから hop サンプルずつ読み、float32 ブロックを out_queue へ。

    初回 open は fail-loud(worker_startup)、以降の runtime device fault は
    自力で再接続する(ADR-0050)。capture 自体は再 open で引き継ぐ状態を持たないが、
    runner(vc_loop, 別タスク)は数秒前の rolling 文脈/クロスフェード tail を抱えた
    ままなので、再 open 境界に CaptureSignal.REOPEN 番兵を capture_queue へ積んで
    runner に文脈 reset を促す(直接触れないので帯域内で知らせる)。fault 時点で
    積むため、番兵は queue 内の「fault 前の stale ブロック」と「再 open 後の fresh
    ブロック」のちょうど境界に入る。満杯でも必ず入るよう drop_oldest_put を使う。
    """

    def _signal_reopen() -> None:
        drop_oldest_put(out_queue, CaptureSignal.REOPEN)

    # VC の warmup 完了まで待ってからマイクを開く。先に開くと、モデルロード中に
    # 実時間で溜まった音声が起動直後にキューへ殺到して drop の嵐になり、
    # 最初の数百 ms が stale な音声で埋まる(実機ログで確認済み)。
    await ready.wait()
    await run_with_device_retry(
        open_stream=lambda: open_stream_vc_input_stream(config, hop),
        run=lambda stream: _capture_read_loop(stream, hop, out_queue),
        worker="stream_vc",
        label="stream vc capture",
        on_reopen=_signal_reopen,
        reopen_metric="stream_vc_capture_reopen",
    )
