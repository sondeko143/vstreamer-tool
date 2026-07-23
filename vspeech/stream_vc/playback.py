"""streaming VC のローカル連続再生。

transport から受けた変換音声を出力デバイスへ連続 write する。M2 は単一マシン
in-process なので欠落は起きないが、seq の飛びは観測して記録する(受入基準:
欠落を黙って無音の穴にしない)。実際の穴埋め/整列は網が絡む M3 の担当。
"""

from asyncio import CancelledError
from asyncio import to_thread

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_output_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.transport import Transport


def detect_gap(prev_seq: int | None, seq: int) -> int:
    """前 seq から見た欠落パケット数(前進の飛びのみ、並べ替え/重複は 0)。"""
    if prev_seq is None:
        return 0
    missing = seq - prev_seq - 1
    return missing if missing > 0 else 0


def open_stream_vc_output_stream(
    config: StreamVcConfig, sample_rate: int
) -> sd.RawOutputStream:
    device = resolve_stream_vc_output_device(config)
    logger.info("stream_vc output device %s: %s", device.index, device.name)
    stream = sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        device=device.index,
        dtype="int16",
        latency="low",
    )
    stream.start()
    return stream


async def playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    """transport から受けて連続再生する。最初のパケットで出力ストリームを開く。"""
    stream: sd.RawOutputStream | None = None
    prev_seq: int | None = None
    try:
        while True:
            packet = await transport.recv()
            if stream is None:
                with worker_startup("stream_vc"):
                    stream = open_stream_vc_output_stream(config, packet.sample_rate)
                logger.info("stream vc playback started")
            gap = detect_gap(prev_seq, packet.seq)
            if gap > 0:
                telemetry.record("stream_vc_gap", float(gap))
                logger.warning("stream_vc playback gap: %d packet(s) missing", gap)
            prev_seq = packet.seq
            await to_thread(stream.write, packet.pcm)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            stream.close()
