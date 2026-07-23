"""streaming VC のローカル連続再生。

transport から受けた変換音声を出力デバイスへ連続 write する。単一マシン内の
in-process 転送では欠落は起きないが、seq の飛びは観測して記録する(受入基準:
欠落を黙って無音の穴にしない)。実際の穴埋め/整列は網トランスポートを入れる
段階の担当。
"""

from asyncio import CancelledError
from asyncio import sleep
from asyncio import to_thread

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_output_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.retry import BACKOFF_START
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.retry import next_backoff
from vspeech.stream_vc.transport import Transport


def detect_gap(prev_seq: int | None, seq: int) -> int:
    """前 seq から見た欠落パケット数(前進の飛びのみ、並べ替え/重複は 0)。"""
    if prev_seq is None:
        return 0
    missing = seq - prev_seq - 1
    return missing if missing > 0 else 0


# 出力 underflow は VC が間に合わなければ毎ブロック (block_ms=160 なら ~6 回/秒)
# 起きうる。telemetry は毎回記録するが、ログは最初の 1 回と以降 N 回ごとに間引く
# — 警告自体が GUI の読む stdout パイプを埋めては本末転倒なので。
UNDERFLOW_LOG_EVERY = 50


def should_log_underflow(count: int) -> bool:
    """通算 count 回目の underflow をログに出すか(1 回目と以降 N 回ごと)。"""
    return count == 1 or count % UNDERFLOW_LOG_EVERY == 0


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
    """transport から受けて連続再生する。最初のパケットで出力ストリームを開く。

    初回 open は fail-loud(worker_startup)。以降 write/再 open で起きる runtime
    device fault は close→backoff→次パケットで lazy 再 open して自力回復する
    (発話系や兄弟タスクは巻き込まない, ADR-0050)。出力の sample_rate はパケットが
    持つので再 open もパケット到着時に行える。
    """
    stream: sd.RawOutputStream | None = None
    prev_seq: int | None = None
    underflow_count = 0
    # 一度でも open に成功したか(初回 fail-loud と runtime 再 open を区別するため)。
    started = False
    backoff = BACKOFF_START
    try:
        while True:
            packet = await transport.recv()
            try:
                if stream is None:
                    if started:
                        # runtime 再 open は fail-loud にしない(backoff 済み)。
                        stream = open_stream_vc_output_stream(
                            config, packet.sample_rate
                        )
                        logger.info("stream vc playback reopened")
                    else:
                        with worker_startup("stream_vc"):
                            stream = open_stream_vc_output_stream(
                                config, packet.sample_rate
                            )
                        started = True
                        logger.info("stream vc playback started")
                    backoff = BACKOFF_START
                gap = detect_gap(prev_seq, packet.seq)
                if gap > 0:
                    telemetry.record("stream_vc_gap", float(gap))
                    logger.warning("stream_vc playback gap: %d packet(s) missing", gap)
                prev_seq = packet.seq
                # write() の戻り値 = paOutputUnderflowed (capture.py の read() の
                # overflowed と対称)。捨てると「無音の穴」が黙って出る — この module が
                # 防ぐと謳っているものそのものなので必ず見る。
                underflowed = await to_thread(stream.write, packet.pcm)
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    underflow_count += 1
                    if should_log_underflow(underflow_count):
                        logger.warning(
                            "stream_vc playback output underflow (total %d)",
                            underflow_count,
                        )
            except (OSError, sd.PortAudioError) as e:
                # runtime device fault: 出力先が消えた/フォーマットが変わった等。
                # サブシステム内で吸収し、次パケットで lazy 再 open する。
                logger.warning(
                    "stream_vc playback device fault; retry for %r (backoff %.1fs)",
                    e,
                    backoff,
                )
                telemetry.record("stream_vc_playback_reopen", 1.0)
                # 再 open 自体が失敗したときは stream が None のままなので guard する。
                if stream is not None:
                    close_quietly(stream)
                stream = None
                await sleep(backoff)
                backoff = next_backoff(backoff)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            stream.close()
