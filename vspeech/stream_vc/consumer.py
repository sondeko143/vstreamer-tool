"""ストリーミング VC の consumer 再生ループ(role=consumer, ADR-0055/0056)。

torch/RVC/GPU を一切 import しない(再生専任マシンは変換音声を鳴らすだけ)。
transport.recv → jitter buffer push → poll で残りも push → pop 1 ブロック → 出力 write。
遅延の計測は skew 免疫の量だけ: 到着間隔ジッタと seq gap(片道遅延は clock skew に
汚染されるので測らない, ADR-0056)。出力デバイス障害は playback.py と同じく
self-heal(次パケットで lazy 再 open)。
"""

from __future__ import annotations

from asyncio import CancelledError
from asyncio import sleep
from asyncio import to_thread
from time import perf_counter

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.playback import open_stream_vc_output_stream
from vspeech.stream_vc.playback import should_log_gap
from vspeech.stream_vc.playback import should_log_underflow
from vspeech.stream_vc.retry import BACKOFF_START
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.retry import next_backoff
from vspeech.stream_vc.transport import Transport


def consume_into_buffer(
    transport: Transport, buffer: JitterBuffer, first: StreamPacket, session: str
) -> None:
    """recv した first と、poll した現セッション packet を jitter buffer へ push する。

    poll 分は session_id で filter する: producer 再起動直後は旧セッションの高 seq が
    socket queue に残りうるが、それを push すると overflow fast-forward が cursor を
    旧 seq へ飛ばし新セッションを late 落ちさせる(永久無音)。late/dup(push False)は
    reorder の観測用に記録する。"""
    if not buffer.push(first):
        telemetry.record("stream_vc_reorder_drop", 1.0)
    for packet in transport.poll():
        if packet.session_id != session:
            telemetry.record("stream_vc_session_skip", 1.0)
            continue
        if not buffer.push(packet):
            telemetry.record("stream_vc_reorder_drop", 1.0)


async def network_playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    # 意図的に context.running (pause) gate を持たない: consumer は vc_loop を回さず
    # 変換音声を鳴らすだけ。全体 pause は producer 側を止めることで達成する(producer の
    # vc_loop が送信を止める → consumer は starve して無音になる)。ADR-0050 の single-check
    # モデルに従い、pause 判定は producer 一箇所だけに置く。
    target_depth = round(config.jitter_buffer_ms / config.block_ms)
    buffer = JitterBuffer(target_depth=target_depth)
    logger.info("stream_vc consumer jitter buffer depth: %d block(s)", target_depth)
    stream: sd.RawOutputStream | None = None
    session: str | None = None
    prev_recv: float | None = None
    started = False
    underflow_count = 0
    gap_count = 0
    backoff = BACKOFF_START
    try:
        while True:
            # concealment は recv 駆動: 到着しているストリーム内の gap にだけ発火する。
            # ネットワークが完全に停止すると recv() でブロックし、ここでは conceal されない
            # — 出力デバイスが自力で underflow し、次の成功 write で記録される。有線 LAN
            # 前提の M3 では許容(ADR-0056 measure-first)。lossy な回線が要るなら出力クロック
            # 駆動の pacer で再訪する。
            packet = await transport.recv()
            now = perf_counter()
            if prev_recv is not None:
                telemetry.record("stream_vc_interarrival", now - prev_recv)
            prev_recv = now
            if packet.session_id != session:
                if session is not None:
                    logger.info("stream_vc consumer: producer session changed; reset")
                    if stream is not None:
                        # a new session may use a different target_sample_rate; drop the
                        # stream so it reopens at the incoming packet's rate.
                        close_quietly(stream)
                        stream = None
                session = packet.session_id
                buffer.reset()
            # session は上のブロックで必ず packet.session_id に揃う(== 現セッション)。
            # packet.session_id を渡すことで型を str に確定させる(session は str | None)。
            consume_into_buffer(transport, buffer, packet, packet.session_id)
            result = buffer.pop()
            telemetry.record("stream_vc_jitter_buffer_depth", float(buffer.depth))
            if result.kind is PopKind.CONCEAL:
                telemetry.record("stream_vc_conceal", 1.0)
            if result.gap:
                telemetry.record("stream_vc_gap", float(result.gap))
                gap_count += 1
                if should_log_gap(gap_count):
                    logger.warning(
                        "stream_vc consumer gap: %d packet(s) missing (total %d)",
                        result.gap,
                        gap_count,
                    )
            if result.dropped:
                telemetry.record("stream_vc_playback_drop", float(result.dropped))
            try:
                if stream is None:
                    if started:
                        stream = open_stream_vc_output_stream(
                            config, packet.sample_rate
                        )
                        logger.info("stream vc consumer playback reopened")
                    else:
                        with worker_startup("stream_vc"):
                            stream = open_stream_vc_output_stream(
                                config, packet.sample_rate
                            )
                        started = True
                        logger.info("stream vc consumer playback started")
                    backoff = BACKOFF_START
                underflowed = await to_thread(stream.write, result.pcm)
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    underflow_count += 1
                    if should_log_underflow(underflow_count):
                        logger.warning(
                            "stream_vc consumer output underflow (total %d)",
                            underflow_count,
                        )
            except (OSError, sd.PortAudioError) as e:
                logger.warning("stream_vc consumer output fault; reopen: %r", e)
                telemetry.record("stream_vc_playback_reopen", 1.0)
                if stream is not None:
                    close_quietly(stream)
                stream = None
                await sleep(backoff)
                backoff = next_backoff(backoff)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            close_quietly(stream)
