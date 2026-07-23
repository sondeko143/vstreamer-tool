"""ストリーミング VC の consumer 再生ループ(role=consumer, ADR-0055/0056)。

torch/RVC/GPU を一切 import しない(再生専任マシンは変換音声を鳴らすだけ)。
transport.recv → jitter buffer push → poll で残りも push → pop 1 ブロック → 出力 write。
遅延の計測は skew 免疫の量だけ: 到着間隔ジッタと seq gap(片道遅延は clock skew に
汚染されるので測らない, ADR-0056)。出力デバイス障害は playback.py と同じく
self-heal(次パケットで lazy 再 open)。
"""

from __future__ import annotations

from asyncio import CancelledError
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
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.transport import Transport


def consume_into_buffer(
    transport: Transport, buffer: JitterBuffer, first: StreamPacket
) -> None:
    """recv した first と poll した残り全部を jitter buffer へ push する。"""
    buffer.push(first)
    for packet in transport.poll():
        buffer.push(packet)


async def network_playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    target_depth = round(config.jitter_buffer_ms / config.block_ms)
    buffer = JitterBuffer(target_depth=target_depth)
    logger.info("stream_vc consumer jitter buffer depth: %d block(s)", target_depth)
    stream: sd.RawOutputStream | None = None
    session: str | None = None
    prev_recv: float | None = None
    started = False
    underflow_count = 0
    gap_count = 0
    try:
        while True:
            packet = await transport.recv()
            now = perf_counter()
            if prev_recv is not None:
                telemetry.record("stream_vc_interarrival", now - prev_recv)
            prev_recv = now
            if packet.session_id != session:
                if session is not None:
                    logger.info("stream_vc consumer: producer session changed; reset")
                session = packet.session_id
                buffer.reset()
            consume_into_buffer(transport, buffer, packet)
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
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            close_quietly(stream)
