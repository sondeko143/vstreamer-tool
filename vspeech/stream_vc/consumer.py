"""The consumer playback loop of streaming VC (role=consumer, ADR-0055/0056).

Imports nothing from torch/RVC/GPU (a playback-only machine merely sounds the converted
audio). transport.recv -> push into the jitter buffer -> poll and push the rest -> pop
one block -> write to the output. Only skew-immune quantities are measured: interarrival
jitter and seq gaps (one-way delay is contaminated by clock skew and is deliberately not
measured, ADR-0056). Output device faults self-heal exactly as in playback.py (lazy
reopen on the next packet).
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
from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.playback import open_stream_vc_output_stream
from vspeech.stream_vc.retry import BACKOFF_START
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.retry import next_backoff
from vspeech.stream_vc.transport import Transport


def consume_into_buffer(
    transport: Transport, buffer: JitterBuffer, first: StreamPacket, session: str
) -> None:
    """Push the recv'd `first` plus the polled packets of the current session into the
    jitter buffer.

    The polled ones are filtered by session_id: right after a producer restart the socket
    queue can still hold high seqs from the old session, and pushing those would make the
    overflow fast-forward jump the cursor to an old seq and drop the new session as late
    (permanent silence). late/dup (push returning False) is recorded to observe
    reordering."""
    if not buffer.push(first):
        telemetry.record("stream_vc_reorder_drop", 1.0)
    for packet in transport.poll():
        if packet.session_id != session:
            telemetry.record("stream_vc_session_skip", 1.0)
            continue
        if not buffer.push(packet):
            telemetry.record("stream_vc_reorder_drop", 1.0)


async def network_playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    # Deliberately has no context.running (pause) gate: the consumer does not run vc_loop,
    # it only sounds the converted audio. A global pause is achieved by stopping the
    # producer (the producer's vc_loop stops sending -> the consumer starves and goes
    # silent). Following ADR-0050's single-check model, the pause decision lives in
    # exactly one place, on the producer.
    target_depth = round(config.jitter_buffer_ms / config.block_ms)
    buffer = JitterBuffer(target_depth=target_depth)
    logger.info("stream_vc consumer jitter buffer depth: %d block(s)", target_depth)
    stream: sd.RawOutputStream | None = None
    session: str | None = None
    prev_recv: float | None = None
    started = False
    # Time-based thinning for the same reason as playback.py (ADR-0062).
    underflow_throttle = LogThrottle()
    gap_throttle = LogThrottle()
    backoff = BACKOFF_START
    try:
        while True:
            # Concealment is recv-driven: it only fires on gaps within a stream that is
            # arriving. If the network stops entirely we block in recv() and nothing is
            # concealed here -- the output device underflows on its own and that is
            # recorded on the next successful write. Acceptable for M3, which assumes a
            # wired LAN (ADR-0056, measure first). Revisit with an output-clock-driven
            # pacer if a lossy link is ever required.
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
            # The block above always makes session equal packet.session_id (== the current
            # session). Passing packet.session_id pins the type to str (session is
            # str | None).
            consume_into_buffer(transport, buffer, packet, packet.session_id)
            result = buffer.pop()
            telemetry.record("stream_vc_jitter_buffer_depth", float(buffer.depth))
            if result.kind is PopKind.CONCEAL:
                telemetry.record("stream_vc_conceal", 1.0)
            if result.gap:
                telemetry.record("stream_vc_gap", float(result.gap))
                if (n := gap_throttle.hit()) is not None:
                    logger.warning(
                        "stream_vc consumer gap: %d packet(s) missing (total %d)",
                        result.gap,
                        n,
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
                    if (n := underflow_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc consumer output underflow (total %d)", n
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
