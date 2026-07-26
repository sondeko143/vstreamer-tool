"""Local continuous playback for streaming VC.

Continuously writes the converted audio received from the transport to the output
device. The producer bursts faster than real time (RTF<1) while the consumer consumes on
the output device clock, so a backlog built up by a transient stall sticks around as
permanent delay. To prevent that, `drain_to_latest` folds the already-arrived backlog
forward to the newest before playing, and whatever was discarded is always observed
through telemetry and the seq (gap) bookkeeping -- this module promises never to turn a
loss into a silent hole, so even deliberate drops are recorded (nothing is dropped
silently). Seq jumps are recorded the same way (an acceptance criterion). Actual
concealment and reordering belong to the stage that introduces the network transport.
"""

from asyncio import CancelledError
from asyncio import sleep
from asyncio import to_thread

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_output_device
from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.retry import BACKOFF_START
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.retry import next_backoff
from vspeech.stream_vc.transport import Transport


def detect_gap(prev_seq: int | None, seq: int) -> int:
    """Packets missing relative to the previous seq (forward jumps only; reordering and
    duplicates give 0)."""
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
    """Receive from the transport and play continuously. The output stream is opened on
    the first packet.

    The first open is fail-loud (worker_startup). Runtime device faults from later
    writes/reopens self-heal via close -> backoff -> lazy reopen on the next packet
    (without dragging in the utterance path or sibling tasks, ADR-0050). The output
    sample_rate travels with the packet, so a reopen can also happen on packet arrival.
    """
    stream: sd.RawOutputStream | None = None
    prev_seq: int | None = None
    # One per condition. Output underflows, stale drops and seq gaps all fire on every
    # block once they start (about 6 times a second at block_ms=160), so throttle by time
    # to keep the warnings themselves from burying the log. Telemetry is recorded every
    # time regardless of the thinning (ADR-0062).
    underflow_throttle = LogThrottle()
    drop_throttle = LogThrottle()
    gap_throttle = LogThrottle()
    # Whether an open has ever succeeded (to tell the fail-loud first open apart from a
    # runtime reopen).
    started = False
    backoff = BACKOFF_START
    try:
        while True:
            packet = await transport.recv()
            # Fold the already-arrived backlog forward to the newest so latency does not
            # stick. When there is a backlog, discard the recv'd packet (the oldest) too
            # and take the newest that drain left behind to play (keep=1). Everything
            # discarded goes through the seq bookkeeping so it is always observed.
            stale = transport.drain_to_latest(keep=1)
            if stale:
                for old in (packet, *stale):
                    gap = detect_gap(prev_seq, old.seq)
                    if gap > 0:
                        telemetry.record("stream_vc_gap", float(gap))
                        if (n := gap_throttle.hit()) is not None:
                            logger.warning(
                                "stream_vc playback gap: %d packet(s) missing "
                                "(total %d)",
                                gap,
                                n,
                            )
                    prev_seq = old.seq
                    telemetry.record("stream_vc_playback_drop", 1.0)
                    if (n := drop_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback dropped stale packet(s) to bound "
                            "latency (total %d)",
                            n,
                        )
                # Re-take the newest that drain left behind (one item in the queue);
                # this does not block.
                packet = await transport.recv()
            try:
                if stream is None:
                    if started:
                        # A runtime reopen is not made fail-loud (it is already backed
                        # off).
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
                    if (n := gap_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback gap: %d packet(s) missing (total %d)",
                            gap,
                            n,
                        )
                prev_seq = packet.seq
                # write()'s return value = paOutputUnderflowed (symmetric with read()'s
                # overflowed in capture.py). Discarding it would let a "silent hole" slip
                # out -- exactly what this module claims to prevent, so always look at it.
                underflowed = await to_thread(stream.write, packet.pcm)
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    if (n := underflow_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback output underflow (total %d)", n
                        )
            except (OSError, sd.PortAudioError) as e:
                # A runtime device fault: the output sink disappeared, the format
                # changed, etc. Absorb it inside the subsystem and lazily reopen on the
                # next packet.
                logger.warning(
                    "stream_vc playback device fault; retry for %r (backoff %.1fs)",
                    e,
                    backoff,
                )
                telemetry.record("stream_vc_playback_reopen", 1.0)
                # When the reopen itself failed stream is still None, so guard it.
                if stream is not None:
                    close_quietly(stream)
                stream = None
                await sleep(backoff)
                backoff = next_backoff(backoff)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        # Wrapped in close_quietly: if the output device is faulted during a Ctrl-C, a
        # raw close() raises sd.PortAudioError, which escapes the finally and replaces
        # the WorkerShutdown that was in flight (turning a clean cancel into an error
        # exit).
        if stream is not None:
            close_quietly(stream)
