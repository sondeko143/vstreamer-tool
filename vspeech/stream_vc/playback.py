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

The device is opened at its own native rate and the packet's rate (the sender's RVC model
rate) is converted to it here, in process (ADR-0070/0071). Opening at the packet's rate
instead would hand the conversion to the OS, whose filter we can neither test nor log,
and WASAPI shared mode refuses any rate but its mix format. `OutputSink` below is the
whole of that; both this module's loop and the consumer's (consumer.py) go through it, so
the two roles cannot drift apart.
"""

from asyncio import CancelledError
from asyncio import sleep
from asyncio import to_thread

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.audio import resolve_stream_vc_output_device
from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.packet import PACKET_CHANNELS
from vspeech.stream_vc.packet import PACKET_FORMAT
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


class OutputSink:
    """An open output stream together with the resampler that feeds it (ADR-0070).

    The two are created and discarded as a unit so a resampler can never be paired with a
    rate its stream was not opened at -- the device rate is fixed for the life of the
    stream, while the packet rate arrives with every packet and may change under us.

    The filter state is kept **across packets**: consecutive packets are one continuous
    signal, and resetting (or rebuilding) per packet would put a transient at every block
    boundary. It is dropped only where the signal really is discontinuous -- `reset()`,
    which the consumer calls when the sender starts a new session.
    """

    def __init__(self, stream: sd.RawOutputStream, device_rate: int) -> None:
        self.stream = stream
        self.device_rate = device_rate
        self._src_rate: int | None = None
        self._resampler: PolyphaseResampler | None = None

    def close(self) -> None:
        """Close the device. Kept `_Closable`-shaped so `close_quietly` takes the sink
        itself and the stream never has to be unwrapped at a call site."""
        self.stream.close()

    def reset(self) -> None:
        """Drop the filter tail **without touching the device**.

        A new sender session is a cut in the audio, not in the device: carrying the tail
        across would smear the end of the old session into the start of the new one, while
        reopening the stream would throw away whatever PortAudio still holds and cost a
        device open on every producer restart. The device rate no sender can change, so
        there is nothing else to rebuild here.
        """
        if self._resampler is not None:
            self._resampler.reset()

    def convert(self, pcm: bytes, src_rate: int) -> bytes:
        """`pcm` (int16 mono at `src_rate`) as int16 mono at the device rate.

        Returns the input object untouched when the rates already match, which keeps that
        path bit-identical to the pre-ADR-0070 code.

        The resampler is rebuilt whenever `src_rate` changes -- the sender's model rate
        travels with every packet, so keying the rebuild on the rate rather than on the
        session id means the ratio cannot silently disagree with the audio, whatever the
        session bookkeeping does.

        A rebuild is a rare event (a producer restart with another model) and has to stay
        one -- not for its cost, which is 0.2-8 ms across every rate pair this boundary
        can now meet (measured: 16000->48000 0.26 ms, 48000->44100 2.5 ms, 11025->48000
        7.7 ms; the pathological coprime pairs that used to dominate this number are
        refused by wire.py now), but because a fresh resampler starts from a zeroed filter
        tail. Rebuilding per packet would fade in the first taps of every block.
        """
        if src_rate != self._src_rate:
            # Built before either field moves: make_resampler rejects a non-positive
            # rate, and recording the new rate first would leave a sink that answers
            # "already at that rate" and passes the next packet through unconverted.
            resampler = make_resampler(src_rate, self.device_rate)
            self._resampler = resampler
            self._src_rate = src_rate
            logger.info(
                "stream_vc output %dHz -> %dHz (%s)",
                src_rate,
                self.device_rate,
                "変換なし" if self._resampler is None else "プロセス内で変換",
            )
        if self._resampler is None:
            return pcm
        samples = decode_pcm(pcm, PACKET_FORMAT, PACKET_CHANNELS)
        # encode_pcm saturates: resampling overshoots the original peak (Gibbs), and a
        # wrapping cast would turn that overshoot into a sign flip = an audible click.
        return encode_pcm(self._resampler.process(samples), PACKET_FORMAT)

    def write(self, pcm: bytes, src_rate: int) -> bool:
        """Convert one packet and write it. Returns paOutputUnderflowed.

        Called through `to_thread`, so the conversion runs off the event loop alongside
        the blocking write it belongs to. No lock guards the filter state: the loops await
        this call, so nothing else touches the sink while a write is in flight.
        """
        return self.stream.write(self.convert(pcm, src_rate))


def open_stream_vc_output(config: StreamVcConfig) -> OutputSink:
    """Open the output device at its own rate and return it paired with its converter.

    Rate resolution sits next to the device resolution that was already here, so an open
    stays a single decision point and the rate has no second, cached copy to drift from
    (the same shape as capture.py's opener). Within one process it re-decides nothing:
    sd.query_devices() is cached at PortAudio init and nothing here re-initialises it, so
    a reopen sees the same table and resolves the same rate.

    Both resolvers raise the DeviceNotFoundError family, which is deliberately in neither
    the callers' `(OSError, sd.PortAudioError)` handler nor retry.py's DEVICE_ERRORS: it
    escapes the reopen path and ends the subsystem rather than backing off forever.
    Backing off cannot help -- an unresolvable rate stays unresolvable however long you
    wait -- and ADR-0050 wants an unrecoverable fault in an explicitly enabled feature to
    fail loud for the supervisor.
    """
    device = resolve_stream_vc_output_device(config)
    rate, how = resolve_device_rate(
        device,
        config.output_device_rate,
        input=False,
        config_key="stream_vc.output_device_rate",
    )
    # Logged before the open so a failing open still says what was attempted.
    logger.info(
        "stream_vc output device %s: %s @%dHz (%s)",
        device.index,
        device.name,
        rate,
        how,
    )
    stream = sd.RawOutputStream(
        samplerate=rate,
        channels=1,
        device=device.index,
        dtype="int16",
        latency="low",
    )
    stream.start()
    # PortAudio may know the endpoint runs at a slightly different rate than the one it
    # accepted. We keep converting at the requested rate (the L/M ratio has to be built
    # from a sane number: 44099 -> 48000 would mean 48000 phases), so a delta shows up
    # only as a slow drift in the audio -- invisible unless it is said out loud here.
    reported = float(stream.samplerate)
    if abs(reported - rate) > 0.5:
        logger.warning(
            "stream_vc playback device reports %.4fHz for a requested %dHz; "
            "converting at the requested rate",
            reported,
            rate,
        )
    return OutputSink(stream, rate)


async def playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    """Receive from the transport and play continuously. The output stream is opened on
    the first packet.

    The first open is fail-loud (worker_startup). Runtime device faults from later
    writes/reopens self-heal via close -> backoff -> lazy reopen on the next packet
    (without dragging in the utterance path or sibling tasks, ADR-0050). A device fault is
    now the only thing that reopens: the stream is opened at the device's own rate, so the
    sample_rate travelling with the packet is converted into it instead (ADR-0070).
    """
    sink: OutputSink | None = None
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
                if sink is None:
                    if started:
                        # A runtime reopen is not made fail-loud (it is already backed
                        # off).
                        sink = open_stream_vc_output(config)
                        logger.info("stream vc playback reopened")
                    else:
                        with worker_startup("stream_vc"):
                            sink = open_stream_vc_output(config)
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
                underflowed = await to_thread(
                    sink.write, packet.pcm, packet.sample_rate
                )
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
                # When the reopen itself failed sink is still None, so guard it.
                if sink is not None:
                    close_quietly(sink)
                sink = None
                await sleep(backoff)
                backoff = next_backoff(backoff)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        # Wrapped in close_quietly: if the output device is faulted during a Ctrl-C, a
        # raw close() raises sd.PortAudioError, which escapes the finally and replaces
        # the WorkerShutdown that was in flight (turning a clean cancel into an error
        # exit).
        if sink is not None:
            close_quietly(sink)
