"""The independent mic capture of streaming VC (ADR-0052).

Leaves the utterance-path recording untouched and opens the mic separately in mono,
emitting float32 blocks of a fixed hop at CAPTURE_RATE. The device is opened at its own
native rate and the conversion down to CAPTURE_RATE happens here, in process
(ADR-0070/0071) -- asking the device for 16 kHz would hand the conversion to the OS,
whose filter we can neither test nor log, and WASAPI shared mode refuses any rate but
its mix format. The fan-out fallback for environments where an exclusive device rejects
a second open is unimplemented (it remains a design in ADR-0052).
"""

from asyncio import Event
from asyncio import Queue
from asyncio import to_thread
from enum import Enum

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.audio import resolve_stream_vc_input_device
from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.retry import run_with_device_retry
from vspeech.stream_vc.transport import drop_oldest_put

CAPTURE_RATE = 16000


class CaptureSignal(Enum):
    """In-band signals from capture to the runner (sentinels pushed onto capture_queue
    among the audio blocks).

    capture and the runner (vc_loop) are separate tasks, so capture's device reopen
    cannot touch the runner's StreamingVc directly. Using a single-member Enum as the
    sentinel lets the runner tell it apart from an audio block by identity
    (`block is CaptureSignal.REOPEN`) and lets the type narrow honestly to
    `NDArray | CaptureSignal`.
    """

    # The boundary of a device reopen. The runner resets its context and VAD gate.
    REOPEN = 0


# The element type capture_queue carries: an audio block, or an in-band signal sentinel.
type CaptureItem = NDArray[np.float32] | CaptureSignal


def ms_to_samples(ms: float, rate: int = CAPTURE_RATE) -> int:
    """Convert ms into a sample count at rate (rounded)."""
    return round(ms * rate / 1000.0)


def pcm16_to_float32(data: bytes) -> NDArray[np.float32]:
    """Convert int16 PCM bytes into float32 in [-1, 1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def device_frames_per_read(hop: int, device_rate: int) -> int:
    """How many device frames make up one `hop`-sample block at CAPTURE_RATE.

    One read is one block's worth of time, and the polyphase resampler is causal (it
    holds nothing back), so one read yields one block -- the cadence the queue had back
    when the device itself ran at CAPTURE_RATE. That is why nothing here pre-fills the
    accumulator: priming is what a resampler with internal latency would need, and doing
    it anyway would delay every block by a whole hop (ADR-0070).
    """
    if device_rate == CAPTURE_RATE:
        return hop
    return max(1, round(hop * device_rate / CAPTURE_RATE))


class InputRateConverter:
    """Turns one device read into whole `hop`-sample blocks at CAPTURE_RATE.

    Holds the two pieces of state a device reopen must discard together: the polyphase
    filter tail and the samples left over from the previous read. `_capture_read_loop`
    builds one per open, so a reopen cannot carry either into the new stream.

    A device that already runs at CAPTURE_RATE builds no resampler and passes the read
    straight through, which keeps that path bit-identical to the pre-ADR-0070 code.
    """

    def __init__(self, device_rate: int, hop: int) -> None:
        self.hop = hop
        self.resampler: PolyphaseResampler | None = make_resampler(
            device_rate, CAPTURE_RATE
        )
        self.frames_per_read = device_frames_per_read(hop, device_rate)
        self._pending: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    def blocks(self, samples: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """The whole blocks this read completes -- one per read at the rates that
        matter, occasionally two.

        Whatever is left over past the last whole block stays here and starts the next
        one. `hop * device_rate / CAPTURE_RATE` is a whole number of device frames at
        every rate the pipeline actually meets (48000 and 44100 both divide evenly at a
        10 ms-multiple hop), and there one read is exactly one block. When it is not --
        44100 with a 100-sample hop is 100.14 output samples per read -- the surplus
        accumulates here and a read emits two blocks each time it crosses a hop
        boundary, roughly once in 735 reads. What must never happen either way is a
        dropped sample or a short block: the runner's gate and envelope assume a fixed
        block length.
        """
        if self.resampler is None:
            return [samples]
        converted = self.resampler.process(samples)
        if self._pending.size:
            converted = np.concatenate([self._pending, converted])
        whole = converted.size - converted.size % self.hop
        # Copies, not views: the blocks outlive `converted` on the queue, and keeping
        # the leftover a view would pin the whole buffer behind it.
        self._pending = converted[whole:].copy()
        return [converted[i : i + self.hop].copy() for i in range(0, whole, self.hop)]


def open_stream_vc_input_stream(
    config: StreamVcConfig, hop: int
) -> tuple[sd.RawInputStream, int]:
    """Open the mic at its native rate; return the stream and that rate.

    Rate resolution sits next to the device resolution that was already here, so an open
    stays a single decision point and the rate has no second, cached copy to drift from.
    It does not re-decide anything within one process: sd.query_devices() is cached at
    PortAudio init and nothing here re-initialises it (see the deferred note in
    worker/playback.py's search_appropriate_device), so a reopen sees the same table and
    resolves the same rate.

    The caller needs the rate back to build the converter, and run_with_device_retry
    hands whatever `open_stream` returns straight to `close_quietly()`, so the pair is
    unpacked by capture_loop's opener rather than returned from it.

    Both resolvers raise the DeviceNotFoundError family, which is deliberately **not** in
    retry.py's DEVICE_ERRORS: on a reopen it escapes run_with_device_retry and ends the
    subsystem instead of backing off. That is the behaviour we want. Backing off cannot
    help -- an unresolvable rate stays unresolvable however long you wait, so retrying
    would spin forever on a config problem -- and ADR-0050 wants an unrecoverable fault
    in an explicitly enabled feature to fail loud for the supervisor. It also keeps the
    two resolvers consistent: the device lookup has always failed this way. (Unreachable
    today for the reason above; this is a statement of intent, not a live path.)
    """
    device = resolve_stream_vc_input_device(config)
    rate, how = resolve_device_rate(
        device,
        config.input_device_rate,
        input=True,
        config_key="stream_vc.input_device_rate",
    )
    # Logged before the open so a failing open still says what was attempted.
    logger.info(
        "stream_vc input device %s: %s @%dHz (%s) -> %dHz (%s)",
        device.index,
        device.name,
        rate,
        how,
        CAPTURE_RATE,
        "プロセス内で変換" if rate != CAPTURE_RATE else "変換なし",
    )
    stream = sd.RawInputStream(
        samplerate=rate,
        blocksize=device_frames_per_read(hop, rate),
        device=device.index,
        channels=1,
        dtype="int16",
        latency="low",
    )
    stream.start()
    # PortAudio may know the endpoint runs at a slightly different rate than the one it
    # accepted. We keep converting at the requested rate (the L/M ratio has to be built
    # from a sane number: 44099 -> 16000 would mean 16000 phases), so a delta shows up
    # only as a slow drift in the audio -- invisible unless it is said out loud here.
    reported = float(stream.samplerate)
    if abs(reported - rate) > 0.5:
        logger.warning(
            "stream_vc capture device reports %.4fHz for a requested %dHz; "
            "converting at the requested rate",
            reported,
            rate,
        )
    return stream, rate


def _put_block(
    out_queue: Queue[CaptureItem],
    block: NDArray[np.float32],
    running: Event,
    drop_throttle: LogThrottle,
) -> None:
    """Put one block on the queue, attributing a drop to the right stage."""
    if drop_oldest_put(out_queue, block):
        return
    if not running.is_set():
        # While paused vc_loop stops consuming, so the queue stays full and every
        # subsequent block is dropped. That is exactly the behaviour ADR-0050 intended
        # (do not accumulate paused audio) and not an anomaly, so no warning. Warning
        # every time would emit about 6 lines a second at block_ms=160 for the whole
        # pause and make the warning meaningless. They are still not discarded silently:
        # they are counted under a pause-specific stage -- mixing them into the same
        # stage would pollute the backpressure metric (stream_vc_capture_drop, used to
        # assess RTF) with the length of the pause.
        telemetry.record("stream_vc_capture_drop_paused", 1.0)
        return
    telemetry.record("stream_vc_capture_drop", 1.0)
    if (n := drop_throttle.hit()) is not None:
        logger.warning(
            "stream_vc capture queue full; dropped oldest block (total %d)",
            n,
        )


async def _capture_read_loop(
    stream: sd.RawInputStream,
    hop: int,
    out_queue: Queue[CaptureItem],
    running: Event,
    device_rate: int,
) -> None:
    """Steady state: keep reading one block's worth of device frames until a fault.

    Device loss surfaces as stream.read() raising (OSError, sd.PortAudioError). It is not
    caught here; it escapes to run_with_device_retry, which recovers within the subsystem
    via close -> backoff -> reopen (without dragging in the sibling vc/playback tasks or
    the utterance path, ADR-0050). `while stream.active` would return silently on
    deactivate and could stall siblings waiting in get()/recv() without a word, so this
    is `while True`.

    `running` is the pause/resume gate shared with the utterance path
    (`context.running`). Capture is **not** stopped by it -- ADR-0050 decided that
    capture keeps running while paused and drop_oldest_put discards the backlog; it is
    consulted here only to avoid misreporting those drops as an anomaly.

    `device_rate` is the rate the stream was opened at, not `stream.samplerate`:
    PortAudio may report a hardware rate that differs by a hair, and an off-by-one rate
    turns a small L/M ratio into a gigantic one (16000 phases for 44099 -> 16000). We
    asked for a rate the device accepted, so that is the ratio to filter with; when the
    two disagree open_stream_vc_input_stream says so in a warning.
    """
    # A drop while running = real backpressure. Throttle by time (ADR-0062).
    drop_throttle = LogThrottle()
    # An input overflow means the reader was late, which persists once it starts, so this
    # fires on every block (about 6 a second at block_ms=160) until it clears. Thin it by
    # time and meter it every occurrence -- exactly what its counterpart on the sink side
    # (playback.py's paOutputUnderflowed) already does.
    overflow_throttle = LogThrottle()
    # Built here rather than in capture_loop: run_with_device_retry calls run(stream)
    # afresh after every reopen, so this coroutine's lifetime IS one stream's lifetime,
    # and the filter tail plus the half-filled block die with it.
    converter = InputRateConverter(device_rate, hop)
    while True:
        data, overflowed = await to_thread(stream.read, converter.frames_per_read)
        if overflowed:
            telemetry.record("stream_vc_capture_overflow", 1.0)
            if (n := overflow_throttle.hit()) is not None:
                logger.warning("stream_vc capture input overflow (total %d)", n)
        for block in converter.blocks(pcm16_to_float32(bytes(data))):
            _put_block(out_queue, block, running, drop_throttle)


async def capture_loop(
    config: StreamVcConfig,
    out_queue: Queue[CaptureItem],
    hop: int,
    ready: Event,
    running: Event,
) -> None:
    """Read one block's worth of mic audio at a time and push hop-sample float32 blocks
    at CAPTURE_RATE to out_queue.

    The first open is fail-loud (worker_startup); runtime device faults after that
    reconnect on their own (ADR-0050). Capture's own state (the resampler's filter tail
    and the partial block) dies with the read loop, so nothing of it crosses a reopen,
    but the runner (vc_loop, a separate task) is still holding a rolling context and
    crossfade tail from seconds ago, so a CaptureSignal.REOPEN sentinel is pushed onto
    capture_queue at the reopen boundary to prompt the runner to reset its context (it
    cannot be touched directly, hence the in-band signal). Pushing it at fault time puts
    the sentinel exactly at the boundary between the "stale pre-fault blocks" and the
    "fresh post-reopen blocks" in the queue. drop_oldest_put is used so it always gets
    in, even when the queue is full.

    `running` (the pause/resume gate shared with the utterance path) is not used to stop
    capture; it is consulted so that drops during a pause are not warned about as an
    anomaly (see _capture_read_loop).
    """

    def _signal_reopen() -> None:
        drop_oldest_put(out_queue, CaptureSignal.REOPEN)

    # The rate the mic is currently open at. It travels from the opener to the read loop
    # through this closure because run_with_device_retry is bound to `[T: _Closable]` and
    # passes open_stream's return value to close_quietly(), so the opener can only return
    # the stream itself. Seeded with 0, not CAPTURE_RATE: run_with_device_retry only
    # calls _read after _open has returned, and 0 turns a hypothetical read-before-open
    # into an immediate ValueError out of make_resampler rather than audio quietly
    # passing through at the wrong rate.
    device_rate = 0

    def _open() -> sd.RawInputStream:
        nonlocal device_rate
        stream, device_rate = open_stream_vc_input_stream(config, hop)
        return stream

    async def _read(stream: sd.RawInputStream) -> None:
        # device_rate is read when run_with_device_retry calls this, which is always
        # after the (re)open above -- so the loop always filters with the rate the
        # stream in its hands was opened at.
        await _capture_read_loop(stream, hop, out_queue, running, device_rate)

    # Wait for the VC warmup to finish before opening the mic. Opening earlier lets the
    # audio that accumulated in real time during model loading flood the queue right
    # after startup, causing a storm of drops and filling the first few hundred ms with
    # stale audio (confirmed in the logs on real hardware).
    await ready.wait()
    await run_with_device_retry(
        open_stream=_open,
        run=_read,
        worker="stream_vc",
        label="stream vc capture",
        on_reopen=_signal_reopen,
        reopen_metric="stream_vc_capture_reopen",
    )
