"""The independent mic capture of streaming VC (ADR-0052).

Leaves the utterance-path recording untouched and opens the mic separately at 16k mono,
emitting float32 blocks of a fixed hop. The fan-out fallback for environments where an
exclusive device rejects a second open is unimplemented (it remains a design in
ADR-0052).
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
from vspeech.lib.log_throttle import LogThrottle
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


def open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream:
    device = resolve_stream_vc_input_device(config)
    # Logged before the open so a failing open still says which device was attempted.
    logger.info(
        "stream_vc input device %s: %s (latency %s requested)",
        device.index,
        device.name,
        config.input_latency,
    )
    stream = sd.RawInputStream(
        samplerate=CAPTURE_RATE,
        blocksize=hop,
        device=device.index,
        channels=1,
        dtype="int16",
        latency=config.input_latency,
    )
    # PortAudio does not have to honour the request, and "low" resolves to a different
    # number per host API. Report what was actually granted, before start() so a failing
    # start still leaves the number in the log.
    logger.info("stream_vc input stream latency: %.3fs", stream.latency)
    stream.start()
    return stream


async def _capture_read_loop(
    stream: sd.RawInputStream,
    hop: int,
    out_queue: Queue[CaptureItem],
    running: Event,
) -> None:
    """Steady state: keep reading hop samples at a time until a device fault.

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
    """
    # A drop while running = real backpressure. Throttle by time (ADR-0062).
    drop_throttle = LogThrottle()
    # An input overflow means the reader was late, which persists once it starts, so this
    # fires on every block (about 6 a second at block_ms=160) until it clears. Thin it by
    # time and meter it every occurrence -- exactly what its counterpart on the sink side
    # (playback.py's paOutputUnderflowed) already does.
    overflow_throttle = LogThrottle()
    while True:
        data, overflowed = await to_thread(stream.read, hop)
        if overflowed:
            telemetry.record("stream_vc_capture_overflow", 1.0)
            if (n := overflow_throttle.hit()) is not None:
                logger.warning("stream_vc capture input overflow (total %d)", n)
        block = pcm16_to_float32(bytes(data))
        if not drop_oldest_put(out_queue, block):
            if not running.is_set():
                # While paused vc_loop stops consuming, so the queue stays full and every
                # subsequent block is dropped. That is exactly the behaviour ADR-0050
                # intended (do not accumulate paused audio) and not an anomaly, so no
                # warning. Warning every time would emit about 6 lines a second at
                # block_ms=160 for the whole pause and make the warning meaningless.
                # They are still not discarded silently: they are counted under a
                # pause-specific stage -- mixing them into the same stage would pollute
                # the backpressure metric (stream_vc_capture_drop, used to assess RTF)
                # with the length of the pause.
                telemetry.record("stream_vc_capture_drop_paused", 1.0)
                continue
            telemetry.record("stream_vc_capture_drop", 1.0)
            if (n := drop_throttle.hit()) is not None:
                logger.warning(
                    "stream_vc capture queue full; dropped oldest block (total %d)",
                    n,
                )


async def capture_loop(
    config: StreamVcConfig,
    out_queue: Queue[CaptureItem],
    hop: int,
    ready: Event,
    running: Event,
) -> None:
    """Read hop samples at a time from the mic and push float32 blocks to out_queue.

    The first open is fail-loud (worker_startup); runtime device faults after that
    reconnect on their own (ADR-0050). Capture itself carries no state across a reopen,
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

    # Wait for the VC warmup to finish before opening the mic. Opening earlier lets the
    # audio that accumulated in real time during model loading flood the queue right
    # after startup, causing a storm of drops and filling the first few hundred ms with
    # stale audio (confirmed in the logs on real hardware).
    await ready.wait()
    await run_with_device_retry(
        open_stream=lambda: open_stream_vc_input_stream(config, hop),
        run=lambda stream: _capture_read_loop(stream, hop, out_queue, running),
        worker="stream_vc",
        label="stream vc capture",
        on_reopen=_signal_reopen,
        reopen_metric="stream_vc_capture_reopen",
    )
