import logging
from asyncio import Event
from asyncio import Queue
from typing import Any

import numpy as np
import pytest

from vspeech.config import StreamVcConfig
from vspeech.lib.resample import PolyphaseResampler
from vspeech.stream_vc import capture as capture_mod
from vspeech.stream_vc.capture import CAPTURE_RATE
from vspeech.stream_vc.capture import CaptureItem
from vspeech.stream_vc.capture import CaptureSignal
from vspeech.stream_vc.capture import InputRateConverter
from vspeech.stream_vc.capture import InputTap
from vspeech.stream_vc.capture import _capture_read_loop
from vspeech.stream_vc.capture import capture_loop
from vspeech.stream_vc.capture import device_frames_per_read
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.capture import open_stream_vc_input_stream
from vspeech.stream_vc.capture import pcm16_to_float32


def test_ms_to_samples():
    assert ms_to_samples(80.0) == 1280  # 80ms @ 16k
    assert ms_to_samples(10.0) == 160
    assert ms_to_samples(0.0) == 0


def test_pcm16_to_float32_range():
    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    out = pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert abs(out[1] - 1.0) < 1e-3
    assert abs(out[2] + 1.0) < 1e-3


def test_pcm16_to_float32_empty():
    out = pcm16_to_float32(b"")
    assert out.shape == (0,)


class _FakeStream:
    """Returns hop samples n_blocks times, then raises OSError to mimic a device fault.

    `_capture_read_loop` is a `while True` that only exits on a device fault (delegating to
    run_with_device_retry), so the test stops at that same exit.
    """

    def __init__(self, n_blocks: int, overflowed: bool = False) -> None:
        self.remaining = n_blocks
        self.overflowed = overflowed
        self.closed = False

    def read(self, frames: int) -> tuple[bytes, bool]:
        if self.remaining <= 0:
            raise OSError("device gone")
        self.remaining -= 1
        return (b"\x00\x00" * frames, self.overflowed)

    def close(self) -> None:
        self.closed = True


class _PausingStream(_FakeStream):
    """A mic that drops into the paused state **during** the `pause_on_read`-th read.

    It reproduces the real-hardware ordering (warn on backpressure, then a pause arrives
    and it goes quiet) within a single loop. The gate closes before that block reaches the
    queue, so the `pause_on_read`-th block is already on the paused side, and the drops on
    the running side number `pause_on_read - 1`.
    """

    def __init__(self, n_blocks: int, running: Event, pause_on_read: int) -> None:
        super().__init__(n_blocks)
        self._running = running
        self._pause_on_read = pause_on_read
        self._read_count = 0

    def read(self, frames: int) -> tuple[bytes, bool]:
        self._read_count += 1
        if self._read_count == self._pause_on_read:
            # Event.clear() wakes no waiters, so touching it from a to_thread worker
            # thread does not touch the loop (unlike set(), it uses no call_soon).
            self._running.clear()
        return super().read(frames)


_open_taps: list[InputTap] = []


def _tap(stream: Any, device_rate: int = CAPTURE_RATE) -> InputTap:
    """Wrap a fake stream in the real InputTap the read loop now takes.

    Deliberately the real class, not another fake: it carries the device rate the loop
    filters with and owns the thread every read is made from (ADR-0077), so going through
    it is what keeps these tests exercising the path the pipeline uses. Every tap made
    here is closed by `close_taps` -- a tap owns a thread, and these tests reach the fault
    that ends the read loop without the pipeline's own close ever running.
    """
    tap = InputTap(stream, device_rate)
    _open_taps.append(tap)
    return tap


@pytest.fixture(autouse=True)
def close_taps():
    """Retire every tap a test made. Closing twice is a no-op, so taps the code under test
    already closed are fine to close again here."""
    yield
    while _open_taps:
        _open_taps.pop().close()


def _full_queue(hop: int) -> Queue[CaptureItem]:
    """Full = every subsequent put discards the oldest (i.e. a drop on every block)."""
    q: Queue[CaptureItem] = Queue(maxsize=1)
    q.put_nowait(np.zeros(hop, dtype=np.float32))
    return q


async def test_capture_drop_while_paused_does_not_warn(caplog, enabled_telemetry):
    """Drops during a pause are by design (ADR-0050), so no warning is emitted.

    While paused, vc_loop stops consuming, so capture_queue stays full and every
    subsequent block is dropped. Warning every time here would emit about 6 lines a second
    at block_ms=160 for the whole pause (the symptom reported on real hardware).
    """
    hop = 4
    running = Event()  # clear = paused
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                # device already at the pipeline rate = pass-through
                _tap(_FakeStream(5)),
                hop,
                _full_queue(hop),
                running,
            )
    assert not [r for r in caplog.records if "capture queue full" in r.getMessage()]
    # Rather than discarding silently, keep them observable under a pause-specific stage.
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop_paused"]["count"] == 5
    # The backpressure metric (used to assess RTF) is not polluted by pause drops.
    assert "stream_vc_capture_drop" not in summary


async def test_capture_drop_while_running_warns_once_per_episode(
    caplog, enabled_telemetry
):
    """Drops while running are real backpressure. Only one line at the head of the
    episode."""
    hop = 4
    running = Event()
    running.set()
    n = 51
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _tap(_FakeStream(n)),
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 1  # a tight loop = all within min_interval_s
    assert "(total 1)" in warnings[0].getMessage()
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == n  # telemetry every time
    assert "stream_vc_capture_drop_paused" not in summary


async def test_capture_overflow_warns_once_per_episode_and_is_metered(
    caplog, enabled_telemetry
):
    """An input overflow persists once it starts (the reader is late), so it fires on
    every block.

    Warning unconditionally would emit about 6 lines a second at block_ms=160 and bury the
    log -- the symptom ADR-0062 exists to remove. The queue is left un-full here so the
    drop path stays quiet and only the overflow path is under test.
    """
    hop = 4
    running = Event()
    running.set()
    n = 40
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _tap(_FakeStream(n, overflowed=True)),
                hop,
                Queue(),
                running,
            )
    warnings = [r for r in caplog.records if "input overflow" in r.getMessage()]
    assert len(warnings) == 1  # a tight loop = all within min_interval_s
    assert "(total 1)" in warnings[0].getMessage()
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_overflow"]["count"] == n  # telemetry every time
    assert "stream_vc_capture_drop" not in summary  # the queue never filled


async def test_capture_drop_switches_side_when_pause_arrives(caplog, enabled_telemetry):
    """Across a running -> pause transition, each block's drop is still attributed to the
    right side.

    Exactly the ordering that happens on real hardware (warn on backpressure, then go
    quiet on pause). This checks that no further warnings appear after the transition and
    that the drops from before the pause remain in the backpressure metric.
    """
    hop = 4
    running = Event()
    running.set()
    total_blocks = 10
    # pause happens during this read -> the 4th block is already on the paused side
    pause_on_read = 4
    running_drops = pause_on_read - 1
    stream = _PausingStream(total_blocks, running, pause_on_read=pause_on_read)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _tap(stream),
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 1  # only the head of the episode on the running side
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == running_drops
    assert summary["stream_vc_capture_drop_paused"]["count"] == (
        total_blocks - running_drops
    )


# --- Native-rate open + in-process conversion (ADR-0073 / ADR-0074) ------------------

_MME = 0
_WASAPI = 1
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
# The MME row lies about the rate (PortAudio hardcodes 44100 there) and truncates the
# name to 31 characters; the WASAPI row for the same endpoint carries the true mix rate.
_DEVICES = [
    {
        "index": 0,
        "name": "Microphone Array (Realtek(R) Au",
        "hostapi": _MME,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "index": 1,
        "name": "Microphone Array (Realtek(R) Audio)",
        "hostapi": _WASAPI,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
]


class _OpenedStream:
    """Stands in for sd.RawInputStream and records how it was opened.

    `samplerate` is the rate PortAudio reports the device actually runs at. A real stream
    always has it, and normally it equals the requested rate; `_reporting_stream` builds
    the variant that disagrees.

    `latency` is what PortAudio granted, which is not required to equal what was
    requested -- a fixed number here so the log assertion can tell the two apart.
    """

    latency = 0.032

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.samplerate = float(kwargs["samplerate"])
        self.started = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        pass


@pytest.fixture
def opened_streams(
    stub_device_table: Any, record_opened_streams: Any
) -> list[_OpenedStream]:
    """Stub the device table and sd.RawInputStream; yield the streams that got opened."""
    stub_device_table(_DEVICES, _HOSTAPIS)
    opened = record_opened_streams(capture_mod, "RawInputStream", _OpenedStream)
    return opened


def _open_log(caplog: pytest.LogCaptureFixture) -> str:
    lines = [r.getMessage() for r in caplog.records if "input device" in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def test_input_device_is_opened_at_the_resolved_native_rate(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    """The endpoint really runs at 48000, so that is what is opened -- not CAPTURE_RATE.

    Asking for 16000 would hand the conversion to the OS, and WASAPI shared mode would
    refuse the open outright (ADR-0073).
    """
    hop = ms_to_samples(160.0)
    with caplog.at_level(logging.INFO):
        tap = open_stream_vc_input_stream(StreamVcConfig(input_device_index=0), hop)
    assert tap.device_rate == 48000
    assert tap.stream is opened_streams[0]
    assert opened_streams[0].kwargs["samplerate"] == 48000
    assert opened_streams[0].kwargs["blocksize"] == 7680  # 160 ms at 48000
    assert opened_streams[0].started
    # The rate, how it was decided, and that a conversion is happening are all logged.
    line = _open_log(caplog)
    assert "48000Hz" in line
    assert "WASAPI" in line
    assert "16000Hz" in line
    assert "プロセス内で変換" in line


def test_configured_input_device_rate_wins_over_the_resolved_one(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    hop = ms_to_samples(160.0)
    with caplog.at_level(logging.INFO):
        tap = open_stream_vc_input_stream(
            StreamVcConfig(input_device_index=0, input_device_rate=44100), hop
        )
    assert tap.device_rate == 44100
    assert opened_streams[0].kwargs["samplerate"] == 44100
    assert opened_streams[0].kwargs["blocksize"] == 7056  # 160 ms at 44100
    assert "stream_vc.input_device_rate" in _open_log(caplog)


def test_a_device_at_the_capture_rate_is_opened_without_conversion(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    hop = ms_to_samples(160.0)
    with caplog.at_level(logging.INFO):
        tap = open_stream_vc_input_stream(
            StreamVcConfig(input_device_index=0, input_device_rate=CAPTURE_RATE), hop
        )
    assert tap.device_rate == CAPTURE_RATE
    assert opened_streams[0].kwargs["samplerate"] == CAPTURE_RATE
    assert opened_streams[0].kwargs["blocksize"] == hop
    assert "変換なし" in _open_log(caplog)


def _reporting_stream(monkeypatch: pytest.MonkeyPatch, reported: float) -> None:
    """Re-patch sd.RawInputStream with one that reports `reported` as its actual rate."""

    def _open(**kwargs: Any) -> _OpenedStream:
        stream = _OpenedStream(**kwargs)
        stream.samplerate = reported
        return stream

    monkeypatch.setattr(capture_mod.sd, "RawInputStream", _open)


def test_a_device_reporting_another_rate_is_warned_about(
    opened_streams: list[_OpenedStream],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The conversion keeps using the requested rate, so a hardware rate that differs is
    a slow drift in the audio and nothing else. This warning is its only trace."""
    _reporting_stream(monkeypatch, 47999.0)
    with caplog.at_level(logging.WARNING):
        tap = open_stream_vc_input_stream(
            StreamVcConfig(input_device_index=0), ms_to_samples(160.0)
        )
    # still the requested rate, not the reported one
    assert tap.device_rate == 48000
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "47999" in warnings[0]
    assert "48000" in warnings[0]


def test_a_device_reporting_the_requested_rate_stays_quiet(
    opened_streams: list[_OpenedStream],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _reporting_stream(monkeypatch, 48000.0)
    with caplog.at_level(logging.WARNING):
        open_stream_vc_input_stream(
            StreamVcConfig(input_device_index=0), ms_to_samples(160.0)
        )
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []


# --- Requested / granted device latency (ADR-0071) -----------------------------------
#
# The rate is decided by the shared opener above; the latency is the one part of the
# stream's shape this boundary still decides for itself, so it is asserted at the sd
# boundary -- `latency` is a passthrough and there is nothing downstream to observe it on.


def test_open_input_stream_requests_configured_latency(
    opened_streams: list[_OpenedStream],
) -> None:
    """The configured value reaches sounddevice unconverted -- a float is seconds,
    PortAudio's own unit (ADR-0071)."""
    open_stream_vc_input_stream(
        StreamVcConfig(input_device_index=0, input_latency=0.05), ms_to_samples(160.0)
    )
    assert opened_streams[0].kwargs["latency"] == 0.05


def test_open_input_stream_defaults_to_low(
    opened_streams: list[_OpenedStream],
) -> None:
    """No setting = the value that used to be hardcoded."""
    open_stream_vc_input_stream(
        StreamVcConfig(input_device_index=0), ms_to_samples(160.0)
    )
    assert opened_streams[0].kwargs["latency"] == "low"


def test_open_input_stream_uses_input_latency_not_output(
    opened_streams: list[_OpenedStream],
) -> None:
    """The output setting must not leak into the input stream."""
    config = StreamVcConfig(
        input_device_index=0, input_latency="low", output_latency="high"
    )
    open_stream_vc_input_stream(config, ms_to_samples(160.0))
    assert opened_streams[0].kwargs["latency"] == "low"


def test_open_input_stream_logs_requested_and_granted_latency(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    """Reading the granted value is the point: "low" resolves to wildly different
    numbers per host API, and it cannot be read off the requested value."""
    with caplog.at_level(logging.INFO):
        open_stream_vc_input_stream(
            StreamVcConfig(input_device_index=0), ms_to_samples(160.0)
        )
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Microphone Array" in messages  # the device line still names the device
    assert "low" in messages  # requested
    assert "0.032" in messages  # granted


def test_open_input_stream_logs_the_requested_latency_before_the_open(
    opened_streams: list[_OpenedStream],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing open must still say which latency was attempted.

    The granted line cannot carry it -- the open raised, so there is no stream to read
    it off -- which is why the request is logged before the open, the way lib/audio.py
    logs the device and the rate it is about to attempt. A device that refuses an
    explicit latency is the one shape preflight cannot pre-empt: its probe opens at the
    default latency (ADR-0076).
    """

    def _explode(**kwargs: Any) -> _OpenedStream:
        raise OSError("Invalid device latency")

    monkeypatch.setattr(capture_mod.sd, "RawInputStream", _explode)
    with caplog.at_level(logging.INFO):
        with pytest.raises(OSError):
            open_stream_vc_input_stream(
                StreamVcConfig(input_device_index=0, input_latency=0.05),
                ms_to_samples(160.0),
            )
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "0.05 requested" in m]
    # Nothing was granted: the line that would have carried the request never ran.
    assert [m for m in messages if "granted" in m] == []


def test_no_resampler_is_built_when_the_device_runs_at_the_capture_rate() -> None:
    """The pass-through path must stay bit-identical to the pre-ADR-0073 code: the read
    is handed on as-is, not even copied."""
    hop = ms_to_samples(160.0)
    converter = InputRateConverter(CAPTURE_RATE, hop)
    assert converter.resampler is None
    assert converter.frames_per_read == hop
    read = np.arange(hop, dtype=np.float32)
    blocks = converter.blocks(read)
    assert len(blocks) == 1
    assert blocks[0] is read


@pytest.mark.parametrize("rate", [48000, 44100])
def test_every_block_is_exactly_one_hop(rate: int) -> None:
    hop = ms_to_samples(160.0)
    converter = InputRateConverter(rate, hop)
    rng = np.random.default_rng(0)
    for _ in range(64):
        read = rng.standard_normal(converter.frames_per_read).astype(np.float32)
        for block in converter.blocks(read):
            assert block.shape == (hop,)
            assert block.dtype == np.float32


def test_a_fractional_ratio_carries_the_leftover_samples_forward() -> None:
    """44100 Hz at a 100-sample hop does not divide evenly: 276 device frames per read
    become 100.14 output samples. The surplus must accumulate into the next block rather
    than be dropped, and no block may be emitted short.
    """
    rate, hop = 44100, 100
    converter = InputRateConverter(rate, hop)
    assert converter.frames_per_read == 276
    rng = np.random.default_rng(3)
    reads = [
        rng.standard_normal(converter.frames_per_read).astype(np.float32)
        for _ in range(200)
    ]
    blocks = [block for read in reads for block in converter.blocks(read)]
    assert {block.shape for block in blocks} == {(hop,)}
    emitted = np.concatenate(blocks)
    # Everything the filter produced for the same audio. What is missing from `emitted`
    # may only be the partial block still inside the converter.
    whole = PolyphaseResampler(rate, CAPTURE_RATE).process(np.concatenate(reads))
    assert 0 < len(whole) - len(emitted) < hop, "this ratio must leave a remainder"
    # Not bit-equality: BLAS sums one 20028-row matvec in a different order than 200
    # small ones (see the irregular-block test in tests/test_resample.py).
    assert np.allclose(emitted, whole[: len(emitted)], atol=1e-5, rtol=0)


def _pcm_chunks(seed: int, count: int, frames: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(-20000, 20000, size=frames, dtype=np.int16).tobytes()
        for _ in range(count)
    ]


def _drain(out_queue: Queue[CaptureItem]) -> list[Any]:
    items: list[Any] = []
    while not out_queue.empty():
        items.append(out_queue.get_nowait())
    return items


class _DeviceStream:
    """A mic that hands out pre-generated int16 reads and then raises `final`.

    Records the frame count each read was asked for and the queue depth at that moment.
    That depth is how the one-read-one-block cadence gets measured instead of reasoned
    about: read i must see exactly i blocks already queued. Reading qsize() from the
    to_thread worker is safe because the loop coroutine is parked on this very call, so
    nothing else can touch the queue meanwhile.
    """

    def __init__(
        self,
        chunks: list[bytes],
        out_queue: Queue[CaptureItem],
        final: BaseException | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._queue = out_queue
        self._final = final if final is not None else OSError("device gone")
        self.frames_seen: list[int] = []
        self.depth_at_read: list[int] = []
        self.closed = False

    def read(self, frames: int) -> tuple[bytes, bool]:
        self.frames_seen.append(frames)
        self.depth_at_read.append(self._queue.qsize())
        if not self._chunks:
            raise self._final
        chunk = self._chunks.pop(0)
        assert len(chunk) == frames * 2, "the loop asked for an unexpected frame count"
        return chunk, False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("rate", [48000, 44100])
async def test_one_device_read_puts_exactly_one_block_on_the_queue(rate: int) -> None:
    """The queue keeps its one-block-per-device-tick cadence at both device rates.

    Measured, not argued: a resampler that held audio back would make the first block
    late and every later one bunch up, which is a whole hop (160 ms) of latency
    (ADR-0073).
    """
    hop = ms_to_samples(160.0)
    frames = device_frames_per_read(hop, rate)
    reads = 40
    out_queue: Queue[CaptureItem] = Queue()
    running = Event()
    running.set()
    stream = _DeviceStream(_pcm_chunks(1, reads, frames), out_queue)
    with pytest.raises(OSError):
        await _capture_read_loop(
            _tap(stream, rate),
            hop,
            out_queue,
            running,
        )
    # The device is read in device frames, one block's worth of time at a time.
    assert stream.frames_seen == [frames] * (reads + 1)
    assert stream.depth_at_read == list(range(reads + 1))
    assert [block.shape for block in _drain(out_queue)] == [(hop,)] * reads


async def test_matching_rate_queues_the_decoded_read_untouched() -> None:
    """No resampler in the path: the queued block is exactly what the decode produced."""
    hop = 4
    chunks = _pcm_chunks(2, 3, hop)
    out_queue: Queue[CaptureItem] = Queue()
    running = Event()
    running.set()
    stream = _DeviceStream(list(chunks), out_queue)
    with pytest.raises(OSError):
        await _capture_read_loop(
            _tap(stream),
            hop,
            out_queue,
            running,
        )
    assert stream.frames_seen == [hop] * 4
    queued = _drain(out_queue)
    assert len(queued) == len(chunks)
    for block, chunk in zip(queued, chunks, strict=True):
        assert np.array_equal(block, pcm16_to_float32(chunk))


class _EndOfTest(Exception):
    """Not a device error, so it ends run_with_device_retry's loop instead of retrying."""


def _patch_instant_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real reconnect logic but skip its backoff sleep."""
    real = capture_mod.run_with_device_retry

    async def _no_sleep(_seconds: float) -> None:
        return None

    async def _instant(**kwargs: Any) -> None:
        await real(**kwargs, sleep=_no_sleep)

    monkeypatch.setattr(capture_mod, "run_with_device_retry", _instant)


async def test_reopen_does_not_carry_converter_state_into_the_new_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a device fault the new stream starts from a clean filter tail and an empty
    partial block.

    Both fake devices hand out the same audio, so "the blocks after the reopen equal the
    blocks before it" is exactly the statement that nothing survived. 44100 -> 16000 at a
    100-sample hop is used because it leaves a partial block behind at the fault, so a
    carried-over remainder would shift every block afterwards.
    """
    rate, hop = 44100, 100
    reads = 6
    chunks = _pcm_chunks(5, reads, device_frames_per_read(hop, rate))
    out_queue: Queue[CaptureItem] = Queue()
    streams = [
        _DeviceStream(list(chunks), out_queue),
        _DeviceStream(list(chunks), out_queue, final=_EndOfTest()),
    ]
    monkeypatch.setattr(
        capture_mod,
        "open_stream_vc_input_stream",
        lambda config, hop: _tap(streams.pop(0), rate),
    )
    _patch_instant_reopen(monkeypatch)
    ready = Event()
    ready.set()
    running = Event()
    running.set()
    with pytest.raises(_EndOfTest):
        await capture_loop(StreamVcConfig(), out_queue, hop, ready, running)
    items = _drain(out_queue)
    boundary = next(i for i, item in enumerate(items) if item is CaptureSignal.REOPEN)
    before, after = items[:boundary], items[boundary + 1 :]
    assert len(before) == reads
    assert len(after) == reads
    for old, new in zip(before, after, strict=True):
        assert np.array_equal(old, new)
