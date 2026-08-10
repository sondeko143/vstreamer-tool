"""Native-rate open + in-process conversion for the utterance recording path
(ADR-0070/0071, Task 7).

Mirrors the fixture shapes already used in tests/test_stream_vc_capture.py (there is
no tests/conftest.py in this repo, so per-file duplication is the house pattern).
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from vspeech.config import RecordingConfig
from vspeech.config import SampleFormat
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.worker import recording as recording_mod
from vspeech.worker.recording import convert_chunk
from vspeech.worker.recording import device_frames_per_read
from vspeech.worker.recording import open_input_stream
from vspeech.worker.recording import sd_recording_worker

# --- device_frames_per_read --------------------------------------------------------


def test_device_frames_per_read_matching_rate_returns_chunk_untouched():
    assert device_frames_per_read(1024, 16000, 16000) == 1024


def test_device_frames_per_read_scales_by_the_rate_ratio():
    # 100ms of 16000Hz audio read from a 48000Hz device is 4800 device frames.
    assert device_frames_per_read(1600, 48000, 16000) == 4800


def test_device_frames_per_read_rounds_a_fractional_ratio():
    assert device_frames_per_read(100, 44100, 16000) == 276


# --- open_input_stream: native-rate open + logging ---------------------------------

_MME = 0
_WASAPI = 1
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
# The MME row lies about the rate (PortAudio hardcodes 44100 there) and truncates the
# name to 31 characters; the WASAPI row for the same endpoint carries the true mix rate
# (identical fixture shape to test_stream_vc_capture.py).
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

    `samplerate` is what PortAudio reports the device actually runs at; a real stream
    always has it, and normally it equals the requested rate.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.samplerate = float(kwargs["samplerate"])
        self.started = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        pass


@pytest.fixture
def opened_streams(monkeypatch: pytest.MonkeyPatch) -> list[_OpenedStream]:
    """Stub the device table and sd.RawInputStream; yield the streams that got opened."""
    import vspeech.lib.audio as audio

    def _query_devices(index: int | None = None):
        if index is None:
            return _DEVICES
        return next(d for d in _DEVICES if d["index"] == index)

    monkeypatch.setattr(audio.sd, "query_devices", _query_devices)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _HOSTAPIS)
    opened: list[_OpenedStream] = []

    def _open(**kwargs: Any) -> _OpenedStream:
        stream = _OpenedStream(**kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(recording_mod.sd, "RawInputStream", _open)
    return opened


def _open_log(caplog: pytest.LogCaptureFixture) -> str:
    lines = [r.getMessage() for r in caplog.records if "input device" in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def test_input_device_is_opened_at_the_resolved_native_rate(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    """The endpoint really runs at 48000, so that is what is opened -- not
    recording.rate (16000 by default). Asking for 16000 would hand the conversion to
    the OS, and WASAPI shared mode would refuse the open outright (ADR-0070)."""
    cfg = RecordingConfig(input_device_index=0)
    with caplog.at_level(logging.INFO):
        stream, rate = open_input_stream(cfg)
    assert rate == 48000
    assert stream is opened_streams[0]
    assert opened_streams[0].kwargs["samplerate"] == 48000
    assert opened_streams[0].kwargs["blocksize"] == device_frames_per_read(
        cfg.chunk, 48000, cfg.rate
    )
    assert opened_streams[0].kwargs["channels"] == cfg.channels
    assert opened_streams[0].started
    line = _open_log(caplog)
    assert "48000Hz" in line
    assert "WASAPI" in line
    assert "16000Hz" in line
    assert "プロセス内で変換" in line


def test_configured_input_device_rate_wins_over_the_resolved_one(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    cfg = RecordingConfig(input_device_index=0, input_device_rate=44100)
    with caplog.at_level(logging.INFO):
        _, rate = open_input_stream(cfg)
    assert rate == 44100
    assert opened_streams[0].kwargs["samplerate"] == 44100
    assert "recording.input_device_rate" in _open_log(caplog)


def test_a_device_at_the_config_rate_is_opened_without_conversion(
    opened_streams: list[_OpenedStream], caplog: pytest.LogCaptureFixture
) -> None:
    cfg = RecordingConfig(input_device_index=0, input_device_rate=16000)
    with caplog.at_level(logging.INFO):
        _, rate = open_input_stream(cfg)
    assert rate == 16000
    assert opened_streams[0].kwargs["blocksize"] == cfg.chunk
    assert "変換なし" in _open_log(caplog)


def _reporting_stream(monkeypatch: pytest.MonkeyPatch, reported: float) -> None:
    """Re-patch sd.RawInputStream with one that reports `reported` as its actual rate."""

    def _open(**kwargs: Any) -> _OpenedStream:
        stream = _OpenedStream(**kwargs)
        stream.samplerate = reported
        return stream

    monkeypatch.setattr(recording_mod.sd, "RawInputStream", _open)


def test_a_device_reporting_another_rate_is_warned_about(
    opened_streams: list[_OpenedStream],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The conversion keeps using the requested rate, so a hardware rate that differs
    is only a slow drift in the audio; this warning is its only trace."""
    _reporting_stream(monkeypatch, 47999.0)
    cfg = RecordingConfig(input_device_index=0)
    with caplog.at_level(logging.WARNING):
        _, rate = open_input_stream(cfg)
    assert rate == 48000  # still the requested rate, not the reported one
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
    cfg = RecordingConfig(input_device_index=0)
    with caplog.at_level(logging.WARNING):
        open_input_stream(cfg)
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []


# --- convert_chunk -------------------------------------------------------------------


def test_convert_chunk_pass_through_is_bit_identical_at_full_scale() -> None:
    """No resampler means no decode/encode round trip: decode_pcm/encode_pcm is not
    bit-exact at full scale (int16 -32768 round-trips to -32767), so skipping the
    round trip is what makes the matching-rate path bit-identical to the pre-ADR-0070
    code."""
    cfg = RecordingConfig()  # INT16 mono, rate=16000
    data = np.array([-32768, 32767, 0, -1, 12345], dtype=np.int16).tobytes()
    out, frame_count = convert_chunk(data, None, cfg)
    assert out == data
    assert frame_count == 5


def test_convert_chunk_resamples_and_reports_the_measured_frame_count() -> None:
    """The frame count must come from what the resampler actually produced, not a
    config constant -- the trap this task exists to avoid."""
    cfg = RecordingConfig(rate=16000)
    resampler = PolyphaseResampler(48000, 16000)
    rng = np.random.default_rng(0)
    data = rng.integers(-20000, 20000, size=4800, dtype=np.int16).tobytes()
    out, frame_count = convert_chunk(data, resampler, cfg)
    assert frame_count == 1600  # 48000 -> 16000 is exact 3:1 decimation
    assert len(out) == frame_count * 2  # INT16 mono = 2 bytes/frame


def test_convert_chunk_preserves_multichannel_layout() -> None:
    """channels > 1 must not be folded to mono by the resample round trip."""
    cfg = RecordingConfig(channels=2, rate=16000)
    resampler = PolyphaseResampler(48000, 16000)
    n = 4800
    stereo = np.empty(n * 2, dtype=np.int16)
    stereo[0::2] = 20000
    stereo[1::2] = -20000
    out, frame_count = convert_chunk(stereo.tobytes(), resampler, cfg)
    decoded = decode_pcm(out, SampleFormat.INT16, 2)
    assert decoded.shape == (frame_count, 2)
    # If the channels had been downmixed, both would collapse toward ~0.
    assert decoded[:, 0].mean() > 0.3
    assert decoded[:, 1].mean() < -0.3


# --- sd_recording_worker: full loop --------------------------------------------------


class _FakeDeviceStream:
    """A mic that hands out reads via `make_chunk(read_index)` until `n_reads` is
    exhausted, then raises `final` -- mirroring a real device fault (`while
    stream.active` inside sd_recording_worker only exits that way).

    `overflow_on` marks 1-based read indices whose overflow flag comes back True.
    """

    def __init__(
        self,
        frames_per_read: int,
        make_chunk: Callable[[int], bytes],
        *,
        n_reads: int = 10_000,
        overflow_on: frozenset[int] = frozenset(),
        final: BaseException | None = None,
    ) -> None:
        self.active = True
        self.frames_per_read = frames_per_read
        self._make_chunk = make_chunk
        self.n_reads = n_reads
        self.overflow_on = overflow_on
        self.final = final if final is not None else OSError("device gone")
        self.reads = 0

    def read(self, frames: int) -> tuple[bytes, bool]:
        assert frames == self.frames_per_read, (frames, self.frames_per_read)
        self.reads += 1
        if self.reads > self.n_reads:
            raise self.final
        return self._make_chunk(self.reads), self.reads in self.overflow_on

    def close(self) -> None:
        self.active = False


def _constant_mono(frames_per_read: int, amplitude: int) -> Callable[[int], bytes]:
    def _make(_read_index: int) -> bytes:
        return np.full(frames_per_read, amplitude, dtype=np.int16).tobytes()

    return _make


def _constant_stereo(
    frames_per_read: int, left: int, right: int
) -> Callable[[int], bytes]:
    def _make(_read_index: int) -> bytes:
        stereo = np.empty(frames_per_read * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        return stereo.tobytes()

    return _make


async def test_time_conversion_matches_real_seconds_at_a_faster_device_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48000Hz device, recording.rate=16000: real elapsed audio time must control
    interval_sec/max_recording_sec, not a device-rate-frame count misread as
    config.rate frames.

    chunk=400 is deliberately smaller than one interval (interval_sec=0.1 -> a 1600-
    frame threshold at config.rate), so FOUR reads accumulate per interval crossing --
    unlike a chunk sized to exactly one interval, where a wrong (device-rate) frame
    count and the right (config-rate) one happen to cross the same threshold after the
    same single read and the bug would go unnoticed. Here a device-rate frame count
    would cross the threshold after only two reads (1200 device frames each, 3x over
    the 1600 threshold after two), corrupting both the read count and the captured
    audio length -- exactly the trap this task exists to avoid.

    frames_per_read = device_frames_per_read(400, 48000, 16000) = 1200 device frames,
    an exact 3:1 decimation to 400 output frames/read. 4 reads/interval x 4 intervals
    (1 transition + 3 accumulated, tripping max_recording_sec=0.25 on the 4th, same
    float-accumulation reasoning as a single-read-per-interval design) = 16 reads and
    0.4s of REAL captured audio.
    """
    cfg = RecordingConfig(
        rate=16000,
        chunk=400,
        interval_sec=0.1,
        silence_threshold=-100,  # anything above near-total silence counts as speech
        max_recording_sec=0.25,
    )
    frames_per_read = device_frames_per_read(cfg.chunk, 48000, cfg.rate)
    assert frames_per_read == 1200
    stream = _FakeDeviceStream(frames_per_read, _constant_mono(frames_per_read, 12000))
    monkeypatch.setattr(
        recording_mod, "open_input_stream", lambda config: (stream, 48000)
    )

    gen = sd_recording_worker(cfg)
    utterance = await anext(gen)
    await gen.aclose()

    assert utterance.stop_reason == "maxlen"
    assert utterance.capture_sec == pytest.approx(0.4, abs=1e-6)
    assert stream.reads == 16


async def test_channels_are_preserved_through_the_full_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """channels > 1 stays 2 channels end to end, distinguishable after resampling --
    not folded to mono. If it had been downmixed, left(+20000) + right(-20000) would
    collapse toward ~0 on both channels."""
    cfg = RecordingConfig(
        rate=16000,
        channels=2,
        chunk=1600,
        interval_sec=0.1,
        silence_threshold=-100,
        max_recording_sec=0.05,  # trips on the 2nd interval crossing
    )
    frames_per_read = device_frames_per_read(cfg.chunk, 48000, cfg.rate)
    stream = _FakeDeviceStream(
        frames_per_read, _constant_stereo(frames_per_read, 20000, -20000)
    )
    monkeypatch.setattr(
        recording_mod, "open_input_stream", lambda config: (stream, 48000)
    )

    gen = sd_recording_worker(cfg)
    utterance = await anext(gen)
    await gen.aclose()

    assert utterance.stop_reason == "maxlen"
    decoded = decode_pcm(utterance.frames, cfg.format, cfg.channels)
    assert decoded.ndim == 2
    assert decoded.shape[1] == 2
    assert decoded[:, 0].mean() > 0.3
    assert decoded[:, 1].mean() < -0.3


async def test_matching_rate_output_is_bit_identical_to_the_raw_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the device already runs at recording.rate, the accumulated utterance bytes
    are exactly the concatenated raw reads -- no decode/encode round trip, which is
    not bit-exact at full scale."""
    frames_per_read = 4
    pattern = np.array([-32768, 32767, 0, -1], dtype=np.int16).tobytes()
    cfg = RecordingConfig(
        rate=16000,
        chunk=frames_per_read,
        interval_sec=frames_per_read / 16000,  # one interval boundary per read
        silence_threshold=-100,
        max_recording_sec=frames_per_read / 16000 / 2,  # trips on the 2nd crossing
    )
    stream = _FakeDeviceStream(frames_per_read, lambda _i: pattern)
    monkeypatch.setattr(
        recording_mod, "open_input_stream", lambda config: (stream, 16000)
    )

    gen = sd_recording_worker(cfg)
    utterance = await anext(gen)
    await gen.aclose()

    assert utterance.stop_reason == "maxlen"
    assert utterance.frames == pattern * 2


async def test_overflow_logs_the_same_warning_as_before(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _EndOfTest(Exception):
        pass

    frames_per_read = 1600
    stream = _FakeDeviceStream(
        frames_per_read,
        _constant_mono(frames_per_read, 100),
        n_reads=1,
        overflow_on=frozenset({1}),
        final=_EndOfTest(),
    )
    monkeypatch.setattr(
        recording_mod, "open_input_stream", lambda config: (stream, 16000)
    )
    cfg = RecordingConfig(rate=16000, chunk=frames_per_read, interval_sec=100.0)
    gen = sd_recording_worker(cfg)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_EndOfTest):
            await anext(gen)
    overflow_warnings = [
        r.getMessage() for r in caplog.records if "overflow" in r.getMessage()
    ]
    assert overflow_warnings == ["recording input overflow: samples were dropped"]


async def test_device_fault_retries_by_reopening_the_stream(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The pre-existing retry behaviour (loop back to the top of the outer while and
    reopen, no backoff) must be unchanged by the native-rate open."""

    class _EndOfTest(Exception):
        pass

    frames_per_read = 1600
    first = _FakeDeviceStream(
        frames_per_read, _constant_mono(frames_per_read, 100), n_reads=2
    )
    second = _FakeDeviceStream(
        frames_per_read,
        _constant_mono(frames_per_read, 100),
        n_reads=0,
        final=_EndOfTest(),
    )
    streams = [first, second]
    opens: list[int] = []

    def _open(config: RecordingConfig):
        opens.append(len(opens))
        return streams.pop(0), 16000

    monkeypatch.setattr(recording_mod, "open_input_stream", _open)
    cfg = RecordingConfig(rate=16000, chunk=frames_per_read, interval_sec=100.0)
    gen = sd_recording_worker(cfg)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_EndOfTest):
            await anext(gen)
    assert len(opens) == 2
    assert first.active is False  # closed on the fault, before the reopen
    assert second.active is False  # closed again once _EndOfTest propagates
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "retry" in warnings[0]
    assert "device gone" in warnings[0]
