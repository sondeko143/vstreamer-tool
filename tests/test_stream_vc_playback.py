import logging

from vspeech.config import StreamVcConfig
from vspeech.lib.audio import DeviceInfo
from vspeech.stream_vc.playback import detect_gap


def test_detect_gap_none_prev():
    assert detect_gap(None, 0) == 0


def test_detect_gap_contiguous():
    assert detect_gap(4, 5) == 0


def test_detect_gap_missing():
    assert detect_gap(4, 7) == 2  # 5, 6 missing


def test_detect_gap_reorder_or_dup_is_zero():
    assert detect_gap(7, 5) == 0  # out-of-order/dup -> not a forward gap


class _RecordingOutputStream:
    """Records the kwargs sounddevice would have been constructed with (mirrors the
    input-side fake in test_stream_vc_capture.py)."""

    # Declared at class level so it is a known attribute (ty) rather than one that
    # springs into existence on first construction.
    kwargs: dict[str, object] = {}
    # What PortAudio granted, which is not required to equal what was requested.
    latency = 0.048

    def __init__(self, **kwargs) -> None:
        _RecordingOutputStream.kwargs = kwargs

    def start(self) -> None:
        pass


def _fake_output_device() -> DeviceInfo:
    return DeviceInfo(
        host_api=0,
        max_input_channels=0,
        max_output_channels=2,
        name="Fake Speaker",
        index=9,
    )


def _patch_output_open(monkeypatch) -> None:
    from vspeech.stream_vc import playback

    monkeypatch.setattr(
        playback,
        "resolve_stream_vc_output_device",
        lambda config: _fake_output_device(),
    )
    monkeypatch.setattr(playback.sd, "RawOutputStream", _RecordingOutputStream)


def test_open_output_stream_requests_configured_latency(monkeypatch):
    """The configured value reaches sounddevice unconverted (ADR-0070)."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    playback.open_stream_vc_output_stream(
        StreamVcConfig(output_latency="high"), sample_rate=16000
    )
    assert _RecordingOutputStream.kwargs["latency"] == "high"


def test_open_output_stream_defaults_to_low(monkeypatch):
    """No setting = the value that used to be hardcoded."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    playback.open_stream_vc_output_stream(StreamVcConfig(), sample_rate=16000)
    assert _RecordingOutputStream.kwargs["latency"] == "low"


def test_open_output_stream_uses_output_latency_not_input(monkeypatch):
    """The input setting must not leak into the output stream."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    config = StreamVcConfig(input_latency="high", output_latency=0.02)
    playback.open_stream_vc_output_stream(config, sample_rate=16000)
    assert _RecordingOutputStream.kwargs["latency"] == 0.02


def test_open_output_stream_logs_requested_and_granted_latency(caplog, monkeypatch):
    """consumer.py reuses this function, so the consumer machine gets the same line."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    with caplog.at_level(logging.INFO):
        playback.open_stream_vc_output_stream(StreamVcConfig(), sample_rate=16000)
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Fake Speaker" in messages  # the device line still names the device
    assert "low" in messages  # requested
    assert "0.048" in messages  # granted
