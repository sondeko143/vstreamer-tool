import time as _time

from vspeech.config import RecordingConfig
from vspeech.lib.telemetry import telemetry
from vspeech.worker.recording import build_recording_output
from vspeech.worker.recording import record_recording_metrics
from vspeech.worker.recording import utterance_capture_sec


def test_utterance_capture_sec_int16_mono():
    cfg = RecordingConfig()  # format=INT16 (2 bytes), channels=1, rate=16000
    one_second = b"\x00" * (2 * 1 * 16000)
    assert utterance_capture_sec(one_second, cfg) == 1.0
    assert utterance_capture_sec(b"", cfg) == 0.0


def test_build_recording_output_corrects_origin_ts_by_silence_lag():
    cfg = RecordingConfig()
    lag = 0.5
    before = _time.time()
    out = build_recording_output(cfg, frames=b"abc", silence_lag=lag)
    after = _time.time()
    assert before - lag <= out.origin_ts <= after - lag


def test_build_recording_output_default_lag_keeps_origin_now():
    cfg = RecordingConfig()
    before = _time.time()
    out = build_recording_output(cfg, frames=b"abc")
    after = _time.time()
    assert before <= out.origin_ts <= after


def test_record_recording_metrics_silence_stop_records_both():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    record_recording_metrics(
        capture_sec=1.0, silence_lag=0.4, stop_reason="silence", trace_id="t"
    )
    summary = telemetry.summary()
    assert summary["rec_capture"]["count"] == 1
    assert summary["rec_capture"]["max"] == 1.0
    assert summary["rec_silence_lag"]["count"] == 1
    assert summary["rec_silence_lag"]["max"] == 0.4


def test_record_recording_metrics_maxlen_stop_skips_silence_lag():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    record_recording_metrics(
        capture_sec=2.0, silence_lag=0.0, stop_reason="maxlen", trace_id="t"
    )
    summary = telemetry.summary()
    assert summary["rec_capture"]["count"] == 1
    assert "rec_silence_lag" not in summary
