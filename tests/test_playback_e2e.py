from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.lib.telemetry import telemetry
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.playback import record_playback_e2e


def _speech(origin_ts: float) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(event=EventType.playback),
        following_events=[],
        text="",
        sound=SoundInput(
            data=b"00", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        file_path="",
        filters=[],
        trace_id="abc",
        origin_ts=origin_ts,
    )


def test_e2e_recorded():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=100.0), now=101.5, cfg=cfg)
    assert e2e == 1.5
    assert telemetry.summary()["e2e"]["count"] == 1


def test_skew_negative_not_recorded():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=200.0), now=100.0, cfg=cfg)
    assert e2e is None
    assert "e2e" not in telemetry.summary()


def test_no_origin_skipped():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=0.0), now=100.0, cfg=cfg)
    assert e2e is None
