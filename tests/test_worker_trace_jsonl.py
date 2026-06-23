import json
from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.lib.telemetry import telemetry
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.playback import record_playback_e2e
from vspeech.worker.vc import record_vc_elapsed


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _speech(origin_ts: float, trace_id: str) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(event=EventType.playback),
        following_events=[],
        text="",
        sound=SoundInput(data=b"00", rate=16000, format=SampleFormat.INT16, channels=1),
        file_path="",
        filters=[],
        trace_id=trace_id,
        origin_ts=origin_ts,
    )


def test_vc_records_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    record_vc_elapsed(0.5, trace_id="vc-tid")
    telemetry.reset()  # close handle so file is flushed/closed
    rows = _read(p)
    assert rows[0]["stage"] == "vc"
    assert rows[0]["trace_id"] == "vc-tid"


def test_playback_e2e_records_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    cfg = TelemetryConfig()
    record_playback_e2e(_speech(origin_ts=100.0, trace_id="pb-tid"), now=101.0, cfg=cfg)
    telemetry.reset()
    rows = _read(p)
    assert rows[0]["stage"] == "e2e"
    assert rows[0]["trace_id"] == "pb-tid"
