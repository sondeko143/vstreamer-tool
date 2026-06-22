from vspeech.config import RecordingConfig
from vspeech.worker.recording import build_recording_output


def test_recording_output_has_trace():
    cfg = RecordingConfig()
    out = build_recording_output(cfg, frames=b"abc")
    assert out.trace_id != ""
    assert out.origin_ts > 0.0
    assert out.sound is not None
    assert out.sound.data == b"abc"
