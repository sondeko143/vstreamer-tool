from vspeech.lib.telemetry import telemetry
from vspeech.worker.vc import record_vc_elapsed


def test_vc_records_duration():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    record_vc_elapsed(0.25)
    s = telemetry.summary()["vc"]
    assert s["count"] == 1
    assert s["max"] == 0.25
