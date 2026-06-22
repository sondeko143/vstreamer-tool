import logging

from vspeech.lib.telemetry import telemetry


def test_log_summary_emits(caplog):
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    telemetry.record("vc", 0.5)
    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert any("telemetry summary" in r.message for r in caplog.records)


def test_log_summary_empty_silent(caplog):
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert not any("telemetry summary" in r.message for r in caplog.records)
