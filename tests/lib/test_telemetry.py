"""Accumulation and summarisation in `Telemetry`.

Most cases build their own instance rather than touching the process-wide `telemetry`
singleton, so they say nothing about global state. The two that do exercise the singleton
(`log_summary`) rely on the autouse reset in `tests/conftest.py`.
"""

import logging

from vspeech.lib.telemetry import Telemetry
from vspeech.lib.telemetry import telemetry


def test_summary_percentiles():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    for v in [10.0, 20.0, 30.0, 40.0]:
        t.record("vc", v)
    s = t.summary()["vc"]
    assert s["count"] == 4
    assert s["p50"] == 25.0
    assert s["p95"] == 38.5
    assert s["max"] == 40.0


def test_max_samples_keeps_recent():
    t = Telemetry()
    t.configure(enabled=True, max_samples=2)
    for v in [1.0, 2.0, 3.0]:
        t.record("vc", v)
    s = t.summary()["vc"]
    assert s["count"] == 2
    assert s["max"] == 3.0
    assert s["p50"] == 2.5  # [2.0, 3.0] linear p50


def test_disabled_is_noop():
    t = Telemetry()
    t.configure(enabled=False, max_samples=5000)
    t.record("vc", 1.0)
    t.record_e2e(1.0)
    assert t.summary() == {}


def test_timer_records_stage():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    with t.timer("tts"):
        pass
    assert t.summary()["tts"]["count"] == 1


def test_record_e2e_present():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    t.record_e2e(1.0)
    t.record_e2e(3.0)
    assert t.summary()["e2e"]["count"] == 2
    assert t.summary()["e2e"]["max"] == 3.0


def test_log_summary_emits(caplog):
    telemetry.configure(enabled=True, max_samples=100)
    telemetry.record("vc", 0.5)
    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert any("telemetry summary" in r.message for r in caplog.records)


def test_log_summary_empty_silent(caplog):
    telemetry.configure(enabled=True, max_samples=100)
    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert not any("telemetry summary" in r.message for r in caplog.records)
