import json

from vspeech.lib.telemetry import Telemetry


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_writes_jsonl(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    t.record("vc", 1.25, trace_id="abc")
    rows = _read(p)
    assert len(rows) == 1
    r = rows[0]
    assert r["stage"] == "vc"
    assert r["dur_s"] == 1.25
    assert r["trace_id"] == "abc"
    assert isinstance(r["pid"], int)
    assert isinstance(r["ts"], float)


def test_timer_writes_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    with t.timer("tts", trace_id="t2"):
        pass
    rows = _read(p)
    assert rows[0]["stage"] == "tts"
    assert rows[0]["trace_id"] == "t2"


def test_record_e2e_writes_stage_e2e(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    t.record_e2e(1.7, trace_id="t3")
    rows = _read(p)
    assert rows[0]["stage"] == "e2e"
    assert rows[0]["dur_s"] == 1.7
    assert rows[0]["trace_id"] == "t3"


def test_no_jsonl_when_path_empty(tmp_path):
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path="")
    t.record("vc", 1.0, trace_id="x")
    assert not (tmp_path / "tel.jsonl").exists()


def test_disabled_writes_nothing(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=False, max_samples=100, jsonl_path=str(p))
    t.record("vc", 1.0, trace_id="x")
    assert not p.exists()


def test_uncreatable_path_is_resilient(tmp_path):
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    bad = str(afile / "sub" / "tel.jsonl")  # parent is a file → mkdir fails
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=bad)  # must not raise
    t.record("vc", 1.0, trace_id="x")  # must not raise
    assert t.summary()["vc"]["count"] == 1  # in-memory aggregation still works
