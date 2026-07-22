import numpy as np

from scripts.stream_vc_rtf import BlockResult
from scripts.stream_vc_rtf import format_table
from scripts.stream_vc_rtf import go_no_go
from scripts.stream_vc_rtf import make_voiced_signal
from scripts.stream_vc_rtf import parse_grid
from scripts.stream_vc_rtf import recommend
from scripts.stream_vc_rtf import summarize


def test_make_voiced_signal_shape_and_range():
    sig = make_voiced_signal(rate=16000, seconds=0.5, seed=0)
    assert sig.dtype == np.float32
    assert sig.shape == (8000,)
    assert np.max(np.abs(sig)) <= 1.0
    assert np.any(sig != 0.0)  # not silent -> f0 extractor won't early-out
    # stronger non-silence guarantee: a genuinely voiced signal has real RMS
    # energy, not just a stray nonzero sample (near-silent input would trip an
    # f0 extractor's unvoiced early-out).
    assert np.sqrt(np.mean(sig.astype(np.float64) ** 2)) > 0.01


def test_make_voiced_signal_is_deterministic():
    a = make_voiced_signal(16000, 0.2, seed=7)
    b = make_voiced_signal(16000, 0.2, seed=7)
    assert np.array_equal(a, b)


def test_parse_grid():
    assert parse_grid("20,40,80") == [20.0, 40.0, 80.0]
    assert parse_grid("100") == [100.0]


def test_summarize_computes_rtf_and_feasible():
    # block of 40ms -> 0.04s wall. compute ~10ms p95 -> RTF 0.25 < 0.5 -> feasible.
    lat = [0.008, 0.009, 0.010, 0.011, 0.012]
    r = summarize(
        lat, block_seconds=0.04, margin=0.5, block_ms=40, context_ms=200, f0="rmvpe"
    )
    assert r.feasible is True
    assert r.rtf_p95 < 0.5
    # latency ~ block(40) + p95(~12) ms
    assert 45.0 < r.latency_ms < 60.0


def test_summarize_infeasible_when_rtf_exceeds_margin():
    lat = [0.030, 0.031, 0.032]  # ~31ms vs 40ms block -> RTF ~0.79 > 0.5
    r = summarize(
        lat, block_seconds=0.04, margin=0.5, block_ms=40, context_ms=800, f0="rmvpe"
    )
    assert r.feasible is False


def _mk(feasible: bool, latency_ms: float) -> BlockResult:
    return BlockResult(
        block_ms=40,
        context_ms=200,
        f0="rmvpe",
        p50_ms=1,
        p95_ms=1,
        max_ms=1,
        rtf_p95=0.1,
        latency_ms=latency_ms,
        feasible=feasible,
    )


def test_recommend_picks_lowest_latency_feasible():
    results = [_mk(True, 60), _mk(False, 30), _mk(True, 45)]
    best = recommend(results)
    assert best is not None and best.latency_ms == 45


def test_recommend_none_when_no_feasible():
    assert recommend([_mk(False, 30)]) is None


def test_go_no_go():
    assert go_no_go([_mk(False, 30), _mk(True, 45)]) is True
    assert go_no_go([_mk(False, 30)]) is False


def test_format_table_marks_feasible():
    lines = format_table([_mk(True, 45), _mk(False, 30)]).splitlines()
    data = [ln for ln in lines if "rmvpe" in ln]  # data rows only (skip header/rule)
    assert len(data) == 2
    marked = [ln for ln in data if "[FEASIBLE]" in ln]
    assert len(marked) == 1
    assert "45.0" in marked[0]  # the feasible row (latency 45) carries the marker
    unmarked = [ln for ln in data if "[FEASIBLE]" not in ln]
    assert "30.0" in unmarked[0]  # the infeasible row (latency 30) does not
