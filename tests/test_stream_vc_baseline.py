"""Tests for the Stream VC baseline harness's pure parts.

The GPU half (capture / compare) needs real RVC assets and is exercised on hardware;
what is pinned here is the judgement, because that is what decides whether the torch
removal is allowed to land.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.stream_vc_baseline import CONFIG_ENV
from scripts.stream_vc_baseline import geometry
from scripts.stream_vc_baseline import judge
from scripts.stream_vc_baseline import latency_stats
from scripts.stream_vc_baseline import make_input_blocks
from scripts.stream_vc_baseline import one_lsb_snr_db
from scripts.stream_vc_baseline import provenance
from scripts.stream_vc_baseline import provenance_mismatches
from scripts.stream_vc_baseline import resolve_config
from scripts.stream_vc_baseline import seed_runtime


def _reference(n_blocks: int = 8, block_len: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-8000, 8000, size=(n_blocks, block_len), dtype=np.int16)


def test_make_input_blocks_shape_and_determinism():
    a = make_input_blocks(2560, 5)
    b = make_input_blocks(2560, 5)
    assert a.shape == (5, 2560)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    # Distinct blocks, i.e. the signal is not silently repeating one block.
    assert not np.array_equal(a[0], a[1])


def test_judge_identical_is_bit_exact_and_exits_zero():
    ref = _reference()
    verdict = judge(ref, ref.copy())
    assert verdict.bit_exact
    assert verdict.max_abs_diff == 0
    assert verdict.blocks_differing == 0
    assert verdict.outcome == "BIT_EXACT"
    assert verdict.exit_code == 0
    assert "BIT-EXACT" in verdict.report()


def test_judge_one_wrecked_block_fails():
    """The gate must not let a single destroyed block dilute away."""
    ref = _reference()
    test = ref.copy()
    test[3] = -ref[3]
    verdict = judge(ref, test)
    assert not verdict.bit_exact
    assert verdict.blocks_differing == 1
    assert verdict.outcome == "FAIL"
    assert verdict.exit_code == 1
    assert verdict.worst_block_correlation < 0.0
    assert "FAIL" in verdict.report()


def test_judge_fails_a_block_that_the_whole_stream_gate_would_dilute():
    """A defect small enough to pass the whole-stream pair must still fail per-block.

    With many blocks, one block's noise is a small share of the total energy, so the
    whole-stream SNR stays high while that block is audibly wrong. The per-block floor
    is what closes this, and this is the test that would go red if it were removed.
    """
    ref = _reference(n_blocks=200, block_len=64)
    test = ref.copy()
    rng = np.random.default_rng(1)
    # Corrupt one block to about 20 dB SNR. Spread over 200 blocks that is a whole-stream
    # SNR near 10*log10(200*100) = 43 dB, comfortably inside the 35 dB gate, while the
    # block itself sits 15 dB below it.
    block = ref[7].astype(np.float64)
    noise = rng.standard_normal(block.size) * np.sqrt(np.mean(block**2)) / 10.0
    test[7] = (block + noise).astype(np.int16)

    verdict = judge(ref, test)
    assert verdict.correlation >= verdict.corr_min
    assert verdict.snr_db >= verdict.snr_min_db
    assert verdict.worst_block_snr_db < verdict.snr_min_db
    assert verdict.worst_block_correlation < verdict.corr_min
    assert verdict.outcome == "FAIL"
    assert verdict.exit_code == 1


def test_judge_one_lsb_difference_is_tolerance_not_a_pass():
    ref = _reference()
    test = (ref.astype(np.int32) + 1).astype(np.int16)
    verdict = judge(ref, test)
    assert not verdict.bit_exact
    assert verdict.max_abs_diff == 1
    assert verdict.within_tolerance
    # Distinct from a pass: bit equality is the contract, so "close" must be its own
    # exit code rather than share 0.
    assert verdict.outcome == "TOLERANCE"
    assert verdict.exit_code == 2
    assert "要判断" in verdict.report()
    # The yardstick is what "1 LSB everywhere" scores, so the measured SNR of exactly
    # that difference must land on it.
    assert verdict.snr_db == pytest.approx(verdict.one_lsb_snr_db, abs=1e-9)


def test_judge_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        judge(_reference(), _reference(n_blocks=4))


def test_one_lsb_snr_db_is_twenty_log_amplitude():
    # A constant-amplitude reference: signal energy is n*v**2 and a 1-LSB error's is n,
    # so the ratio is v**2 -> 20*log10(v).
    ref = np.full((4, 16), 1000, dtype=np.int16)
    assert one_lsb_snr_db(ref) == pytest.approx(60.0)


def test_one_lsb_snr_db_on_silence_is_minus_inf():
    assert one_lsb_snr_db(np.zeros((2, 8), dtype=np.int16)) == float("-inf")


def test_latency_stats():
    stats = latency_stats(np.array([0.010, 0.020, 0.030, 0.040], dtype=np.float64))
    assert stats.n == 4
    assert stats.p50_ms == pytest.approx(25.0)
    assert stats.max_ms == pytest.approx(40.0)
    assert "N=4" in stats.line("latency")


def test_geometry_reports_the_window_defining_fields():
    from vspeech.config import StreamVcConfig

    config = StreamVcConfig(block_ms=160.0, context_ms=500.0, crossfade_ms=25.0)
    geo = geometry(config)
    assert geo["block_ms"] == 160.0
    assert geo["context_ms"] == 500.0
    assert geo["crossfade_ms"] == 25.0
    assert set(geo) == {
        "block_ms",
        "context_ms",
        "crossfade_ms",
        "sola_search_ms",
        "lookahead_ms",
    }


def _config(**overrides):
    from vspeech.config import RvcConfig
    from vspeech.config import StreamVcConfig

    return StreamVcConfig(rvc=RvcConfig(**overrides))


def test_provenance_records_the_models_and_their_parameters():
    geo = provenance(_config(f0_up_key=13), target_sample_rate=48000)
    assert geo["target_sample_rate"] == 48000
    assert geo["rvc"]["f0_up_key"] == 13
    assert "model_file" in geo["rvc"]
    assert "hubert_model_file" in geo["rvc"]
    assert "f0_extractor_type" in geo["rvc"]


def test_provenance_ignores_which_gpu_was_used():
    """The same baseline must stay checkable on another card."""
    a = provenance(_config(gpu_id=0, gpu_name="RTX 4060"), 48000)
    b = provenance(_config(gpu_id=1, gpu_name="RTX 5060"), 48000)
    assert a == b
    assert provenance_mismatches(a, b) == []


def test_provenance_mismatch_names_the_changed_field():
    a = provenance(_config(f0_up_key=13), 48000)
    b = provenance(_config(f0_up_key=0), 48000)
    lines = provenance_mismatches(a, b)
    assert len(lines) == 1
    assert "rvc.f0_up_key" in lines[0]
    assert "13" in lines[0]


def test_provenance_mismatch_catches_a_different_target_sample_rate():
    a = provenance(_config(), 48000)
    b = provenance(_config(), 40000)
    lines = provenance_mismatches(a, b)
    assert len(lines) == 1
    assert "target_sample_rate" in lines[0]


def test_provenance_survives_a_json_round_trip():
    """It is stored in the npz as JSON, so equality has to hold after that trip."""
    original = provenance(_config(f0_up_key=13), 48000)
    restored = json.loads(json.dumps(original, sort_keys=True, ensure_ascii=False))
    assert provenance_mismatches(original, restored) == []


def test_seed_runtime_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown seed mode"):
        seed_runtime(0, "torch-only")


def test_seed_runtime_none_touches_nothing():
    seed_runtime(0, "none")  # must not import or raise


def test_resolve_config_prefers_the_explicit_path(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "from-env.toml")
    assert resolve_config(Path("explicit.toml")) == Path("explicit.toml")


def test_resolve_config_falls_back_to_the_env_var(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "from-env.toml")
    assert resolve_config(None) == Path("from-env.toml")


def test_resolve_config_without_either_is_a_usage_error(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    with pytest.raises(SystemExit, match=CONFIG_ENV):
        resolve_config(None)


def test_footprint_reads_the_memory_of_an_exact_pid():
    from scripts.vc_pipeline_footprint import read_memory

    sample = read_memory(os.getpid())
    assert sample is not None
    assert sample.working_set_mb > 0.0
    assert sample.peak_working_set_mb >= sample.working_set_mb
    assert sample.private_mb > 0.0


def test_footprint_returns_none_for_a_dead_pid():
    from scripts.vc_pipeline_footprint import read_memory

    # A PID that cannot exist (Windows PIDs are multiples of 4 and bounded well below
    # this), so OpenProcess must fail rather than report someone else's memory.
    assert read_memory(0x7FFFFFFF) is None


def test_footprint_keeps_draining_stdout_after_the_ready_marker():
    """The pipe must stay drained past the marker, or "settling" measures a frozen process."""
    import io
    from threading import Event
    from time import perf_counter

    from scripts.vc_pipeline_footprint import PID_MARKER
    from scripts.vc_pipeline_footprint import _ChildOutput
    from scripts.vc_pipeline_footprint import drain_stdout

    stream = io.StringIO(
        f"{PID_MARKER} 4242\nloading\nvc worker started\nafter-1\nafter-2\n"
    )
    seen = _ChildOutput(t0=perf_counter())
    ready = Event()
    drain_stdout(stream, "vc worker started", seen, ready)

    assert ready.is_set()
    assert seen.child_pid == 4242
    assert seen.startup_s > 0.0
    assert "after-2" in seen.tail


def test_footprint_drain_signals_on_eof_without_the_marker():
    """A child that dies before the marker must wake the waiter, not burn the timeout."""
    import io
    from threading import Event
    from time import perf_counter

    from scripts.vc_pipeline_footprint import PID_MARKER
    from scripts.vc_pipeline_footprint import _ChildOutput
    from scripts.vc_pipeline_footprint import drain_stdout

    seen = _ChildOutput(t0=perf_counter())
    ready = Event()
    drain_stdout(io.StringIO(f"{PID_MARKER} 7\nboom\n"), "never", seen, ready)

    assert ready.is_set()
    # 0 is how the caller tells EOF apart from a real readiness signal.
    assert seen.startup_s == 0.0
    assert "boom" in seen.report()


def test_footprint_bootstrap_announces_its_own_pid():
    from scripts.vc_pipeline_footprint import BOOTSTRAP
    from scripts.vc_pipeline_footprint import PID_MARKER

    assert PID_MARKER in BOOTSTRAP
    assert "os.getpid()" in BOOTSTRAP
    assert "runpy.run_module('vspeech'" in BOOTSTRAP
