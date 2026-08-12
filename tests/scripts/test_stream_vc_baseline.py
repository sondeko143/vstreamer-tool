"""Tests for the Stream VC baseline harness's pure parts.

The GPU half (capture / compare) needs real RVC assets and is exercised on hardware;
what is pinned here is the judgement, because that is what decides whether the torch
removal is allowed to land.
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.stream_vc_baseline import CONFIG_ENV
from scripts.stream_vc_baseline import _redact_home
from scripts.stream_vc_baseline import check_cuda_libraries_are_identified
from scripts.stream_vc_baseline import check_self_noise
from scripts.stream_vc_baseline import classify_cuda_library
from scripts.stream_vc_baseline import cuda_library_suppliers
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


def _call_lines(function_name: str, call_name: str) -> list[int]:
    """Source lines where `function_name`'s body calls `call_name`.

    `capture` and `compare` need a GPU and real RVC assets, so no test in this file can
    execute them; the guards below are therefore verified as functions, and their
    *wiring* is verified here from the source. Crude, but it is the wiring that failed
    before -- the self-noise verdict was computed, printed and then not acted on -- and an
    unwired guard passes every one of its own unit tests.
    """
    import scripts.stream_vc_baseline as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    )


def test_capture_runs_both_guards_before_it_writes_anything():
    """An artifact that fails either guard must never reach the disk.

    Once written it is indistinguishable from a good one, and the next `compare` trusts
    it without asking.
    """
    write = _call_lines("capture", "savez")
    assert write, "capture no longer writes an npz; this test needs rewriting"
    for guard in ("check_self_noise", "check_cuda_libraries_are_identified"):
        called = _call_lines("capture", guard)
        assert called, f"capture no longer calls {guard}"
        assert called[0] < write[0], f"{guard} runs after the artifact is written"


def test_compare_refuses_to_judge_against_an_unidentified_supplier():
    assert _call_lines("compare", "check_cuda_libraries_are_identified")


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


def test_capture_refuses_to_write_a_reference_that_is_not_reproducible():
    """The self-noise number is a verdict, not a printout.

    `capture` used to compute it, print it, store it and return 0 regardless, so a
    non-reproducible run would have become the reference every later `compare` trusts.
    """
    ref = _reference()
    test = ref.copy()
    test[2, 5] += 1
    with pytest.raises(SystemExit) as exc:
        check_self_noise(judge(ref, test), "ort")
    assert "bit 一致しません" in str(exc.value)


def test_a_reproducible_capture_passes_the_self_noise_gate():
    """The positive control: a gate that fires on everything is not a gate."""
    ref = _reference()
    check_self_noise(judge(ref, ref.copy()), "ort")


def test_seed_mode_none_cannot_produce_a_baseline():
    """It measures the unseeded spread; by construction that is not a reference.

    The message has to say so, because "not bit-exact" is the expected outcome there and
    would otherwise read as a defect.
    """
    ref = _reference()
    test = ref.copy()
    test[0, 0] += 7
    with pytest.raises(SystemExit) as exc:
        check_self_noise(judge(ref, test), "none")
    assert "--seed-mode ort" in str(exc.value)


def test_a_provenance_that_identified_no_cuda_supplier_is_refused():
    """Two empty supplier maps compare equal, i.e. "matches" having compared nothing.

    That is the defect the supplier record exists to prevent, reappearing one level down,
    so both the capture and the compare paths refuse an empty map instead of recording it.
    """
    with pytest.raises(SystemExit) as exc:
        check_cuda_libraries_are_identified(
            {"cuda_libraries": {"suppliers": {}, "versions": {}}}, "この採取"
        )
    assert "この採取" in str(exc.value)


def test_a_provenance_missing_the_section_entirely_is_refused():
    """An artifact predating the record is exactly the one whose supplier is unknown."""
    with pytest.raises(SystemExit):
        check_cuda_libraries_are_identified({"rvc": {}}, "この実行")


def test_a_provenance_that_named_a_supplier_passes():
    check_cuda_libraries_are_identified(
        {"cuda_libraries": {"suppliers": _NVIDIA_LIBS, "versions": _NVIDIA_VERSIONS}},
        "この実行",
    )


def test_a_home_relative_config_path_is_recorded_unchanged():
    """`~/...` is what this project's configs use, so existing baselines stay comparable."""
    assert _redact_home("~/.config/vstreamer/rvc/model.onnx") == (
        "~/.config/vstreamer/rvc/model.onnx"
    )
    assert _redact_home("./models/model.onnx") == "./models/model.onnx"


def test_an_absolute_path_under_home_is_collapsed_to_a_tilde():
    """`C:\\Users\\<name>\\...` is the environment PII the scanning gate keeps out."""
    absolute = str(Path.home() / ".config" / "vstreamer" / "model.onnx")
    redacted = _redact_home(absolute)
    assert redacted == "~/.config/vstreamer/model.onnx"
    assert str(Path.home()) not in redacted


def test_an_absolute_path_outside_home_becomes_a_digest():
    """A UNC share names a host; a digest discriminates without naming anything."""
    a = _redact_home(r"\\nas-host\share\rvc\model.onnx")
    b = _redact_home(r"\\nas-host\share\rvc\other.onnx")
    assert a.startswith("sha256:")
    assert "nas-host" not in a
    # Still a discriminator: two different paths must not collapse to one record.
    assert a != b


def test_provenance_records_no_absolute_config_path():
    """The end-to-end statement: nothing under the user's home reaches the artifact."""
    home = Path.home()
    prov = provenance(
        _config(model_file=home / "rvc" / "model.onnx"), target_sample_rate=48000
    )
    assert str(home) not in json.dumps(prov)
    assert prov["rvc"]["model_file"] == "~/rvc/model.onnx"


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


def test_provenance_records_where_the_cuda_libraries_came_from():
    """The supplier is part of what decides the samples, so it is part of provenance.

    ADR-0083 moved cuBLAS/cuDNN from torch's `lib` to the `nvidia-*` wheels, which
    changed the emitted waveform while every other recorded field stayed identical --
    so the first baseline went on comparing and reported the difference as a code
    regression. Both halves are recorded: which package supplied each loaded library,
    and at which version, since a wheel bump changes the kernels just as a supplier
    swap does.
    """
    prov = provenance(_config(), 48000)
    assert set(prov["cuda_libraries"]) == {"suppliers", "versions"}
    assert isinstance(prov["cuda_libraries"]["suppliers"], dict)
    assert isinstance(prov["cuda_libraries"]["versions"], dict)
    # Only the shape is asserted here, never the values: this is the one test that reads
    # the live process, and which CUDA libraries are loaded depends on what ran earlier
    # in the session. The tokens are a closed set, so that much is safe to pin.
    assert set(prov["cuda_libraries"]["suppliers"].values()) <= {
        "torch",
        "nvidia-wheel",
        "system",
    }


def test_provenance_ignores_which_gpu_was_used():
    """The same baseline must stay checkable on another card."""
    a = provenance(_config(gpu_id=0, gpu_name="RTX 4060"), 48000)
    b = provenance(_config(gpu_id=1, gpu_name="RTX 5060"), 48000)
    assert a == b
    assert provenance_mismatches(a, b) == []


def test_classify_cuda_library_names_the_supplying_package():
    """Only the supplier token, never the path (it is machine-specific, and PII)."""
    venv = r"C:\proj\.venv\Lib\site-packages"
    assert classify_cuda_library(rf"{venv}\torch\lib\cublasLt64_13.dll") == "torch"
    assert (
        classify_cuda_library(rf"{venv}\nvidia\cu13\bin\x86_64\cublasLt64_13.dll")
        == "nvidia-wheel"
    )
    assert (
        classify_cuda_library(rf"{venv}\nvidia\cudnn\bin\cudnn64_9.dll")
        == "nvidia-wheel"
    )
    assert (
        classify_cuda_library(r"C:\Program Files\NVIDIA\CUDA\v13.3\bin\cublas64_13.dll")
        == "system"
    )


def test_cuda_library_suppliers_picks_only_the_math_libraries():
    """The CUDA major is in the filename, so the match is by prefix, not exact name."""
    venv = r"C:\proj\.venv\Lib\site-packages"
    got = cuda_library_suppliers(
        [
            rf"{venv}\nvidia\cu13\bin\x86_64\cublasLt64_13.dll",
            rf"{venv}\nvidia\cu13\bin\x86_64\cudart64_13.dll",
            rf"{venv}\nvidia\cudnn\bin\cudnn64_9.dll",
            rf"{venv}\torch\lib\cufft64_12.dll",
            r"C:\Windows\System32\kernel32.dll",  # unrelated, must not appear
            rf"{venv}\onnxruntime\capi\onnxruntime_providers_cuda.dll",  # not a math lib
        ]
    )
    assert got == {
        "cublaslt64_13.dll": "nvidia-wheel",
        "cudart64_13.dll": "nvidia-wheel",
        "cudnn64_9.dll": "nvidia-wheel",
        "cufft64_12.dll": "torch",
    }


def _prov_with_cuda(suppliers: dict[str, str], versions: dict[str, str] | None = None):
    """A provenance record whose `cuda_libraries` section is chosen, not observed.

    `provenance()` reads the **live process's** loaded CUDA DLLs, which is right in
    production (it is called once, after the sessions are open) but makes any test that
    reasons about the recorded values depend on what ran earlier in the session. Injecting
    a value on top of a real reading is worse than useless: run alone nothing has loaded
    those DLLs and the injected value differs from the empty reading, but inside the full
    suite an earlier test has already pulled torch's CUDA libraries in, the injected
    "torch" then *matches reality*, and the assertion that a supplier change is reported
    silently stops testing anything. That is not a hypothesis -- it is how
    `test_provenance_mismatch_names_a_changed_cuda_supplier` came to pass alone and fail
    in the suite. So both sides are built here instead of read.
    """
    prov = provenance(_config(), 48000)
    prov["cuda_libraries"] = {
        "suppliers": dict(suppliers),
        "versions": dict(versions or {}),
    }
    return prov


_NVIDIA_LIBS = {"cublaslt64_13.dll": "nvidia-wheel", "cudnn64_9.dll": "nvidia-wheel"}
_NVIDIA_VERSIONS = {"nvidia-cublas": "13.6.0.2", "nvidia-cudnn-cu13": "9.24.0.43"}


def test_provenance_mismatch_is_silent_when_the_cuda_libraries_match():
    """The positive control: a gate that fires on everything is not a gate.

    Without this, every assertion below would still pass if `provenance_mismatches`
    started reporting the section unconditionally.
    """
    a = _prov_with_cuda(_NVIDIA_LIBS, _NVIDIA_VERSIONS)
    b = _prov_with_cuda(_NVIDIA_LIBS, _NVIDIA_VERSIONS)
    assert provenance_mismatches(a, b) == []


def test_provenance_mismatch_names_a_changed_cuda_supplier():
    """A supplier swap must stop the comparison, not be blamed on the code under test.

    The two sides differ in exactly one DLL's supplier, so `lines` collapsing to empty
    (the function no longer looking at the section at all) fails the length assertion.
    """
    a = _prov_with_cuda(_NVIDIA_LIBS, _NVIDIA_VERSIONS)
    b = _prov_with_cuda(
        {**_NVIDIA_LIBS, "cublaslt64_13.dll": "torch"}, _NVIDIA_VERSIONS
    )
    lines = provenance_mismatches(a, b)
    assert len(lines) == 1, lines
    assert "cuda_libraries.suppliers.cublaslt64_13.dll" in lines[0]
    assert "nvidia-wheel" in lines[0]
    assert "torch" in lines[0]


def test_provenance_mismatch_names_a_bumped_cuda_library_version():
    """Same supplier, newer wheel: the kernels can still change, so it must be caught."""
    a = _prov_with_cuda(_NVIDIA_LIBS, _NVIDIA_VERSIONS)
    b = _prov_with_cuda(_NVIDIA_LIBS, {**_NVIDIA_VERSIONS, "nvidia-cublas": "13.7.0.0"})
    lines = provenance_mismatches(a, b)
    assert len(lines) == 1, lines
    assert "cuda_libraries.versions.nvidia-cublas" in lines[0]
    assert "13.6.0.2" in lines[0]
    assert "13.7.0.0" in lines[0]


def test_provenance_mismatch_rejects_a_baseline_that_predates_the_supplier_record():
    """An npz captured before the supplier was recorded cannot be judged against.

    It is exactly the artifact whose supplier is unknown, which is the situation that
    caused the confusion in the first place -- so it must refuse rather than compare.
    """
    current = _prov_with_cuda(_NVIDIA_LIBS, _NVIDIA_VERSIONS)
    old = {k: v for k, v in current.items() if k != "cuda_libraries"}
    lines = provenance_mismatches(old, current)
    assert lines
    assert all(line.startswith("cuda_libraries") for line in lines), lines


def test_an_unrecorded_cuda_section_is_not_the_same_as_an_empty_one():
    """A section recorded as empty and a section never recorded are different things.

    Collapsing the two would let the pre-provenance artifact be judged against, so
    `_flatten` keeps an empty dict as a leaf. This stays pinned even though
    `check_cuda_libraries_are_identified` now refuses both at the capture and compare
    boundaries: `_flatten`'s behaviour is what makes the mismatch *message* name the
    section, and the two guards fail differently (one names the changed field, the other
    refuses to judge at all). Pinned synthetically because the live reading is only empty
    when nothing has loaded CUDA yet -- inside the full suite it is not, and this guard
    would then never be exercised.
    """
    empty = _prov_with_cuda({}, {})
    old = {k: v for k, v in empty.items() if k != "cuda_libraries"}
    assert provenance_mismatches(old, empty)


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


def test_seed_runtime_both_degrades_with_a_clear_message_when_torch_is_missing(
    monkeypatch,
):
    """`--seed-mode both` needs torch installed separately now that it has left the
    runtime (ADR-0081); denied it, `seed_runtime` must fail loud with an actionable
    message naming the alternative (`ort`), not let a bare ModuleNotFoundError
    traceback surface.
    """
    import builtins

    real_import = builtins.__import__

    def _deny_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _deny_torch)
    with pytest.raises(SystemExit, match="--seed-mode both needs torch"):
        seed_runtime(0, "both")


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
