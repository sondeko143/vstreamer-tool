"""Measure what the runtime's import paths pull in, and keep the gate's baseline current.

ADR-0085 moved the protection of the runtime's weight off a list of package names and onto
the outcome: what running the runtime actually loads, and what it costs in resident memory.
ADR-0087 made that the *only* protection -- there is no list of forbidden names anywhere in
this repo any more -- and in exchange requires this measurement to cover the paths that
actually get heavy, not just the entry point. This module is that measurement;
`tests/test_runtime_footprint.py` reads it once per session, and a maintainer regenerates
the recorded baseline with `REGENERATE_COMMAND` below. Note the absence of a `--` separator
in it: poe forwards `--` to the task literally, so the form with one is rejected by argparse
(measured: exit 2, "unrecognized arguments"). `tests/test_runtime_footprint.py` pins that,
because this command is the one instruction a maintainer receives at the moment the gate
fires.

**Names appear here only as paths to measure and as modules a path must reach.** Nothing in
this file forbids a package (ADR-0087); a package's weight is what gets judged, wherever it
came from.

`MEASURED_PATHS` below is the load-bearing part. `vspeech.main` on its own is a poor gate:
every worker is imported lazily inside `vspeech_coro` behind `config.<section>.enable`, and
several workers defer their heaviest import one step further into a function body. The
worst case is the transcription worker, which defers `faster_whisper` -- and therefore
`ctranslate2`, which picks up a merely *installed* torch through its own
`try: import torch` -- into `get_transcriber()`. Measuring only the entry point leaves that
invisible: the entry point stays light while every running pipeline gets heavy. So each
path here names the module *chain a running worker actually reaches*, deferred imports
included, and `MeasuredPath.reaches` records what that chain must load for the coverage
claim to hold; `tests/test_runtime_footprint.py` checks it, so a path that quietly stops
loading the heavy thing fails instead of looking like coverage.

**What is deliberately out of scope**: anything that needs a GPU, model assets on disk or a
config file. Those cannot be measured in the default suite, and `scripts/vc_pipeline_footprint.py`
(`uv run poe vc-footprint`) measures a real, fully warmed pipeline process on demand for
exactly that reason. What is measured here is import weight -- which is where a returning
dependency shows up first, and by far the largest share of what ADR-0080 removed.

Startup time is measured and printed, and is deliberately **not** part of any verdict:
ADR-0085 rejected it on measured grounds (the same suite on identical code took 30.45s /
113.70s / 35.40s on this machine).
"""

import argparse
import json
import os
import subprocess  # nosec B404 - spawning a pristine interpreter *is* the measurement
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tests" / "runtime_startup_baseline.json"


@dataclass(frozen=True)
class MeasuredPath:
    """One import chain the gate measures, and the claim it is there to back.

    `imports` is what a child process imports, in order. `reaches` is the top-level names
    that chain has to end up loading -- the evidence that the path really exercises what
    `covers` says it does. A path whose heavy dependency moved elsewhere would otherwise go
    on passing while measuring nothing, which is worse than not measuring it.
    """

    name: str
    imports: tuple[str, ...]
    covers: str
    reaches: tuple[str, ...]


# Every path the gate measures. To add a worker or a startup path, add it here and
# re-record; the rule for what belongs is mechanical -- a chain a running pipeline
# executes. `covers` is prose for whoever reads a failure; `reaches` is checked.
MEASURED_PATHS: tuple[MeasuredPath, ...] = (
    MeasuredPath(
        name="entry_point",
        imports=("vspeech.main",),
        covers=(
            "What `python -m vspeech` loads before it has read a config file: click, the "
            "config schema, the logger, telemetry, preflight, the shared context and the "
            "two infrastructure workers (sender/receiver) with their gRPC transport. Every "
            "pipeline pays this whichever workers it enables."
        ),
        reaches=("click", "pydantic", "grpc", "google"),
    ),
    MeasuredPath(
        name="transcription",
        imports=(
            "vspeech.worker.transcription",
            "av",
            "faster_whisper",
            "onnxruntime",
        ),
        covers=(
            "The transcription worker as it actually runs, deferred imports included: `av` "
            "(pcm_to_waveform), `faster_whisper` (get_transcriber) and `onnxruntime` (the "
            "VAD gate, ADR-0037). This is the heaviest path in the project and the one the "
            "entry-point measurement could not see -- faster_whisper imports ctranslate2, "
            "whose own `try: import torch` loads a merely installed torch, so ADR-0080's "
            "477MB comes back here first and nowhere else."
        ),
        reaches=("av", "faster_whisper", "ctranslate2", "onnxruntime", "numpy"),
    ),
    MeasuredPath(
        name="vc",
        imports=(
            "vspeech.worker.vc",
            "vspeech.lib.rvc",
            "vspeech.lib.onnx_session",
            "vspeech.lib.cuda_util",
            "vspeech.lib.pitch_extract",
        ),
        covers=(
            "The RVC voice-changer worker with the modules `rvc_worker` defers into its "
            "function body: the ONNX session factory, the device layer, the conversion "
            "path and the f0 extractor. This is where ADR-0080/0081/0082 took torch and "
            "torchaudio out and left numpy plus onnxruntime."
        ),
        reaches=("onnxruntime", "numpy"),
    ),
    MeasuredPath(
        name="audio_devices",
        imports=("vspeech.worker.recording", "vspeech.worker.playback"),
        covers=(
            "The two device-facing workers and the sounddevice binding they open PortAudio "
            "through (ADR-0031), plus the in-house resampler (ADR-0073..0077)."
        ),
        reaches=("sounddevice", "numpy"),
    ),
    MeasuredPath(
        name="tts",
        imports=(
            "vspeech.worker.tts",
            "vspeech.lib.voicevox",
            "vspeech.lib.voiceroid",
        ),
        covers=(
            "The TTS worker and both back ends it dispatches to by worker_type: "
            "voicevox_core and the VOICEROID2 binding. Neither is on the entry point."
        ),
        reaches=("voicevox_core", "pyvcroid2"),
    ),
    MeasuredPath(
        name="subtitle_obs",
        imports=("vspeech.worker.subtitle", "vspeech.worker.subtitle_obs"),
        covers=(
            "The subtitle worker's OBS back end, which ADR-0040 split out so an OBS "
            "pipeline runs headless. Measuring it apart from the TK back end is what keeps "
            "that headless claim honest: a GUI toolkit arriving on this path shows up as "
            "both a new top-level name and a jump in resident memory."
        ),
        reaches=("websockets",),
    ),
    MeasuredPath(
        name="subtitle_tk",
        imports=("vspeech.worker.subtitle", "vspeech.worker.subtitle_tk"),
        covers="The subtitle worker's TK back end, the only path that opens a GUI toolkit.",
        reaches=("tkinter",),
    ),
    MeasuredPath(
        name="translation",
        imports=("vspeech.worker.translation",),
        covers=(
            "The translation worker and the Google Cloud Translate client stack it imports "
            "at module level."
        ),
        reaches=("google", "grpc"),
    ),
    MeasuredPath(
        name="stream_vc_producer",
        imports=(
            "vspeech.stream_vc.subsystem",
            "vspeech.stream_vc.capture",
            "vspeech.stream_vc.runner",
            "vspeech.stream_vc.transport",
            "vspeech.lib.stream_vc",
        ),
        covers=(
            "The streaming-VC producer role (ADR-0055): capture, the conversion runner and "
            "the UDP transport, i.e. the modules `create_stream_vc_task` defers to at "
            "startup."
        ),
        reaches=("sounddevice", "numpy"),
    ),
    MeasuredPath(
        name="stream_vc_consumer",
        imports=(
            "vspeech.stream_vc.consumer",
            "vspeech.stream_vc.udp",
            "vspeech.stream_vc.jitter",
            "vspeech.stream_vc.wire",
        ),
        covers=(
            "The streaming-VC consumer role (ADR-0055). The whole point of the role split "
            "is that a playback-only host stays light: it receives converted audio over "
            "UDP and plays it, and must never carry the conversion stack's weight. This "
            "path used to be asserted torch-free by name; ADR-0087 keeps the invariant and "
            "measures its cost instead."
        ),
        reaches=("sounddevice", "numpy"),
    ),
    MeasuredPath(
        name="device_layer",
        imports=(
            "vspeech.lib.cuda_driver",
            "vspeech.lib.cuda_util",
            "vspeech.lib.onnx_session",
        ),
        covers=(
            "Resolving a GPU and opening an ONNX session (ADR-0078/0024). Deciding which "
            "device to use must not drag an inference framework in for one integer: while "
            "this layer spoke `torch.device`, the whisper pipeline paid 477MB of RSS and "
            "3.2s of startup for it. This path used to be asserted torch-free by name; "
            "ADR-0087 keeps the invariant and measures its cost instead."
        ),
        reaches=("onnxruntime", "numpy"),
    ),
)

PATHS_BY_NAME = {path.name: path for path in MEASURED_PATHS}

# The path the resident-memory calibration is taken on. It is the lightest one, and the
# only one `CALIBRATION_MODULE` is absent from, so it is the one where adding that module
# measures a marginal cost rather than nothing.
CALIBRATION_PATH = "entry_point"

# The one instruction a maintainer gets when the gate fires, so it is defined once and
# every message quotes this. No `--` separator: poe passes it through to the task
# verbatim and argparse then rejects the whole tail.
REGENERATE_COMMAND = "uv run poe runtime-baseline --update --runs 10"

# `--update` below this many runs is refused. A budget is an upper bound anchored on the
# worst run seen, so too few runs is not merely a weak measurement -- it writes a *tighter*
# budget than the code deserves and the gate starts flapping. Measured: an N=2 re-record of
# an unchanged runtime moved the resident-memory budget from 64.0 to 63.0 MiB.
MIN_RUNS_FOR_UPDATE = 10

# Modules that record how this venv was provisioned rather than what the runtime depends
# on: setuptools' distutils shim (setuptools is in uv.lock only as a transitive edge of
# ctranslate2, declared by nothing) and virtualenv's own patch module. Both are injected by
# `.pth` files at site initialisation, so they would come and go with packaging plumbing
# and fail the staleness check for a reason that has nothing to do with runtime weight.
PROVISIONING_ARTIFACTS = frozenset({"_distutils_hack", "_virtualenv"})

# How the recorded budgets are derived from a measurement. Both live here rather than in
# the JSON so that there is one place to change the rule; `--update` bakes the resulting
# numbers *and* these values into the JSON's prose, so the two cannot drift apart. The same
# two constants apply to every path, which is what lets one calibration speak for all of
# them: a budget is always `worst observed run + this much`, so what a budget catches is
# the constant, not the level it happens to sit at.
#
# What pydantic_settings adds to the *entry point* path, which is the signal ADR-0085
# requires the module indicator to catch. Measured with N=10 child processes each way:
# `vspeech.main` 716 modules and `vspeech.main` + `pydantic_settings` 747, zero spread on
# either side (2026-08-12). ADR-0066's older "about 32 modules" is the same measurement to
# within one module; 31 is what this repo reads today.
PYDANTIC_SETTINGS_MODULE_SIGNAL = 31
# Module-count slack: it has to stay **below** that signal, or the arrival of
# pydantic_settings would fit inside the budget. 16 clears that -- though only just, at
# slightly over half of 31, not under half -- and the gate fires with 15 modules to spare
# (747 against a budget of 732). It still absorbs the submodule shuffle of a routine
# dependency upgrade, and it is comfortably above the largest per-path run-to-run module
# spread measured across MEASURED_PATHS (recorded in each path's basis).
MODULE_COUNT_SLACK = 16
# Resident-memory headroom, in MiB. Sized against the measured run-to-run spread, not as a
# percentage, and deliberately not sized to absorb *growth*: any package arriving on a path
# trips that path's top-level check and forces a re-record anyway, so this budget only has
# to survive measurement noise. The first draft took 7.0 MiB and the calibration below
# caught numpy by 0.14 MiB -- a budget that only just works is a budget that does not work.
RSS_HEADROOM_MIB = 4.0

# Put on the calibration path once per `--update`, to record what the resident-memory
# budgets are actually proved to catch. numpy is the lightest heavy native dependency this
# project installs at all, so it is the hardest realistic case. It lives in the whisper/rvc
# extras; when it is absent the calibration is recorded as not measured, which
# `tests/test_runtime_footprint.py` treats as a failure of the *record* rather than of the
# environment reading it.
CALIBRATION_MODULE = "numpy"

# The child's payload. The preamble before the snapshot is `sys` and `time` only -- both
# are already in sys.modules before the interpreter hands over -- so the snapshot is the
# measured path's own footprint and nothing else. The memory read is deferred past the
# snapshot for the same reason, and reads only modules the path already loaded.
_PROBE_TEMPLATE = '''\
import sys
from time import perf_counter

_started = perf_counter()
__IMPORTS__
_seconds = perf_counter() - _started
_modules = sorted(sys.modules)

import ctypes
import json
from ctypes import wintypes


class _Counters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX (psapi.h).

    Spelled out again rather than imported from scripts/vc_pipeline_footprint.py, which
    has the same struct: importing that module here would pull argparse, statistics and
    the rest of its imports into the process being measured. Everything this payload
    touches after the snapshot has to be something the measured path already loaded.
    """

    _fields_ = (
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    )


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.GetCurrentProcess.argtypes = ()
_kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
_kernel32.K32GetProcessMemoryInfo.argtypes = (
    wintypes.HANDLE,
    ctypes.POINTER(_Counters),
    wintypes.DWORD,
)
_counters = _Counters()
_counters.cb = ctypes.sizeof(_counters)
if not _kernel32.K32GetProcessMemoryInfo(
    _kernel32.GetCurrentProcess(), ctypes.byref(_counters), _counters.cb
):
    raise SystemExit("GetProcessMemoryInfo failed: %d" % ctypes.get_last_error())

_mib = 1024.0 * 1024.0
print(
    json.dumps(
        {
            "modules": _modules,
            "working_set_mib": _counters.WorkingSetSize / _mib,
            "private_mib": _counters.PrivateUsage / _mib,
            "seconds": _seconds,
        }
    )
)
'''


@dataclass(frozen=True)
class Measurement:
    """One child process's import footprint.

    `modules` has already had `PROVISIONING_ARTIFACTS` filtered out, so both indicators
    read the same population.
    """

    modules: tuple[str, ...]
    working_set_mib: float
    private_mib: float
    seconds: float

    @property
    def top_level(self) -> frozenset[str]:
        """The distinct top-level names behind `modules`.

        Gating on top-level names rather than on all the entries is what keeps the gate
        from firing on a routine dependency upgrade that merely reorganises its own
        submodules. Anything genuinely *new* on the path still shows up here, because a
        package cannot arrive without its top-level name arriving; sub-tree growth inside
        an already-present package is what the module *count* budget is for.
        """
        return frozenset(name.split(".")[0] for name in self.modules)


def _child_environment() -> dict[str, str]:
    """The parent's environment minus anything that would import into the child.

    coverage's `a1_coverage.pth` starts a collector in any interpreter launched with
    `COVERAGE_PROCESS_START` set, which loads coverage and its dependencies before the
    snapshot is taken (measured: 718 modules -> 779). pytest-cov does not export it, but a
    developer running coverage by hand does, and a gate that depends on how the suite was
    invoked is a gate that flaps.
    """
    env = dict(os.environ)
    for key in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG"):
        env.pop(key, None)
    return env


def measure_startup(entry_points: tuple[str, ...]) -> Measurement:
    """Import `entry_points` in a pristine child process and report what that cost.

    A child rather than an in-process check because `sys.modules` inside the test process
    is contaminated by whatever ran before, so an in-process reading answers a different
    question every time.

    `-E -s` makes the reading independent of the developer's `PYTHON*` variables and user
    site-packages. `sys.path[0]` is deliberately left alone (i.e. no `-I`) so the check
    still works in a checkout where the project has not been installed into the venv.

    `entry_points` is a tuple both because a measured path is usually several modules and
    so a test can measure a path with something extra on it, which is how the failure
    message is proved to name a newcomer.
    """
    imports = "\n".join(f"import {name}" for name in entry_points)
    probe = _PROBE_TEMPLATE.replace("__IMPORTS__", imports)
    result = subprocess.run(  # nosec B603 - fixed argv built here, no shell
        [sys.executable, "-E", "-s", "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        # The payload prints ASCII-only JSON, but a failing child can put Japanese on
        # stderr; `replace` keeps a decoding error from masking the real failure.
        errors="replace",
        env=_child_environment(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"startup probe failed ({result.returncode}) for "
            f"{', '.join(entry_points)}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout)
    modules = tuple(
        name
        for name in payload["modules"]
        if name.split(".")[0] not in PROVISIONING_ARTIFACTS
    )
    return Measurement(
        modules=modules,
        working_set_mib=payload["working_set_mib"],
        private_mib=payload["private_mib"],
        seconds=payload["seconds"],
    )


def measure_path(path: MeasuredPath) -> Measurement:
    return measure_startup(path.imports)


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measure_calibration() -> Measurement | None:
    """The calibration path with `CALIBRATION_MODULE` on it, or None if it is absent."""
    calibrated = PATHS_BY_NAME[CALIBRATION_PATH].imports + (CALIBRATION_MODULE,)
    try:
        return measure_startup(calibrated)
    except RuntimeError:
        return None


def _calibration_sentence(calibration: Measurement | None, rss_budget: float) -> str:
    """What the budgets are *shown* to catch, rather than what they are hoped to catch."""
    if calibration is None:
        return (
            f"What the budgets catch was not calibrated on this run: "
            f"{CALIBRATION_MODULE} is not installed here, so re-record in an environment "
            "that has it."
        )
    over = calibration.working_set_mib - rss_budget
    verdict = f"{over:.2f} MiB over" if over > 0 else f"{-over:.2f} MiB UNDER"
    return (
        f"Calibrated on the {CALIBRATION_PATH} path, the lightest one and the only one "
        f"{CALIBRATION_MODULE} is absent from: putting {CALIBRATION_MODULE} on it takes "
        f"the working set to {calibration.working_set_mib:.2f} MiB, {verdict} that path's "
        f"budget of {rss_budget:.2f} MiB. {CALIBRATION_MODULE} is the lightest heavy "
        "native dependency this project installs at all, so it is the hardest realistic "
        "case. One calibration speaks for every path because every budget is "
        f"`worst observed run + {RSS_HEADROOM_MIB:.1f} MiB` rounded up: what a budget "
        "catches is that headroom, which is the same everywhere, not the level the budget "
        "sits at. If this margin ever reads UNDER, the headroom has stopped guarding "
        "anything and must be tightened; tests/test_runtime_footprint.py fails on a "
        "record that says so."
    )


def _calibration_record(calibration: Measurement | None, rss_budget: float) -> dict:
    if calibration is None:
        return {
            "path": CALIBRATION_PATH,
            "module": CALIBRATION_MODULE,
            "measured": False,
        }
    return {
        "path": CALIBRATION_PATH,
        "module": CALIBRATION_MODULE,
        "measured": True,
        "working_set_mib": round(calibration.working_set_mib, 2),
        "over_budget_mib": round(calibration.working_set_mib - rss_budget, 2),
        "module_count": len(calibration.modules),
    }


def _build_path_record(path: MeasuredPath, runs: list[Measurement]) -> dict:
    """Turn N measurements of one path into its record, budgets and their basis.

    The basis text is generated from the same numbers and constants the budgets are
    computed from, so it cannot describe a rule other than the one in force.
    """
    counts = sorted(len(m.modules) for m in runs)
    working = sorted(m.working_set_mib for m in runs)
    n = len(runs)
    # The gate is an upper bound, so the budget is anchored on the worst run, not the
    # median: anchoring on a middling value would leave half the population above it.
    count_budget = counts[-1] + MODULE_COUNT_SLACK
    rss_budget = float(ceil(working[-1] + RSS_HEADROOM_MIB))
    spread = working[-1] - working[0]
    spread_pct = 100.0 * spread / median(working)
    headroom = rss_budget - working[-1]
    # A single run has no spread to compare the headroom against; say so rather than
    # divide by zero or quote a ratio the measurement cannot support.
    versus_spread = (
        f"about {headroom / spread:.0f}x the observed spread"
        if spread
        else "with no run-to-run spread measured to compare it against"
    )

    # Every run has to agree, and the union is not good enough: the gate compares a
    # *single* run against this list, so a name seen in only some runs would be recorded
    # and then read as a stale entry on the runs that lack it. Recording the intersection
    # instead just moves the flap to the additions check. Any disagreement at all means
    # the gate would flap, so refuse to record one rather than bake it in.
    seen = [measurement.top_level for measurement in runs]
    top_level = frozenset.union(*seen)
    unstable = sorted(top_level - frozenset.intersection(*seen))
    if unstable:
        raise SystemExit(
            f"経路 {path.name} のモジュール集合が実行ごとに揺れているので"
            "基準データを記録できません。"
            f"全 {n} 回に現れなかった名前: {', '.join(unstable)}\n"
            "このまま記録するとゲートが暴れます。揺れの原因 (子プロセスへ漏れている "
            "環境変数、条件付き import など) を先に潰してください。"
        )

    missing = sorted(name for name in path.reaches if name not in top_level)
    if missing:
        raise SystemExit(
            f"経路 {path.name} が読み込むはずのモジュールを読み込んでいません: "
            f"{', '.join(missing)}\n"
            "この経路は " + path.covers + "\n"
            "重い依存を通らない経路を記録すると、測っていないのに測ったように見えます。"
            "import 連鎖が動いたのなら MEASURED_PATHS の imports/reaches を直してください。"
        )

    count_spread = counts[-1] - counts[0]
    return {
        "imports": list(path.imports),
        "covers": path.covers,
        "reaches": list(path.reaches),
        "module_count": {
            "observed_max": counts[-1],
            "observed_min": counts[0],
            "runs": n,
            "budget": count_budget,
            "basis": (
                f"len(sys.modules) after importing {', '.join(path.imports)}. N={n} "
                f"consecutive runs: min {counts[0]}, max {counts[-1]} (spread "
                f"{count_spread}). Budget = max + {MODULE_COUNT_SLACK}. ADR-0085 requires "
                "this indicator to catch pydantic_settings, measured at "
                f"+{PYDANTIC_SETTINGS_MODULE_SIGNAL} modules on the entry point (N=10), so "
                "the slack is held below that signal -- slightly over half of it, not "
                "under -- while still absorbing the submodule shuffle of a routine "
                "dependency upgrade."
            ),
        },
        "resident_memory_mib": {
            "observed_max": round(working[-1], 2),
            "observed_min": round(working[0], 2),
            "observed_median": round(median(working), 2),
            "runs": n,
            "budget": rss_budget,
            "basis": (
                "Working set (GetProcessMemoryInfo) of the child itself after importing "
                f"{', '.join(path.imports)}. N={n} consecutive runs: min {working[0]:.2f}, "
                f"median {median(working):.2f}, max {working[-1]:.2f} MiB -- spread "
                f"{spread:.2f} MiB ({spread_pct:.1f}% of the median). Budget = max + "
                f"{RSS_HEADROOM_MIB:.1f} MiB headroom rounded up, i.e. {headroom:.2f} MiB "
                f"of headroom, {versus_spread}, so it does not flap. What this indicator "
                "deliberately does not catch is pydantic_settings' ~1.5 MiB of cost unique "
                "to the entry point (N=10; ADR-0085) -- the module indicators cover that, "
                "which is why there are two."
            ),
        },
        "top_level_modules": sorted(top_level),
    }


def _build_baseline(
    runs_by_path: dict[str, list[Measurement]], calibration: Measurement | None
) -> dict:
    """Turn the per-path measurements into the recorded baseline."""
    paths = {
        path.name: _build_path_record(path, runs_by_path[path.name])
        for path in MEASURED_PATHS
        if path.name in runs_by_path
    }
    calibration_budget = paths[CALIBRATION_PATH]["resident_memory_mib"]["budget"]
    return {
        "what_this_is": (
            "The outcome gate of ADR-0085, widened to the paths that actually get heavy by "
            "ADR-0087: what importing each of the runtime's real import chains in a "
            "pristine child process loads, and what it costs in resident memory. Read by "
            "tests/test_runtime_footprint.py. Regenerate, do not hand-edit."
        ),
        "regenerate_with": REGENERATE_COMMAND,
        "budget_rule": {
            "module_count_slack": MODULE_COUNT_SLACK,
            "rss_headroom_mib": RSS_HEADROOM_MIB,
            "basis": _calibration_sentence(calibration, calibration_budget),
            "calibration": _calibration_record(calibration, calibration_budget),
        },
        "paths": paths,
    }


def _write_baseline(baseline: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a factory so a test can check `REGENERATE_COMMAND` really parses."""
    parser = argparse.ArgumentParser(
        description="ランタイムの各 import 経路が持ち込むモジュール集合と常駐メモリを"
        "実測する（ADR-0085/0087 の成果ゲートの基準データ）"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=MIN_RUNS_FOR_UPDATE,
        help=f"測定回数。1 回は測定ではない。--update には {MIN_RUNS_FOR_UPDATE} 回以上が要る",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=f"実測値で {BASELINE_PATH.name} を書き直す（差分を見て承認すること）",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(PATHS_BY_NAME),
        help="この経路だけ測る（調査用。--update とは併用できない）",
    )
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    parser = build_parser()
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs は 1 以上")
    # 予算は「見た中で最悪の run」に張り付くので、回数が少ないほど予算は**狭く**なる。
    # 実測: 変更していないランタイムを N=2 で採り直すと常駐メモリの予算が 64.0 -> 63.0MiB
    # に締まった。そのまま記録すればゲートが暴れる。
    if args.update and args.runs < MIN_RUNS_FOR_UPDATE:
        parser.error(
            f"--update には --runs {MIN_RUNS_FOR_UPDATE} 以上が要ります"
            f"（指定は {args.runs}）。回数が足りないと、実際より狭い予算を"
            "書き込んでしまいます"
        )
    if args.update and args.only:
        parser.error(
            "--only は調査用です。一部の経路だけで記録すると、残りの経路が基準データから"
            "消えてゲートが素通りになります"
        )

    selected = [
        path for path in MEASURED_PATHS if not args.only or path.name in args.only
    ]
    runs_by_path: dict[str, list[Measurement]] = {}
    for path in selected:
        runs: list[Measurement] = []
        for i in range(args.runs):
            measurement = measure_path(path)
            runs.append(measurement)
            print(
                f"{path.name} run {i + 1}: modules={len(measurement.modules)} "
                f"top_level={len(measurement.top_level)} "
                f"ws={measurement.working_set_mib:.2f}MiB "
                f"private={measurement.private_mib:.2f}MiB "
                f"import={measurement.seconds:.2f}s (not gated)"
            )
        runs_by_path[path.name] = runs
        module_sets = [frozenset(m.modules) for m in runs]
        unstable = frozenset.union(*module_sets) - frozenset.intersection(*module_sets)
        working = sorted(m.working_set_mib for m in runs)
        counts = sorted(len(m.modules) for m in runs)
        print(
            f"  N={len(runs)}: modules {counts[0]}..{counts[-1]} "
            f"(unstable across runs: {len(unstable)}) "
            f"ws min/median/max {working[0]:.2f}/{median(working):.2f}/"
            f"{working[-1]:.2f}MiB\n"
        )
        if unstable:
            print("  unstable module names: " + ", ".join(sorted(unstable)))

    calibration = measure_calibration() if CALIBRATION_PATH in runs_by_path else None
    baseline = _build_baseline(runs_by_path, calibration)
    for name, record in baseline["paths"].items():
        print(
            f"{name}: module budget {record['module_count']['budget']}, "
            f"rss budget {record['resident_memory_mib']['budget']}MiB"
        )
    print("calibration: " + json.dumps(baseline["budget_rule"]["calibration"]))
    if args.update:
        _write_baseline(baseline)
        print(f"wrote {BASELINE_PATH.relative_to(REPO_ROOT)}")
    else:
        print("(--update to record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
