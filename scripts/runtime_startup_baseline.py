"""Measure what the runtime's startup pulls in, and keep the gate's baseline current.

ADR-0085 moves the protection of the runtime's weight off a list of package names and onto
the outcome: what starting the runtime actually loads, and what it costs in resident
memory. This module is the measurement behind that gate --
`tests/test_runtime_footprint.py` calls `measure_startup()` once per session, and a
maintainer regenerates the recorded baseline with `REGENERATE_COMMAND` below. Note the
absence of a `--` separator in it: poe forwards `--` to the task literally, so the form
with one is rejected by argparse (measured: exit 2, "unrecognized arguments").
`tests/test_runtime_footprint.py` pins that, because this command is the one instruction a
maintainer receives at the moment the gate fires.

**"The runtime's startup path" here means importing `vspeech.main`.** That is everything
`python -m vspeech` loads before it has read a config file: click, the config schema, the
logger, telemetry, preflight, the shared context, and the two infrastructure workers
(sender/receiver) with their gRPC transport. Every worker past that point is imported
lazily inside `vspeech_coro`, gated by `config.<section>.enable`, and needs a config file
plus model assets and often a GPU -- which ADR-0085 rules out of the default test suite.
So this slice is both the largest one measurable without any of that, and the one *every*
pipeline pays whichever workers it enables.

**What that excludes, by name, so nobody over-reads this gate: torch.** Nothing on this
path imports `ctranslate2` -- the transcription worker is imported lazily, and it defers
`faster_whisper` into a function body, so even `import vspeech.worker.transcription` loads
no ctranslate2 (measured, both). ctranslate2 is what grabs a merely *installed* torch, so
`uv add torch` leaves every check in `tests/test_runtime_footprint.py` green.
**[ADR-0084](../docs/adr/0084-dependency-table-torch-gate.md)'s dependency-table gate
remains the load-bearing guard for torch**, and this gate does not replace it. The same
goes for onnxruntime and every other dependency reached only from a lazily-imported
worker.

Startup time is measured and printed, and is deliberately **not** part of any verdict:
ADR-0085 rejected it on measured grounds (the same suite on identical code took 30.45s /
113.70s / 35.40s on this machine).

`scripts/vc_pipeline_footprint.py` measures a different thing -- a real, fully warmed
pipeline process, with GPU and model assets -- and stays the on-demand tool for that.
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

ENTRY_POINT = "vspeech.main"

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
# numbers *and* these values into the JSON's prose, so the two cannot drift apart.
#
# What pydantic_settings adds to *this* startup path, which is the signal ADR-0085 requires
# the module indicator to catch. Re-measured here rather than carried over: N=10 child
# processes each way, `vspeech.main` 716 modules and `vspeech.main` + `pydantic_settings`
# 747, zero spread on either side (2026-08-12). ADR-0066's older "about 32 modules" is the
# same measurement to within one module; 31 is what this repo reads today.
PYDANTIC_SETTINGS_MODULE_SIGNAL = 31
# Module-count slack: it has to stay **below** that signal, or the arrival of
# pydantic_settings would fit inside the budget. 16 clears that -- though only just, at
# slightly over half of 31, not under half -- and the gate fires with 15 modules to spare
# (747 against a budget of 732). It still absorbs the submodule shuffle of a routine
# dependency upgrade.
MODULE_COUNT_SLACK = 16
# Resident-memory headroom, in MiB. Sized against the measured run-to-run spread, not as a
# percentage, and deliberately not sized to absorb *growth*: any package arriving on the
# path trips the top-level check and forces a re-record anyway, so this budget only has to
# survive measurement noise. The first draft took 7.0 MiB and the calibration below caught
# numpy by 0.14 MiB -- a budget that only just works is a budget that does not work.
RSS_HEADROOM_MIB = 4.0

# Put on the startup path once per `--update`, to record what the resident-memory budget is
# actually proved to catch. numpy is the lightest heavy native dependency this project
# installs at all, so it is the hardest realistic case for the budget. It lives in the
# whisper/rvc extras; when it is absent the calibration is recorded as not measured, which
# `tests/test_runtime_footprint.py` treats as a failure of the *record* rather than of the
# environment reading it.
CALIBRATION_MODULE = "numpy"

# The child's payload. The preamble before the snapshot is `sys` and `time` only -- both
# are already in sys.modules before the interpreter hands over -- so the snapshot is the
# entry point's own footprint and nothing else. The memory read is deferred past the
# snapshot for the same reason, and reads only modules the entry point already loaded.
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
    touches after the snapshot has to be something the entry point already loaded.
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
    """One child process's startup footprint.

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

        Gating on top-level names rather than on all 700-odd entries is what keeps the
        gate from firing on a routine dependency upgrade that merely reorganises its own
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


def measure_startup(entry_points: tuple[str, ...] = (ENTRY_POINT,)) -> Measurement:
    """Import `entry_points` in a pristine child process and report what that cost.

    A child rather than an in-process check for the reason the rest of
    `tests/test_forbidden_imports.py` already spawns children: `sys.modules` inside the
    test process is contaminated by whatever ran before, so an in-process reading answers
    a different question every time.

    `-E -s` makes the reading independent of the developer's `PYTHON*` variables and user
    site-packages. `sys.path[0]` is deliberately left alone (i.e. no `-I`) so the check
    still works in a checkout where the project has not been installed into the venv.

    `entry_points` is a tuple so a test can measure a startup path with something extra on
    it, which is how the failure message is proved to name the newcomer.
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


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measure_calibration() -> Measurement | None:
    """The startup path with `CALIBRATION_MODULE` on it, or None if it is not installed."""
    try:
        return measure_startup((ENTRY_POINT, CALIBRATION_MODULE))
    except RuntimeError:
        return None


def _calibration_sentence(calibration: Measurement | None, rss_budget: float) -> str:
    """What the budget is *shown* to catch, rather than what it is hoped to catch."""
    if calibration is None:
        return (
            f"What it catches was not calibrated on this run: {CALIBRATION_MODULE} is "
            "not installed here, so re-record in an environment that has it."
        )
    over = calibration.working_set_mib - rss_budget
    verdict = f"{over:.2f} MiB over" if over > 0 else f"{-over:.2f} MiB UNDER"
    return (
        f"Calibrated the same way: putting {CALIBRATION_MODULE} on this startup path "
        f"takes the working set to {calibration.working_set_mib:.2f} MiB, {verdict} the "
        f"budget. {CALIBRATION_MODULE} is the lightest heavy native dependency this "
        "project installs at all, so it is the hardest realistic case for the budget. If "
        "that margin ever reads UNDER, the budget has stopped guarding anything and must "
        "be tightened; tests/test_runtime_footprint.py fails on a record that says so. "
        "What this budget does NOT reach: torch. Nothing on this startup path imports "
        "ctranslate2 (the transcription worker is imported lazily and defers "
        "faster_whisper into a function body), so a merely installed torch never appears "
        "in this measurement -- ADR-0084's dependency-table gate is the load-bearing "
        "guard for torch, not this one."
    )


def _calibration_record(calibration: Measurement | None, rss_budget: float) -> dict:
    if calibration is None:
        return {"module": CALIBRATION_MODULE, "measured": False}
    return {
        "module": CALIBRATION_MODULE,
        "measured": True,
        "working_set_mib": round(calibration.working_set_mib, 2),
        "over_budget_mib": round(calibration.working_set_mib - rss_budget, 2),
        "module_count": len(calibration.modules),
    }


def _build_baseline(runs: list[Measurement], calibration: Measurement | None) -> dict:
    """Turn N measurements into the recorded baseline, budgets and their basis.

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
        else ("with no run-to-run spread measured to compare it against")
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
            "モジュール集合が実行ごとに揺れているので基準データを記録できません。"
            f"全 {n} 回に現れなかった名前: {', '.join(unstable)}\n"
            "このまま記録するとゲートが暴れます。揺れの原因 (子プロセスへ漏れている "
            "環境変数、条件付き import など) を先に潰してください。"
        )

    count_spread = counts[-1] - counts[0]
    return {
        "entry_point": ENTRY_POINT,
        "what_this_is": (
            "The outcome gate of ADR-0085: what importing the runtime's entry point in a "
            "pristine child process loads, and what it costs in resident memory. Read by "
            "tests/test_runtime_footprint.py. Regenerate, do not hand-edit."
        ),
        "regenerate_with": REGENERATE_COMMAND,
        "module_count": {
            "observed_max": counts[-1],
            "observed_min": counts[0],
            "runs": n,
            "budget": count_budget,
            "basis": (
                f"len(sys.modules) after importing {ENTRY_POINT}. N={n} consecutive runs: "
                f"min {counts[0]}, max {counts[-1]} (spread {count_spread}). "
                f"Budget = max + {MODULE_COUNT_SLACK}. ADR-0085 requires this indicator "
                "to catch pydantic_settings, measured at "
                f"+{PYDANTIC_SETTINGS_MODULE_SIGNAL} modules on this path (N=10), so the "
                "slack is held below that signal -- slightly over half of it, not under -- "
                "while still absorbing the submodule shuffle of a routine dependency "
                "upgrade."
            ),
        },
        "resident_memory_mib": {
            "observed_max": round(working[-1], 2),
            "observed_min": round(working[0], 2),
            "observed_median": round(median(working), 2),
            "runs": n,
            "budget": rss_budget,
            "basis": (
                f"Working set (GetProcessMemoryInfo) of the child itself after importing "
                f"{ENTRY_POINT}. N={n} consecutive runs: min {working[0]:.2f}, median "
                f"{median(working):.2f}, max {working[-1]:.2f} MiB -- spread "
                f"{spread:.2f} MiB ({spread_pct:.1f}% of the median). Budget = max + "
                f"{RSS_HEADROOM_MIB:.1f} MiB headroom rounded up, i.e. {headroom:.2f} MiB "
                f"of headroom, {versus_spread}, so it does not flap. "
                + _calibration_sentence(calibration, rss_budget)
                + " What it deliberately does not catch is pydantic_settings' ~1.5 MiB of "
                "cost unique to this path (N=10; ADR-0085) -- the module indicators cover "
                "that, which is why there are two."
            ),
            "calibration": _calibration_record(calibration, rss_budget),
        },
        "top_level_modules": sorted(top_level),
    }


def _write_baseline(baseline: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a factory so a test can check `REGENERATE_COMMAND` really parses."""
    parser = argparse.ArgumentParser(
        description="ランタイムの起動が持ち込むモジュール集合と常駐メモリを実測する"
        "（ADR-0085 の成果ゲートの基準データ）"
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

    runs: list[Measurement] = []
    for i in range(args.runs):
        measurement = measure_startup()
        runs.append(measurement)
        print(
            f"run {i + 1}: modules={len(measurement.modules)} "
            f"top_level={len(measurement.top_level)} "
            f"ws={measurement.working_set_mib:.2f}MiB "
            f"private={measurement.private_mib:.2f}MiB "
            f"import={measurement.seconds:.2f}s (not gated)"
        )

    module_sets = [frozenset(m.modules) for m in runs]
    unstable = set.union(*(set(s) for s in module_sets)) - set.intersection(
        *(set(s) for s in module_sets)
    )
    working = sorted(m.working_set_mib for m in runs)
    print(
        f"\nN={len(runs)}: modules {len(module_sets[0])} "
        f"(unstable across runs: {len(unstable)}) "
        f"ws min/median/max {working[0]:.2f}/{median(working):.2f}/{working[-1]:.2f}MiB"
    )
    if unstable:
        print("unstable module names: " + ", ".join(sorted(unstable)))

    baseline = _build_baseline(runs, measure_calibration())
    memory = baseline["resident_memory_mib"]
    print(
        f"module budget {baseline['module_count']['budget']}, "
        f"rss budget {memory['budget']}MiB"
    )
    print("calibration: " + json.dumps(memory["calibration"]))
    if args.update:
        _write_baseline(baseline)
        print(f"wrote {BASELINE_PATH.relative_to(REPO_ROOT)}")
    else:
        print("(--update to record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
