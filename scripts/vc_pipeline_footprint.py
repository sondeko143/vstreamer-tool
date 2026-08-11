"""Measure a real vspeech pipeline process's startup time and resident memory.

ADR-0080 removes torch from the runtime so that the VC host stops paying its resident
cost. That claim is only worth what the measurement behind it is worth, so this is the
procedure -- it is meant to be re-run verbatim on the torch-free venv and the two numbers
subtracted.

    uv run poe vc-footprint --config <a config with [vc] or [stream_vc] enabled>

Two Windows traps it exists to avoid:

- **The PID you spawn is not the PID that holds the memory.** A uv-created `.venv`
  ships a *trampoline* `python.exe` which launches the real base interpreter as a child
  (measured: trampoline about 5MB, child hundreds of MB). Reading a process-tree total
  -- what Task Manager's process row shows -- would fold the launcher in. So the child
  reports `os.getpid()` about itself before handing control to the module, and every
  sample below is taken against that exact PID via `GetProcessMemoryInfo`.
- **Startup is not "until the process exists".** The models load and warm up after the
  interpreter is up, so the process is only meaningfully resident once the worker says
  so. Time runs from `Popen` to the readiness marker on stdout (`--ready-marker`,
  default the vc worker's), and memory is sampled only after it.

`--runs` repeats the whole launch, because one run is not a measurement; the medians are
what to quote. The pipeline is killed with `taskkill /T` so the trampoline cannot leave
an orphaned interpreter behind.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess  # nosec B404 - launching the pipeline under test is the whole point
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from time import sleep

REPO_ROOT = Path(__file__).resolve().parents[1]

# Printed by the launched process before it runs the module, so we learn the real
# interpreter's PID from the process itself rather than guessing at the process tree.
PID_MARKER = "__VSPEECH_FOOTPRINT_PID__"

# `-c` payload: announce our own PID, then run `vspeech` exactly as `-m` would. click
# reads sys.argv[1:], which `-c` fills with everything after the payload, so
# `--config <path>` arrives unchanged.
BOOTSTRAP = (
    "import os,sys,runpy;"
    f"print('{PID_MARKER}',os.getpid(),flush=True);"
    "runpy.run_module('vspeech',run_name='__main__')"
)

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010


class _MemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX (psapi.h)."""

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


@dataclass
class Sample:
    working_set_mb: float
    peak_working_set_mb: float
    private_mb: float


def read_memory(pid: int) -> Sample | None:
    """Working set / private bytes of exactly `pid`, or None if it is gone."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid
    )
    if not handle:
        return None
    try:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = kernel32.K32GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        mb = 1024.0 * 1024.0
        return Sample(
            working_set_mb=counters.WorkingSetSize / mb,
            peak_working_set_mb=counters.PeakWorkingSetSize / mb,
            private_mb=counters.PrivateUsage / mb,
        )
    finally:
        kernel32.CloseHandle(handle)


def venv_python(root: Path = REPO_ROOT) -> Path:
    return root / ".venv" / "Scripts" / "python.exe"


@dataclass
class RunResult:
    startup_s: float
    child_pid: int
    steady: Sample
    launcher_working_set_mb: float


def kill_tree(pid: int) -> None:
    subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        check=False,
    )


def measure_once(
    config: Path, ready_marker: str, settle_s: float, timeout_s: float
) -> RunResult:
    """One launch: time to the readiness marker, then the steady-state footprint."""
    env = dict(os.environ)
    # Unbuffered so the readiness line arrives when it is logged, not when the pipe
    # buffer fills; UTF-8 because vspeech's log lines are Japanese.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    argv = [
        str(venv_python()),
        "-c",
        BOOTSTRAP,
        "--config",
        str(config.expanduser()),
    ]
    t0 = perf_counter()
    proc = subprocess.Popen(  # nosec B603 - fixed argv built here, no shell
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.stdout is not None  # nosec B101 - stdout=PIPE guarantees it
    child_pid = 0
    startup_s = 0.0
    tail: list[str] = []
    try:
        for line in proc.stdout:
            tail.append(line.rstrip())
            del tail[:-40]
            if line.startswith(PID_MARKER):
                child_pid = int(line.split()[1])
            elif ready_marker in line:
                startup_s = perf_counter() - t0
                break
            if perf_counter() - t0 > timeout_s:
                raise SystemExit(
                    f"{timeout_s}s 以内に readiness marker "
                    f"{ready_marker!r} が現れませんでした:\n" + "\n".join(tail)
                )
        if not startup_s:
            raise SystemExit(
                "プロセスが readiness marker の前に終了しました:\n" + "\n".join(tail)
            )
        if not child_pid:
            raise SystemExit("子プロセスの PID を取得できませんでした")
        launcher = read_memory(proc.pid)
        # Sample after the marker and let it settle: the worker has warmed up by then,
        # so what is left is arena growth, not model loading.
        sleep(settle_s)
        steady = read_memory(child_pid)
        if steady is None:
            raise SystemExit(f"PID {child_pid} が測定前に消えました")
        return RunResult(
            startup_s=startup_s,
            child_pid=child_pid,
            steady=steady,
            launcher_working_set_mb=launcher.working_set_mb if launcher else 0.0,
        )
    finally:
        kill_tree(proc.pid)
        proc.stdout.close()
        proc.wait(timeout=30)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    parser = argparse.ArgumentParser(
        description="vspeech パイプラインの起動時間と常駐メモリを実測する"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ready-marker", default="vc worker started")
    parser.add_argument("--settle", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--gap",
        type=float,
        default=8.0,
        help="次の run を始めるまでの待ち。killed 直後は gRPC の listen ポートが "
        "まだ解放されておらず、次の run が bind に失敗する",
    )
    parser.add_argument(
        "--label", default="", help="出力に付ける識別子 (torch あり/なしの区別用)"
    )
    args = parser.parse_args()

    print(f"config: {args.config}")
    print(f"ready marker: {args.ready_marker!r}  runs: {args.runs}")
    results: list[RunResult] = []
    for i in range(args.runs):
        if i:
            sleep(args.gap)
        result = measure_once(args.config, args.ready_marker, args.settle, args.timeout)
        results.append(result)
        print(
            f"run {i + 1}: startup={result.startup_s:.2f}s "
            f"pid={result.child_pid} "
            f"ws={result.steady.working_set_mb:.1f}MB "
            f"peak_ws={result.steady.peak_working_set_mb:.1f}MB "
            f"private={result.steady.private_mb:.1f}MB "
            f"(launcher ws={result.launcher_working_set_mb:.1f}MB)"
        )

    label = f" [{args.label}]" if args.label else ""
    print(
        f"\nmedian over N={len(results)}{label}: "
        f"startup={median(r.startup_s for r in results):.2f}s "
        f"ws={median(r.steady.working_set_mb for r in results):.1f}MB "
        f"peak_ws={median(r.steady.peak_working_set_mb for r in results):.1f}MB "
        f"private={median(r.steady.private_mb for r in results):.1f}MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
