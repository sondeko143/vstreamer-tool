"""Capture a py-spy startup profile of vspeech and rank where the time goes.

Runs ``py-spy record`` over ``python -m vspeech --config <cfg>`` for a bounded
window, writes a speedscope JSON, and analyzes it inline (via ``startup_analyze``)
so one command both captures and reports.

Two modes:

- **default / single** — profile one config. With no ``--config`` this profiles
  the bundled BASELINE (every optional worker disabled; only the always-on gRPC
  ``sender``/``receiver`` run), i.e. the pure import/infra floor every config
  pays. Pass ``--config <path>`` for a real config.
- **sweep** (``--sweep``) — profile the baseline plus one config per optional
  worker (recording, transcription, subtitle, translation, tts, playback; ``vc``
  is excluded — its torch + CUDA model load dwarfs the window and needs real
  model assets) and print a comparison table of active time by config.

Two Windows specifics drive the py-spy flags:

- ``--subprocesses``: a uv-created ``.venv`` ships a *trampoline* ``python.exe``
  (not a CPython copy); py-spy can't identify it as Python and would fail with
  "Failed to find python version from target process". The trampoline launches
  the real base interpreter as a child, so ``--subprocesses`` lets py-spy follow
  into it. It also captures credential helpers vspeech shells out to (e.g.
  ``gcloud``).
- ``--idle``: Python releases the GIL during blocking I/O (DNS/socket, waiting
  on a subprocess), so the default on-CPU sampling would miss exactly the stalls
  this skill hunts for.

py-spy spawns and reaps the child tree itself, so no separate process management
is needed here. Advisory only — never edits source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Active (non-idle) buckets, ordered for the comparison table.
_ACTIVE_BUCKETS = ("blocking-io", "import", "compute", "other")


def venv_python(root: Path, platform: str) -> Path:
    if platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def default_config() -> Path:
    """The config profiled when neither --config nor --sweep is given."""
    return fixtures_dir() / "baseline_startup.toml"


def sweep_fixtures() -> list[tuple[str, Path]]:
    """Ordered (label, path) configs for ``--sweep``: the baseline floor plus one
    single-worker config per optional worker. ``vc`` is intentionally excluded —
    its torch/onnxruntime CUDA load exceeds a normal window and needs real RVC
    model assets to run at all."""
    base = fixtures_dir()
    sweep = base / "sweep"
    return [
        ("baseline", base / "baseline_startup.toml"),
        ("recording", sweep / "recording.toml"),
        ("transcription", sweep / "transcription.toml"),
        ("subtitle", sweep / "subtitle.toml"),
        ("translation", sweep / "translation.toml"),
        ("tts", sweep / "tts.toml"),
        ("playback", sweep / "playback.toml"),
    ]


def build_vspeech_cmd(python: str, config: str) -> list[str]:
    return [python, "-m", "vspeech", "--config", config]


def build_pyspy_cmd(
    out: str,
    python: str,
    config: str,
    duration: float,
    rate: int,
    native: bool = False,
) -> list[str]:
    cmd = [
        "uvx",
        "py-spy",
        "record",
        "-f",
        "speedscope",
        "-o",
        out,
        "--rate",
        str(rate),
        "--duration",
        str(int(duration)),
        "--subprocesses",
        "--idle",
    ]
    if native:
        cmd.append("--native")
    cmd.append("--")
    cmd.extend(build_vspeech_cmd(python, config))
    return cmd


def _print_log_tail(log_path: Path, n: int = 12) -> None:
    try:
        lines = log_path.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines[-n:]:
        print("  " + line)


def _load_analyzer():
    """Import the sibling analyzer whether we're run as ``scripts/startup_capture.py``
    (scripts/ on sys.path) or imported as ``scripts.startup_capture`` (tests)."""
    try:
        import startup_analyze as sa  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        from scripts import startup_analyze as sa
    return sa


def analyze_profile(profile_path: Path, top: int) -> tuple[str, dict]:
    """Render the bucket summary for a speedscope profile and return it alongside
    a row of per-bucket self-time totals for the comparison table."""
    sa = _load_analyzer()
    doc = json.loads(profile_path.read_text(encoding="utf-8"))
    stats, total, unit = sa.compute_frame_stats(doc)
    summary = sa.render_summary(stats, total, unit, top)
    totals = sa.bucket_totals(stats)
    row = {
        "active": total - totals["idle"],
        "total": total,
        "unit": unit,
        **{b: totals[b] for b in _ACTIVE_BUCKETS},
    }
    return summary, row


def _dominant_bucket(row: dict) -> str:
    active = {b: row[b] for b in _ACTIVE_BUCKETS}
    if row["active"] <= 0 or not any(active.values()):
        return "-"
    return max(active, key=lambda b: active[b])


def render_comparison(rows: list[dict]) -> str:
    """A table of active time by config, sorted heaviest first. Idle excluded."""
    if not rows:
        return "\n=== startup sweep: no profiles captured ==="
    unit = rows[0].get("unit", "s")
    ordered = sorted(rows, key=lambda r: r["active"], reverse=True)
    head = (
        f"{'config':<16}{'active':>8}{'block':>8}{'import':>8}"
        f"{'compute':>8}{'other':>8}  dominant"
    )
    lines = [
        "",
        f"=== startup sweep: active {unit} by config (idle excluded) ===",
        head,
    ]
    for r in ordered:
        lines.append(
            f"{r['label']:<16}{r['active']:>8.2f}{r['blocking-io']:>8.2f}"
            f"{r['import']:>8.2f}{r['compute']:>8.2f}{r['other']:>8.2f}"
            f"  {_dominant_bucket(r)}"
        )
    lines.append("")
    lines.append(
        "active = total - idle. Subtract the baseline row to isolate each "
        "worker's own startup cost."
    )
    return "\n".join(lines)


def run_config(
    config: Path, label: str, python: Path, args: argparse.Namespace
) -> dict | None:
    """Capture one config, print its capture log tail + bucket analysis, and
    return a comparison row (or None if no profile was written)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    profile_path = out_dir / f"startup-{stamp}-{label}.speedscope.json"
    log_path = out_dir / f"startup-{stamp}-{label}.log"

    cmd = build_pyspy_cmd(
        str(profile_path),
        str(python),
        str(config),
        args.duration,
        args.rate,
        args.native,
    )
    print(f"config:   {config}")
    print(f"python:   {python}")
    print(f"profile:  {profile_path}")
    print(f"log:      {log_path}")
    print(
        f"sampling: {args.duration:.0f}s @ {args.rate}Hz (py-spy --subprocesses --idle)"
    )

    t0 = perf_counter()
    with log_path.open("wb") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    elapsed = perf_counter() - t0

    print(f"\ncaptured in {elapsed:.1f}s; vspeech log tail:")
    _print_log_tail(log_path)
    if not profile_path.exists():
        print("[error] no profile written — see log above", file=sys.stderr)
        return None

    summary, row = analyze_profile(profile_path, args.top)
    print(summary)
    print(f"\n(open {profile_path} at https://speedscope.app for the flamegraph)")
    row["label"] = label
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="startup-profile")
    parser.add_argument(
        "--config",
        default=None,
        help="config to profile (default: bundled baseline, all workers disabled)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="profile the baseline + one config per worker and print a comparison",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="sampling window seconds (per config)",
    )
    parser.add_argument("--rate", type=int, default=100, help="py-spy sample Hz")
    parser.add_argument("--out-dir", default=".startup-profiles")
    parser.add_argument(
        "--native", action="store_true", help="also unwind native frames"
    )
    parser.add_argument(
        "--top", type=int, default=12, help="frames to list per config analysis"
    )
    args = parser.parse_args(argv)

    if shutil.which("uvx") is None:
        print(
            "[skip] uvx/py-spy not found — install uv, then `uvx py-spy` is used "
            "to record. This skill is advisory and does not gate anything.",
            file=sys.stderr,
        )
        return 3

    root = Path.cwd().resolve()
    python = venv_python(root, sys.platform)
    if not python.exists():
        python = Path(sys.executable)  # fall back to the launching interpreter

    if args.sweep:
        if args.config:
            print(
                "[error] --sweep profiles the bundled fixtures; drop --config",
                file=sys.stderr,
            )
            return 2
        rows: list[dict] = []
        fixtures = sweep_fixtures()
        for i, (label, cfg) in enumerate(fixtures, 1):
            print("\n" + "#" * 56)
            print(f"#### [{i}/{len(fixtures)}] {label}  ({cfg.name})")
            print("#" * 56)
            if not cfg.exists():
                print(f"[warn] fixture not found, skipping: {cfg}", file=sys.stderr)
                continue
            row = run_config(cfg, label, python, args)
            if row:
                rows.append(row)
        print(render_comparison(rows))
        return 0

    config = Path(args.config) if args.config else default_config()
    if not config.exists():
        print(f"[error] config not found: {config}", file=sys.stderr)
        return 2
    label = config.stem
    row = run_config(config, label, python, args)
    return 0 if row else 1


if __name__ == "__main__":
    raise SystemExit(main())
