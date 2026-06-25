"""Capture a py-spy startup profile of vspeech.

Runs ``py-spy record`` over ``python -m vspeech --config <cfg>`` (default: the
bundled minimal config) for a bounded window and writes a speedscope JSON for
``startup_analyze.py`` (or speedscope.app).

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
is needed here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


def venv_python(root: Path, platform: str) -> Path:
    if platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def default_config() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "minimal_startup.toml"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="startup-profile")
    parser.add_argument(
        "--config", default=None, help="config to profile (default: bundled minimal)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="sampling window seconds"
    )
    parser.add_argument("--rate", type=int, default=100, help="py-spy sample Hz")
    parser.add_argument("--out-dir", default=".startup-profiles")
    parser.add_argument(
        "--native", action="store_true", help="also unwind native frames"
    )
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    python = venv_python(root, sys.platform)
    if not python.exists():
        python = Path(sys.executable)  # fall back to the launching interpreter
    config = Path(args.config) if args.config else default_config()
    if not config.exists():
        print(f"[error] config not found: {config}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    profile_path = out_dir / f"startup-{stamp}.speedscope.json"
    log_path = out_dir / f"startup-{stamp}.log"

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
        try:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
        except FileNotFoundError:
            print(
                "[error] uvx/py-spy not found — install uv, then `uvx py-spy`",
                file=sys.stderr,
            )
            return 3
    elapsed = perf_counter() - t0

    print(f"\ncaptured in {elapsed:.1f}s; vspeech log tail:")
    _print_log_tail(log_path)
    if profile_path.exists():
        print("\nnext: analyze it with")
        print(f"  uv run poe startup-analyze --input {profile_path}")
        print(f"  (or open {profile_path} at https://speedscope.app)")
        return 0
    print("[error] no profile written — see log above", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
