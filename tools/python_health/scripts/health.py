from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class Targets:
    packages: list[str]
    project_name: str


@dataclass
class Gate:
    name: str
    phase: str
    check: list[str]
    kind: str
    fix: list[str] | None = None
    prepare: list[list[str]] = field(default_factory=list)
    advisory: bool = False


@dataclass
class GateResult:
    name: str
    status: str
    summary: str
    detail: str = ""
    advisory: bool = False


def derive_targets(pyproject: dict) -> Targets:
    project = pyproject.get("project", {})
    name = str(project.get("name", ""))
    tool = pyproject.get("tool", {})
    module_name = tool.get("uv", {}).get("build-backend", {}).get("module-name")

    packages: list[str] = []
    if isinstance(module_name, list):
        packages = [str(m) for m in module_name]
    elif isinstance(module_name, str):
        packages = [module_name]

    if not packages:
        for entry in project.get("scripts", {}).values():
            top = str(entry).split(":", 1)[0].split(".", 1)[0]
            if top and top not in packages:
                packages.append(top)

    if not packages and name:
        packages = [name.replace("-", "_")]

    return Targets(packages=packages, project_name=name)


def classify(returncode: int, stdout: str, stderr: str) -> str:
    if returncode == 0:
        return "pass"
    if returncode == 127 or "command not found" in (stderr or "").lower():
        return "skipped"
    return "fail"


def build_gates(targets: Targets) -> list[Gate]:
    pkgs = targets.packages or ["."]
    cov_args: list[str] = [f"--cov={p}" for p in pkgs]
    req = os.path.join(tempfile.gettempdir(), "python-health-requirements.txt")
    return [
        Gate(
            "ruff-format",
            "static",
            ["uv", "run", "ruff", "format", "--check", "."],
            "fixable",
            fix=["uv", "run", "ruff", "format", "."],
        ),
        Gate(
            "ruff-lint",
            "static",
            ["uv", "run", "ruff", "check", "."],
            "fixable",
            fix=["uv", "run", "ruff", "check", "--fix", "."],
        ),
        Gate("ty", "static", ["uv", "run", "ty", "check"], "report"),
        Gate(
            "pytest-cov",
            "tests",
            ["uv", "run", "pytest", *cov_args, "--cov-report=term-missing"],
            "report",
        ),
        Gate("uv-lock-check", "deps", ["uv", "lock", "--check"], "report"),
        Gate(
            "pip-audit",
            "deps",
            ["uvx", "pip-audit", "-r", req],
            "report",
            prepare=[
                [
                    "uv",
                    "export",
                    "--no-hashes",
                    "--no-emit-project",
                    "--frozen",
                    "-o",
                    req,
                ]
            ],
        ),
        Gate(
            "outdated",
            "deps",
            ["uv", "pip", "list", "--outdated"],
            "report",
            advisory=True,
        ),
        Gate("bandit", "extra", ["uvx", "bandit", "-q", "-r", *pkgs], "report"),
        Gate(
            "vulture",
            "extra",
            ["uvx", "vulture", *pkgs, "--min-confidence", "80"],
            "report",
            advisory=True,
        ),
    ]


CommandRunner = Callable[[list[str]], tuple[int, str, str]]


def run_gate(gate: Gate, run: CommandRunner) -> GateResult:
    for prep in gate.prepare:
        prc, _pout, perr = run(prep)
        if classify(prc, _pout, perr) == "skipped":
            return GateResult(
                gate.name,
                "skipped",
                "prepare step unavailable",
                perr.strip(),
                gate.advisory,
            )

    rc, out, err = run(gate.check)
    status = classify(rc, out, err)

    if status == "pass":
        return GateResult(gate.name, "pass", "ok", advisory=gate.advisory)
    if status == "skipped":
        return GateResult(
            gate.name, "skipped", "tool unavailable", err.strip(), gate.advisory
        )

    if gate.kind == "fixable" and gate.fix is not None:
        run(gate.fix)
        rc2, out2, err2 = run(gate.check)
        if classify(rc2, out2, err2) == "pass":
            return GateResult(gate.name, "fixed", "auto-fixed", advisory=gate.advisory)
        return GateResult(
            gate.name, "fail", "fix incomplete", (err2 or out2).strip(), gate.advisory
        )

    return GateResult(
        gate.name, "fail", "needs attention", (err or out).strip(), gate.advisory
    )


_STATUS_TAG = {
    "pass": "PASS",
    "fail": "FAIL",
    "fixed": "FIXED",
    "skipped": "SKIP",
    "error": "ERR",
}


def overall_exit(results: list[GateResult]) -> int:
    for r in results:
        if r.status in ("fail", "error") and not r.advisory:
            return 1
    return 0


def render_summary(results: list[GateResult]) -> str:
    lines = ["", "=== python-health summary ==="]
    for r in results:
        tag = _STATUS_TAG.get(r.status, r.status.upper())
        adv = " (advisory)" if r.advisory else ""
        lines.append(f"[{tag}] {r.name}{adv}: {r.summary}")
        if r.status in ("fail", "error") and r.detail:
            lines.append(f"       {r.detail.splitlines()[-1][:200]}")
    return "\n".join(lines)


def results_to_json(results: list[GateResult]) -> str:
    return json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2)


def parse_coverage(stdout: str) -> float | None:
    match = re.search(r"^TOTAL\s+.*?(\d+(?:\.\d+)?)%\s*$", stdout, re.MULTILINE)
    return float(match.group(1)) if match else None


def subprocess_runner(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    return proc.returncode, proc.stdout, proc.stderr


def load_pyproject(root: Path) -> dict:
    import tomllib

    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def apply_no_fix(gates: list[Gate]) -> list[Gate]:
    return [
        Gate(
            g.name,
            g.phase,
            g.check,
            "report" if g.kind == "fixable" else g.kind,
            None if g.kind == "fixable" else g.fix,
            g.prepare,
            g.advisory,
        )
        for g in gates
    ]


def run_all(
    gates: list[Gate], run: CommandRunner, fail_fast: bool = False
) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in gates:
        try:
            result = run_gate(gate, run)
        except (
            Exception
        ) as exc:  # gate isolation (spec §7): a crash must not abort the rest
            result = GateResult(
                gate.name, "error", "gate raised", str(exc), gate.advisory
            )
        results.append(result)
        if fail_fast and result.status in ("fail", "error") and not result.advisory:
            break
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python-health")
    parser.add_argument("--root", default=".")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fix", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    targets = derive_targets(load_pyproject(root))
    gates = build_gates(targets)
    if args.no_fix:
        gates = apply_no_fix(gates)

    results = run_all(gates, subprocess_runner, fail_fast=args.fail_fast)

    if args.json:
        print(results_to_json(results))
    else:
        print(render_summary(results))
    return overall_exit(results)


if __name__ == "__main__":
    raise SystemExit(main())
