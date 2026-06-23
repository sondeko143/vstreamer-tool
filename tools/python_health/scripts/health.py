from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field


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
                ["uv", "export", "--no-hashes", "--no-emit-project", "--frozen", "-o", req]
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
                gate.name, "skipped", "prepare step unavailable", perr.strip(), gate.advisory
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
