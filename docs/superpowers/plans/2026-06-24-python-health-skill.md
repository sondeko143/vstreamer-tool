# python-health skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `python-health`, a user-level Claude Code skill that runs Python project health gates (ruff, ty, pytest+coverage, uv lock, pip-audit, bandit, vulture) on-demand in this dev environment, auto-fixing only mechanical issues and triaging the rest.

**Architecture:** A stdlib-only Python orchestrator (`health.py`) defines gates as data, runs each via an injectable command runner, classifies results (pass/fail/fixed/skipped/error), auto-applies safe fixes (`ruff format`, `ruff check --fix`) and reports everything else for human triage. A `SKILL.md` tells Claude how to invoke it and how to present/triage results. Source is developed and TDD'd in this repo under `tools/python_health/`, then installed to `~/.claude/skills/python-health/`.

**Tech Stack:** Python 3.11 (stdlib `tomllib`, `subprocess`, `argparse`, `dataclasses`), `uv` as the command runner, `uvx` for ephemeral tools (pip-audit, bandit, vulture), `pytest` for the orchestrator's own unit tests.

## Global Constraints

- **Python 3.11 only** (`requires-python = ">=3.11,<3.12"`). Use stdlib `tomllib` (3.11+) for TOML parsing.
- **`health.py` is stdlib-only on Python ≥3.11** — gate work is shelled out, never imported. The ONE allowed exception is a guarded `tomli` fallback for TOML parsing on Python <3.11 (`try: import tomllib / except ModuleNotFoundError: import tomli`), since stdlib `tomllib` is 3.11+. The skill is launched with `uv run --with tomli` so the fallback is available on older projects. (Validation discovered the orchestrator otherwise crashes on a 3.10 project.)
- **The orchestrator dogfoods the repo's own gates:** `health.py` and its tests live under `tools/python_health/` and ARE scanned by `ruff check .` and `ty check`. They MUST pass: `ruff format`, ruff `I` + `UP`, **one import per line** (`force-single-line = true`), and `ty`.
- **Dev tests are not in the default suite.** Project `pyproject.toml` has `testpaths = ["tests"]`, so the orchestrator's tests under `tools/python_health/scripts/tests/` are NOT collected by a bare `pytest`. Run them explicitly: `uv run pytest tools/python_health/scripts/tests/ -v`.
- **Honor the target project's pytest config** — never override `addopts` (which excludes `voicevox_e2e`). The pytest gate appends only `--cov=...` / `--cov-report`.
- **Auto-fix is restricted** to `ruff format` and `ruff check --fix` (safe fixes only; never `--unsafe-fixes`). Everything else is report-only.
- **Install target:** `~/.claude/skills/python-health/` containing `SKILL.md`, `references/`, `scripts/health.py` (the in-repo `scripts/tests/` are NOT installed).
- **No package-relative imports in `health.py`** — it is copied and run as a standalone file path.

## File Structure

In-repo development home (mirrors the installed layout so SKILL.md's relative paths are correct):

```
tools/__init__.py                       # empty — makes the chain a regular package for ty
tools/python_health/__init__.py         # empty
tools/python_health/scripts/__init__.py # empty
tools/python_health/
  SKILL.md                       # skill manifest (frontmatter + procedure) — installed
  references/gate-catalog.md     # gate definitions, fix policy, FP tuning — installed
  scripts/health.py              # the orchestrator (stdlib-only) — installed
  scripts/tests/__init__.py      # empty
  scripts/tests/test_health.py   # unit tests — NOT installed
```

Tests import the orchestrator as `from tools.python_health.scripts import health`. `pytest` resolves this via `pythonpath = "."`. **`ty` does NOT resolve implicit namespace packages here**, so the `__init__.py` files above (regular package chain) are required for the dogfood `ty check` to resolve the test import. The empty `__init__.py` files are not installed (only `SKILL.md`, `references/`, `scripts/health.py` ship).

Installed layout (after Task 10):

```
~/.claude/skills/python-health/
  SKILL.md
  references/gate-catalog.md
  scripts/health.py
```

---

### Task 1: Scaffold + project target detection

**Files:**
- Create: `tools/python_health/scripts/health.py`
- Create: `tools/python_health/scripts/tests/test_health.py`

**Interfaces:**
- Produces: `Targets` dataclass with fields `packages: list[str]`, `project_name: str`; and `derive_targets(pyproject: dict) -> Targets`.

- [ ] **Step 1: Write the failing test**

```python
# tools/python_health/scripts/tests/test_health.py
from tools.python_health.scripts import health


def test_derive_targets_from_module_name():
    pyproject = {
        "project": {"name": "voicerecog"},
        "tool": {"uv": {"build-backend": {"module-name": ["vspeech"]}}},
    }
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["vspeech"]
    assert targets.project_name == "voicerecog"


def test_derive_targets_falls_back_to_script_entrypoint():
    pyproject = {"project": {"name": "foo", "scripts": {"foo": "foo_pkg.main:cmd"}}}
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["foo_pkg"]


def test_derive_targets_falls_back_to_normalized_project_name():
    pyproject = {"project": {"name": "my-tool"}}
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["my_tool"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'health' has no attribute 'derive_targets'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/python_health/scripts/health.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Targets:
    packages: list[str]
    project_name: str


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/python_health/scripts/health.py tools/python_health/scripts/tests/test_health.py
git commit -m "feat(python-health): project target detection"
```

---

### Task 2: Gate model + result classification + gate catalog

**Files:**
- Modify: `tools/python_health/scripts/health.py`
- Modify: `tools/python_health/scripts/tests/test_health.py`

**Interfaces:**
- Consumes: `Targets` (Task 1).
- Produces:
  - `Gate` dataclass: `name: str`, `phase: str`, `check: list[str]`, `kind: str` (`"fixable"` | `"report"`), `fix: list[str] | None = None`, `prepare: list[list[str]] = []`, `advisory: bool = False`.
  - `GateResult` dataclass: `name: str`, `status: str`, `summary: str`, `detail: str = ""`, `advisory: bool = False`.
  - `classify(returncode: int, stdout: str, stderr: str) -> str` → one of `"pass"`, `"skipped"`, `"fail"`.
  - `build_gates(targets: Targets) -> list[Gate]`.

- [ ] **Step 1: Write the failing test**

```python
def test_classify_pass_fail_skip():
    assert health.classify(0, "", "") == "pass"
    assert health.classify(1, "errors", "") == "fail"
    assert health.classify(127, "", "command not found: ruff") == "skipped"


def test_build_gates_uses_targets_for_cov_and_security():
    targets = health.Targets(packages=["vspeech"], project_name="voicerecog")
    gates = health.build_gates(targets)
    names = [g.name for g in gates]
    assert names == [
        "ruff-format",
        "ruff-lint",
        "ty",
        "pytest-cov",
        "uv-lock-check",
        "pip-audit",
        "outdated",
        "bandit",
        "vulture",
    ]
    pytest_gate = next(g for g in gates if g.name == "pytest-cov")
    assert "--cov=vspeech" in pytest_gate.check
    fmt = next(g for g in gates if g.name == "ruff-format")
    assert fmt.kind == "fixable" and fmt.fix is not None
    outdated = next(g for g in gates if g.name == "outdated")
    assert outdated.advisory is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: FAIL — `AttributeError` for `Gate` / `classify` / `build_gates`.

- [ ] **Step 3: Write minimal implementation**

Add to `health.py` (imports at top, one per line):

```python
import os
import tempfile
from dataclasses import field
```

```python
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
        # bandit/vulture run under the PROJECT interpreter (`uv run --with`),
        # not `uvx` — uvx may pick a Python that can't parse project syntax
        # (e.g. 3.11 `except*`), causing false syntax errors / silent skips.
        Gate(
            "bandit",
            "extra",
            ["uv", "run", "--with", "bandit", "bandit", "-q", "-r", *pkgs],
            "report",
        ),
        Gate(
            "vulture",
            "extra",
            ["uv", "run", "--with", "vulture", "vulture", *pkgs, "--min-confidence", "80"],
            "report",
            advisory=True,
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/python_health/scripts/health.py tools/python_health/scripts/tests/test_health.py
git commit -m "feat(python-health): gate model, classification, catalog"
```

---

### Task 3: Gate runner with injectable executor + auto-fix flow

**Files:**
- Modify: `tools/python_health/scripts/health.py`
- Modify: `tools/python_health/scripts/tests/test_health.py`

**Interfaces:**
- Consumes: `Gate`, `GateResult`, `classify` (Task 2).
- Produces:
  - `CommandRunner` type alias = `Callable[[list[str]], tuple[int, str, str]]` (returns `(returncode, stdout, stderr)`).
  - `run_gate(gate: Gate, run: CommandRunner) -> GateResult`.
  - Behavior: run `prepare` commands first (a prepare returning 127 ⇒ result `skipped`); run `check`; `pass` ⇒ `GateResult(status="pass")`; `skipped` ⇒ `status="skipped"`; `fail` + `kind=="fixable"` ⇒ run `fix`, re-run `check`, `status="fixed"` if now clean else `status="fail"` ("fix incomplete"); `fail` + `kind=="report"` ⇒ `status="fail"`.

- [ ] **Step 1: Write the failing test**

```python
class _ScriptedRunner:
    # callable class (not a function with an attached attribute) so `ty`
    # resolves `.calls` — a function-attribute assignment fails `ty check`.
    def __init__(self, script):
        self._script = list(script)
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self._script.pop(0)


def _scripted_runner(script):
    return _ScriptedRunner(script)


def test_run_gate_pass():
    gate = health.Gate("ty", "static", ["ty", "check"], "report")
    run = _scripted_runner([(0, "ok", "")])
    result = health.run_gate(gate, run)
    assert result.status == "pass"


def test_run_gate_report_fail():
    gate = health.Gate("ty", "static", ["ty", "check"], "report")
    run = _scripted_runner([(1, "type error", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fail"
    assert "type error" in result.detail


def test_run_gate_fixable_autofixes():
    gate = health.Gate(
        "ruff-format", "static", ["fmt", "--check"], "fixable", fix=["fmt"]
    )
    # check fails, then fix runs, then re-check passes
    run = _scripted_runner([(1, "would reformat", ""), (0, "", ""), (0, "", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fixed"
    assert run.calls == [["fmt", "--check"], ["fmt"], ["fmt", "--check"]]


def test_run_gate_fixable_incomplete_stays_fail():
    gate = health.Gate(
        "ruff-lint", "static", ["lint"], "fixable", fix=["lint", "--fix"]
    )
    run = _scripted_runner([(1, "E501", ""), (0, "", ""), (1, "E501 remains", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fail"
    assert "remains" in result.detail


def test_run_gate_skipped_when_tool_missing():
    gate = health.Gate("bandit", "extra", ["bandit"], "report")
    run = _scripted_runner([(127, "", "command not found: bandit")])
    result = health.run_gate(gate, run)
    assert result.status == "skipped"


def test_run_gate_prepare_failure_skips():
    gate = health.Gate(
        "pip-audit", "deps", ["pip-audit", "-r", "x"], "report",
        prepare=[["uv", "export"]],
    )
    run = _scripted_runner([(127, "", "command not found: uv")])
    result = health.run_gate(gate, run)
    assert result.status == "skipped"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: FAIL — `run_gate` not defined.

- [ ] **Step 3: Write minimal implementation**

Add import at top of `health.py` (one per line):

```python
from collections.abc import Callable
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/python_health/scripts/health.py tools/python_health/scripts/tests/test_health.py
git commit -m "feat(python-health): gate runner with auto-fix flow"
```

---

### Task 4: Report rendering + coverage parsing + overall exit

**Files:**
- Modify: `tools/python_health/scripts/health.py`
- Modify: `tools/python_health/scripts/tests/test_health.py`

**Interfaces:**
- Consumes: `GateResult` (Task 2).
- Produces:
  - `overall_exit(results: list[GateResult]) -> int` → `1` if any non-advisory result has status `fail` or `error`, else `0`.
  - `render_summary(results: list[GateResult]) -> str` (human-readable).
  - `results_to_json(results: list[GateResult]) -> str`.
  - `parse_coverage(stdout: str) -> float | None` — reads the `TOTAL ... NN%` line from pytest-cov term output.

- [ ] **Step 1: Write the failing test**

```python
def test_overall_exit_advisory_does_not_fail():
    results = [
        health.GateResult("ruff-format", "fixed", "auto-fixed"),
        health.GateResult("outdated", "fail", "stale", advisory=True),
    ]
    assert health.overall_exit(results) == 0


def test_overall_exit_hard_fail():
    results = [health.GateResult("ty", "fail", "needs attention")]
    assert health.overall_exit(results) == 1


def test_render_summary_lists_each_gate():
    results = [
        health.GateResult("ruff-format", "fixed", "auto-fixed"),
        health.GateResult("ty", "fail", "needs attention"),
    ]
    text = health.render_summary(results)
    assert "ruff-format" in text
    assert "FIXED" in text
    assert "FAIL" in text


def test_parse_coverage_reads_total():
    out = "TOTAL                      1234    200    84%\n"
    assert health.parse_coverage(out) == 84.0


def test_parse_coverage_none_when_absent():
    assert health.parse_coverage("no coverage here") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Write minimal implementation**

Add imports at top of `health.py` (one per line):

```python
import json
import re
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/python_health/scripts/health.py tools/python_health/scripts/tests/test_health.py
git commit -m "feat(python-health): report rendering, coverage parse, exit aggregation"
```

---

### Task 5: CLI `main()` + subprocess runner + collect-all / --fail-fast

**Files:**
- Modify: `tools/python_health/scripts/health.py`
- Modify: `tools/python_health/scripts/tests/test_health.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `subprocess_runner(cmd: list[str]) -> tuple[int, str, str]` (default real runner; `FileNotFoundError` ⇒ `(127, "", "command not found: <cmd0>")`).
  - `load_pyproject(root: Path) -> dict` (uses `tomllib`).
  - `run_all(gates: list[Gate], run: CommandRunner, fail_fast: bool = False) -> list[GateResult]`. Isolates gate exceptions: if `run_gate` raises, the gate is recorded as a `GateResult` with status `"error"` (spec §7 — one gate's crash must not abort the rest), and the loop continues (or breaks on `--fail-fast` for a non-advisory error).
  - `main(argv: list[str] | None = None) -> int` — flags: `--root PATH` (default `.`), `--fail-fast`, `--json`, `--no-fix` (treat fixable gates as report-only). Prints summary (or JSON) and returns `overall_exit(...)`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_all_collects_all_by_default():
    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    run = _scripted_runner([(1, "boom", ""), (0, "", "")])
    results = health.run_all(gates, run, fail_fast=False)
    assert [r.status for r in results] == ["fail", "pass"]


def test_run_all_fail_fast_stops_after_first_hard_fail():
    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    run = _scripted_runner([(1, "boom", "")])
    results = health.run_all(gates, run, fail_fast=True)
    assert [r.status for r in results] == ["fail"]


def test_run_all_isolates_gate_exception_as_error():
    def boom(cmd):
        raise OSError("kaboom")

    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    results = health.run_all(gates, boom, fail_fast=False)
    assert [r.status for r in results] == ["error", "error"]
    assert "kaboom" in results[0].detail


def test_overall_exit_error_status_hard_fails():
    results = [health.GateResult("x", "error", "gate raised")]
    assert health.overall_exit(results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: FAIL — `run_all` not defined.

- [ ] **Step 3: Write minimal implementation**

Add imports at top of `health.py` (one per line):

```python
import argparse
import subprocess
import sys
from pathlib import Path
```

```python
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


def run_all(
    gates: list[Gate], run: CommandRunner, fail_fast: bool = False
) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in gates:
        try:
            result = run_gate(gate, run)
        except Exception as exc:  # gate isolation (spec §7): a crash must not abort the rest
            result = GateResult(gate.name, "error", "gate raised", str(exc), gate.advisory)
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
        gates = [
            Gate(g.name, g.phase, g.check, "report", None, g.prepare, g.advisory)
            for g in gates
        ]

    results = run_all(gates, subprocess_runner, fail_fast=args.fail_fast)

    if args.json:
        print(results_to_json(results))
    else:
        print(render_summary(results))
    return overall_exit(results)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/python_health/scripts/tests/test_health.py -v`
Expected: PASS (full file green).

- [ ] **Step 5: Verify the orchestrator dogfoods clean, then commit**

Run: `uv run ruff format tools/python_health/ && uv run ruff check --fix tools/python_health/ && uv run ty check tools/python_health`
Expected: ruff reports formatted/clean; `ty check tools/python_health` → "All checks passed!".

NOTE: scope the dogfood `ty` check to `tools/python_health` — a repo-wide `uv run ty check` reports ~48 PRE-EXISTING diagnostics in the existing `vspeech` code that are out of scope for this task (they are surfaced for triage in Task 8, not fixed here).

```bash
git add tools/python_health/scripts/health.py tools/python_health/scripts/tests/test_health.py
git commit -m "feat(python-health): CLI main, subprocess runner, run_all"
```

---

### Task 6: Author `SKILL.md`

**Files:**
- Create: `tools/python_health/SKILL.md`

**Interfaces:**
- Consumes: the finished `scripts/health.py` (invoked as `scripts/health.py` in the installed layout).
- Produces: the skill manifest Claude follows.

- [ ] **Step 1: Write `SKILL.md`**

````markdown
---
name: python-health
description: Use to assure the health of a uv-based Python project on-demand — runs ruff (format/lint), ty, pytest+coverage, uv lock --check, pip-audit, bandit, and vulture as gates, auto-fixes only mechanical issues, and reports the rest for triage. Use when asked to check project health, run quality gates, verify a Python project is clean before committing, or after finishing a change.
---

# Python Health

Run the project's health gates in this environment and triage the results. Mechanical issues are auto-fixed; substantive issues are reported for your judgment — never silently changed.

## Procedure

1. Confirm the working directory is a uv project (`pyproject.toml` + `uv.lock` exist). If not, say so and stop.
2. Run the orchestrator from the project root:
   - Default (auto-fix mechanical issues): `uv run python <skill-dir>/scripts/health.py --root .`
   - Report-only (no edits): add `--no-fix`.
   - Machine-readable: add `--json`.
   The script exits non-zero if any non-advisory gate fails.
3. Read the summary. For each gate:
   - **PASS / FIXED / SKIP** — note it; FIXED means `ruff format` / `ruff check --fix` already edited files. Show the user `git diff` so they can review or revert.
   - **FAIL (ty, pytest-cov, uv-lock-check, pip-audit, bandit)** — do NOT auto-edit. Investigate the root cause, explain it, and propose a fix for approval. For `pip-audit`, surface the CVE and the fixed version. For `uv-lock-check`, the fix is usually `uv lock`.
   - **FAIL/advisory (outdated, vulture)** — report as advisory; don't block.
4. Present a short triage list: what was auto-fixed, what needs the user's decision, and your recommended next action for each.

## Rules

- Only `ruff format` and `ruff check --fix` (safe fixes) are ever applied automatically. Everything else is report-only.
- Never modify `addopts` or other project config to make a gate pass.
- A missing optional tool/extra is a SKIP, not a failure — keep going.
- Coverage is advisory: report the TOTAL %, compare to baseline if one exists, warn on regression — don't hard-fail.
````

- [ ] **Step 2: Commit**

```bash
git add tools/python_health/SKILL.md
git commit -m "docs(python-health): SKILL.md manifest and triage procedure"
```

---

### Task 7: Author `references/gate-catalog.md`

**Files:**
- Create: `tools/python_health/references/gate-catalog.md`

- [ ] **Step 1: Write `references/gate-catalog.md`**

```markdown
# Gate Catalog

Each gate is `(name, phase, check command, kind, advisory)`. `kind=fixable` gates auto-apply a fix and re-check; `kind=report` gates never edit files.

| Gate | Phase | Command | Kind | Notes |
|---|---|---|---|---|
| ruff-format | static | `uv run ruff format --check .` | fixable | fix: `ruff format .` |
| ruff-lint | static | `uv run ruff check .` | fixable | fix: `ruff check --fix .` (safe only; never `--unsafe-fixes`) |
| ty | static | `uv run ty check` | report | type errors are never auto-fixed |
| pytest-cov | tests | `uv run pytest --cov=<pkg> --cov-report=term-missing` | report | honors project `addopts` (e2e excluded) |
| uv-lock-check | deps | `uv lock --check` | report | fix is `uv lock` (proposed, not auto-run) |
| pip-audit | deps | `uvx pip-audit -r <exported-reqs>` | report | prepared via `uv export`; surfaces known CVEs |
| outdated | deps | `uv pip list --outdated` | report (advisory) | informational; never blocks |
| bandit | extra | `uvx bandit -q -r <pkg>` | report | security lint |
| vulture | extra | `uvx vulture <pkg> --min-confidence 80` | report (advisory) | dead-code; high min-confidence to cut false positives |

## Auto-fix policy

- **Auto-applied** (deterministic, reversible, no logic change): `ruff format`, `ruff check --fix` (safe fixes).
- **Never auto-applied** (need human judgment): ty errors, test failures, dependency upgrades, pip-audit/bandit findings, vulture deletions.

## False-positive tuning

- bandit: scan only the package dir(s); excludes test code by default. Add a `# nosec` with justification for accepted findings.
- vulture: `--min-confidence 80` and treat as advisory. Maintain a project allowlist if needed.

## Coverage baseline

- First run records the TOTAL % as the baseline (`references/coverage-baseline.json`) only when `--update-baseline` is passed (future flag).
- Subsequent runs warn if TOTAL drops below baseline. Coverage never hard-fails in v1.

## Status semantics

`pass` ok · `fixed` mechanical fix applied · `fail` needs attention · `skipped` tool/extra unavailable · `error` gate could not run. Advisory gates never affect the overall exit code.
```

- [ ] **Step 2: Commit**

```bash
git add tools/python_health/references/gate-catalog.md
git commit -m "docs(python-health): gate catalog and policies"
```

---

### Task 8: Validate on this repo (real run) + triage

**Files:**
- No source changes expected (this is a verification task). May produce auto-fix commits if the repo has format/lint drift.

- [ ] **Step 1: Run the full suite (the orchestrator's own unit tests)**

Run: `uv run pytest tools/python_health/scripts/tests/ -v`
Expected: PASS (all tasks' tests green).

- [ ] **Step 2: Run the orchestrator against this repo, report-only first**

Run: `uv run python tools/python_health/scripts/health.py --root . --no-fix`
Expected: a summary listing all 9 gates. Confirm `pytest-cov` used `--cov=vspeech` and that `voicevox_e2e` tests were excluded (no VOICEVOX asset errors).

- [ ] **Step 3: Triage the report**

For each non-PASS gate, write down the finding. Auto-fixable drift (ruff) → proceed to Step 4. Substantive findings (ty errors, failing tests, CVEs, lock drift) → list them with a proposed fix; do NOT fix them as part of this task — surface to the user.

EXPECTED pre-existing findings (this validates the premise — surface, do not fix): the `ty` gate will report ~48 diagnostics in the existing `vspeech` code; the `pytest-cov` gate runs the existing suite (some pre-existing failures/warnings possible). The skill's job here is to detect and report these accurately, NOT to clean up the pre-existing codebase. Only ruff format/lint drift is auto-fixed.

- [ ] **Step 4: Run with auto-fix and review the diff**

Run: `uv run python tools/python_health/scripts/health.py --root .`
Then: `git diff`
Expected: any changes are limited to `ruff format` / `ruff check --fix` output. If the diff touches logic, STOP — that is a bug in the gate definition; investigate.

- [ ] **Step 5: Commit (only if ruff auto-fixed repo files)**

```bash
git add -A
git commit -m "style: ruff format/lint fixes surfaced by python-health"
```

If there was nothing to fix, skip the commit and note "repo already clean".

---

### Task 9: Validate generalization on `vstreamer-protos/python`

**Files:**
- No changes to this repo; read-only validation against the second project.

- [ ] **Step 1: Run report-only against the second project**

Run: `uv run python tools/python_health/scripts/health.py --root ../vstreamer-protos/python --no-fix`
(Adjust the path to the actual `vstreamer-protos` python package root — it must contain a `pyproject.toml`.)
Expected: gates run; `derive_targets` picks a sensible `--cov=<pkg>` for that project (verify the package name in the printed `pytest-cov` invocation or `--json` output).

- [ ] **Step 2: Note any generalization gaps**

If `derive_targets` produced the wrong package, or a gate assumed a vspeech-specific layout, record it. Small fixes go back through Task 1–5 TDD; otherwise note as a known limitation in `references/gate-catalog.md`.

- [ ] **Step 3: Commit any generalization fixes**

```bash
git add tools/python_health/
git commit -m "fix(python-health): generalize target detection across projects"
```

(Skip if no changes were needed.)

---

### Task 10: Install to the user skills directory

**Files:**
- Create: `~/.claude/skills/python-health/SKILL.md`
- Create: `~/.claude/skills/python-health/references/gate-catalog.md`
- Create: `~/.claude/skills/python-health/scripts/health.py`

- [ ] **Step 1: Copy the skill into the user skills dir (exclude dev tests)**

Run (PowerShell):
```powershell
$dst = "$env:USERPROFILE\.claude\skills\python-health"
New-Item -ItemType Directory -Force "$dst\scripts","$dst\references" | Out-Null
Copy-Item tools\python_health\SKILL.md $dst\SKILL.md -Force
Copy-Item tools\python_health\references\gate-catalog.md $dst\references\gate-catalog.md -Force
Copy-Item tools\python_health\scripts\health.py $dst\scripts\health.py -Force
```
Expected: `~/.claude/skills/python-health/` contains `SKILL.md`, `references/gate-catalog.md`, `scripts/health.py` — and NOT `scripts/tests/`.

- [ ] **Step 2: Verify the installed script runs standalone**

Run: `uv run python "$env:USERPROFILE\.claude\skills\python-health\scripts\health.py" --root . --no-fix --json`
Expected: valid JSON array of gate results — confirms the file works when run from its installed path (no package-relative imports).

- [ ] **Step 3: Verify the skill is discoverable**

In a new Claude Code session, confirm `python-health` appears in the available skills list and its `description` reads correctly. (No code change — this is the acceptance check that the skill is installed.)

- [ ] **Step 4: Final commit (repo source of truth)**

```bash
git add tools/python_health/
git commit -m "feat(python-health): complete skill (source in-repo, installed to user skills)"
```

---

## Self-Review

**1. Spec coverage** (against `2026-06-24-python-health-skill-design.md`):
- §2 approach A (auto-fix mechanical / report substantive) → Task 3 (`run_gate`) + Task 5 (`--no-fix`) + Task 6 SKILL rules. ✅
- §3 skill form/placement (user-level, SKILL.md + scripts/health.py + references) → Tasks 6, 7, 10. ✅
- §4 full gate pipeline (9 gates, cheap→expensive) → Task 2 `build_gates`. ✅
- §5 auto-fix policy → Task 3 + Task 7 catalog. ✅
- §6 coverage baseline/warn-only → Task 4 `parse_coverage`, Task 7 (baseline doc; `--update-baseline` flagged future). ✅
- §7 error handling (skip on missing tool, gate isolation, fail-fast) → Task 3 (skipped) + Task 5 (`run_all`, `--fail-fast`). ✅
- §8 generalization (derive targets, no `vspeech` hardcode, 2nd repo) → Task 1 + Task 9. ✅
- §9 validation (this repo, break-and-classify, 2nd repo) → Tasks 8, 9; classification covered by unit tests in Tasks 2–5. ✅
- §10 real-world examples — documentation in the spec, not implemented (correct). ✅
- §11 scope-out (no GitHub Actions / pre-commit / hard coverage gate / auto-upgrade) → honored; none of those appear as tasks. ✅

**2. Placeholder scan:** `<pkg>`, `<skill-dir>`, `<exported-reqs>` are runtime/install-relative values documented at point of use; no TBD/TODO/"handle edge cases" left. ✅

**3. Type consistency:** `Targets(packages, project_name)`, `Gate(name, phase, check, kind, fix, prepare, advisory)`, `GateResult(name, status, summary, detail, advisory)`, `CommandRunner = Callable[[list[str]], tuple[int, str, str]]`, and function names (`derive_targets`, `classify`, `build_gates`, `run_gate`, `run_all`, `overall_exit`, `render_summary`, `results_to_json`, `parse_coverage`, `subprocess_runner`, `load_pyproject`, `main`) are used consistently across Tasks 1–10. ✅
```
