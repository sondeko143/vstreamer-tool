---
name: python-health
description: Use to assure the health of a uv-based Python project on-demand — runs ruff (format/lint), ty, pytest+coverage, uv lock --check, pip-audit, bandit, and vulture as gates, auto-fixes only mechanical issues, and reports the rest for triage. Use when asked to check project health, run quality gates, verify a Python project is clean before committing, or after finishing a change.
---

# Python Health

Run the project's health gates in this environment and triage the results. Mechanical issues are auto-fixed; substantive issues are reported for your judgment — never silently changed.

## Procedure

1. Confirm the working directory is a uv project (`pyproject.toml` + `uv.lock` exist). If not, say so and stop. If the project declares **optional/GPU extras** that the scanned code imports (e.g. `whisper`/`rvc`/`audio`), plan to pass `--extras` in step 2 — gates resolve imports against the *installed* env, and without it `ty`/`pytest` report env artifacts as defects (see **Environment artifacts** below).
2. Run the orchestrator from the project root:
   - Default (auto-fix mechanical issues): `uv run --with tomli python <skill-dir>/scripts/health.py --root .`
   - With extras (GPU/ML projects): add `--extras all` (or `--extras whisper,rvc`) — injects `--all-extras`/`--extra` into the `uv run` of every gate so `ty`/`pytest` resolve optional backends.
   - Report-only (no edits): add `--no-fix`.
   - Machine-readable: add `--json`.
   The script exits non-zero if any non-advisory gate fails.
   Note: `--with tomli` lets the orchestrator parse pyproject on projects whose Python is < 3.11 (stdlib `tomllib` is 3.11+).
3. Read the summary. For each gate:
   - **PASS / FIXED / SKIP** — note it; FIXED means `ruff format` / `ruff check --fix` already edited files. Show the user `git diff` so they can review or revert.
   - **FAIL (ty, pytest-cov, uv-lock-check, pip-audit, bandit)** — do NOT auto-edit. Investigate the root cause, explain it, and propose a fix for approval. For `pip-audit`, surface the CVE and the fixed version. For `uv-lock-check`, the fix is usually `uv lock`.
   - **FAIL/advisory (outdated, vulture)** — report as advisory; don't block.
4. Present a short triage list: what was auto-fixed, what needs the user's decision, and your recommended next action for each.

## Environment artifacts (check FIRST, before triaging a `ty` or `pytest` FAIL)

Gates resolve against the **installed** env, so an incomplete sync masquerades as code defects. These symptoms are almost always env artifacts, not findings:

- **A flood of `unresolved-import` diagnostics**, especially for optional/GPU backends (`torch`, `torchaudio`, `onnxruntime`, `fairseq`, `faster_whisper`, `pyworld`) → that extra isn't installed; not a type error.
- **A pytest *collection* error** — `ModuleNotFoundError` at import time, `collected N items / 1 error` → same cause: a test imports an extra-only module at module level.
- **An onnxruntime "binding error" / silent CPU fallback at runtime** → `onnxruntime-gpu` (an extra) isn't installed.

**Action: re-run with the extras installed; only what survives is a real finding.**
- Via the orchestrator: add `--extras all` (or `--extras whisper,rvc`) to the step-2 command.
- To check one gate by hand: `uv run --all-extras ty check` / `uv run --all-extras pytest`.

A bare `uv run` re-syncs to the *default* env and strips extras, so a standalone `uv sync --all-extras` does **not** stick across the orchestrator's gates — the extras must ride on the `uv run` itself (which is what `--extras` does). The IDE's ty language server may also resolve against a different env than `uv run ty` — trust the `--extras` `uv run` result. If an extra genuinely can't install on this platform, report its imports as *expected-absent / advisory* — do not change source or config to green the gate.

## Rules

- Only `ruff format` and `ruff check --fix` (safe fixes) are ever applied automatically. Everything else is report-only.
- Never modify project config or source to make a gate pass — this includes `addopts`, `[tool.ty]` overrides or `unresolved-import = "ignore"`, blanket `pytest.importorskip` / `# type: ignore`. Re-scoping a real diagnostic class to silence it is still making the gate pass, however "defensible" the framing.
- A missing optional tool/extra is a SKIP, not a failure — keep going.
- Coverage is advisory: report the TOTAL %, compare to baseline if one exists, warn on regression — don't hard-fail.
