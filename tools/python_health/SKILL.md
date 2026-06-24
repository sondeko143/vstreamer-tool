---
name: python-health
description: Use to assure the health of a uv-based Python project on-demand — runs ruff (format/lint), ty, pytest+coverage, uv lock --check, pip-audit, bandit, and vulture as gates, auto-fixes only mechanical issues, and reports the rest for triage. Use when asked to check project health, run quality gates, verify a Python project is clean before committing, or after finishing a change.
---

# Python Health

Run the project's health gates in this environment and triage the results. Mechanical issues are auto-fixed; substantive issues are reported for your judgment — never silently changed.

## Procedure

1. Confirm the working directory is a uv project (`pyproject.toml` + `uv.lock` exist). If not, say so and stop.
2. Run the orchestrator from the project root:
   - Default (auto-fix mechanical issues): `uv run --with tomli python <skill-dir>/scripts/health.py --root .`
   - Report-only (no edits): add `--no-fix`.
   - Machine-readable: add `--json`.
   The script exits non-zero if any non-advisory gate fails.
   Note: `--with tomli` lets the orchestrator parse pyproject on projects whose Python is < 3.11 (stdlib `tomllib` is 3.11+).
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
