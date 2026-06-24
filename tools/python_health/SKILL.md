---
name: python-health
description: Use to assure the health of a uv-based Python project on-demand — runs ruff (format/lint), ty, pytest+coverage, uv lock --check, uv audit, bandit, and vulture as poethepoet (`poe`) tasks, auto-fixing only mechanical issues and reporting the rest for triage. Use when asked to check project health, run quality gates, verify a Python project is clean before committing, or after finishing a change.
---

# Python Health

Run the project's health gates and triage the results. The gates are defined as **poethepoet (`poe`) tasks** in the project's `pyproject.toml`; this skill runs them and decides what to do with each result. Mechanical issues are auto-fixed; substantive issues are reported for your judgment — never silently changed.

There is no orchestrator script: `poe` sequences the commands, and *you* (the skill) own the triage — the "auto-fix vs. report" split and the environment-artifact reasoning below.

## Procedure

1. **Confirm it's a uv project** (`pyproject.toml` + `uv.lock` exist). If not, say so and stop.

2. **Ensure the poe health tasks exist.** Look for `[tool.poe.tasks]` with a `check` task in `pyproject.toml`. If absent, set them up first:
   - Add the runner: `uv add --dev poethepoet`.
   - Add the canonical block from **Task block** below, substituting the project's package name(s) into `test` / `security` / `deadcode`. Derive the package in this order: `[tool.uv.build-backend].module-name` → top package of a `[project.scripts]` entry → `[tool.poetry].packages[].include` → normalized `[project].name`/`[tool.poetry].name` (`-`→`_`) → fall back to `.`.

3. **Run the gates (report-only, collect-all):**
   - Default: `uv run poe check`
   - **GPU/ML projects** (optional extras like `whisper`/`rvc`/`audio` whose modules the scanned code imports): `uv run --all-extras poe check` (or `uv run --extra whisper --extra rvc poe check`). The extras MUST ride on the `uv run` that launches `poe` — a bare `uv run` re-syncs to the default env and strips them, so a standalone `uv sync --all-extras` does not stick. Without this, `ty`/`pytest` report env artifacts as defects (see **Environment artifacts**).

   `check` runs every gate even if one fails, and exits non-zero if any did.

4. **Triage each gate result:**
   - **fmt-check / lint FAIL** → mechanical. Run `uv run poe fix` (= `ruff format` + `ruff check --fix`, safe fixes only), then show the user `git diff` so they can review or revert.
   - **type / test / lock-check / audit / security FAIL** → do NOT auto-edit. Investigate the root cause, explain it, and propose a fix for approval. `uv audit` → surface the CVE and the fixed version. `uv lock --check` → the fix is usually `uv lock`.
   - **deadcode (vulture) / other advisory** → report as advisory; don't block.

5. **Present a short triage list:** what was auto-fixed, what needs the user's decision, and the recommended next action for each.

## Environment artifacts (check FIRST, before triaging a `ty` or `pytest` FAIL)

Gates resolve against the **installed** env, so an incomplete sync masquerades as code defects. These symptoms are almost always env artifacts, not findings:

- **A flood of `unresolved-import` diagnostics**, especially for optional/GPU backends (`torch`, `torchaudio`, `onnxruntime`, `fairseq`, `faster_whisper`, `pyworld`, `ttkbootstrap`, `pyaudio`) → that extra isn't installed; not a type error.
- **A pytest *collection* error** — `ModuleNotFoundError` at import time, `collected N items / 1 error` → same cause: a test imports an extra-only module at module level.
- **An onnxruntime "binding error" / silent CPU fallback at runtime** → `onnxruntime-gpu` (an extra) isn't installed.

**Action: re-run with the extras installed; only what survives is a real finding.**
- Whole suite: `uv run --all-extras poe check`.
- One gate by hand: `uv run --all-extras poe type` / `uv run --all-extras poe test`.

The extras must ride on the `uv run` itself (a bare `uv run` re-syncs to the default env and strips them). The IDE's ty language server may also resolve against a different env than `uv run` — trust the `--all-extras` `uv run` result. If an extra genuinely can't install on this platform, report its imports as *expected-absent / advisory* — do not change source or config to green the gate.

## Rules

- Only `uv run poe fix` (`ruff format` + `ruff check --fix`, safe fixes) is ever applied automatically. Everything else is report-only.
- Never modify project config or source to make a gate pass — this includes `addopts`, `[tool.ty]` overrides or `unresolved-import = "ignore"`, blanket `pytest.importorskip` / `# type: ignore`. Re-scoping a real diagnostic class to silence it is still making the gate pass, however "defensible" the framing.
- A missing optional tool/extra is a SKIP, not a failure — keep going and note it.
- Coverage is advisory: report the TOTAL % from the `test` gate, warn on regression — don't hard-fail.
- An accepted `bandit` finding is suppressed at the line with `# nosec <ID> - <reason>`, not by disabling the check globally.

## Task block (canonical `[tool.poe.tasks]`)

Add this to the target project's `pyproject.toml`, replacing `<pkg>` with the detected package (use `vspeech` in this repo). `<pkg2>` rows are optional (e.g. a `gui` package for bandit):

```toml
[tool.poe.tasks]
fmt = { cmd = "ruff format .", help = "Apply ruff formatting" }
fmt-check = { cmd = "ruff format --check .", help = "Check formatting (no writes)" }
lint = { cmd = "ruff check .", help = "Lint (no writes)" }
lint-fix = { cmd = "ruff check --fix .", help = "Apply safe lint fixes" }
type = { cmd = "ty check", help = "Type-check (ty)" }
test = { cmd = "pytest --cov=<pkg> --cov-report=term-missing", help = "Tests + coverage (honors addopts)" }
lock-check = { cmd = "uv lock --check", help = "Lockfile is in sync with pyproject" }
audit = { cmd = "uv audit", help = "Dependency vulnerability audit" }
security = { cmd = "uv run --with bandit bandit -q -r <pkg>", help = "Security lint (bandit, project interpreter)" }
deadcode = { cmd = "uv run --with vulture vulture <pkg> --min-confidence 80", help = "Dead-code scan (advisory)" }

[tool.poe.tasks.fix]
help = "Auto-apply mechanical fixes (ruff format + safe lint fixes)"
sequence = ["fmt", "lint-fix"]
default_item_type = "ref"

[tool.poe.tasks.check]
help = "Run every health gate report-only; keeps going if one fails, exits non-zero if any did"
sequence = ["fmt-check", "lint", "type", "test", "lock-check", "audit", "security", "deadcode"]
default_item_type = "ref"
ignore_fail = "return_non_zero"
```

Notes:
- `audit` uses native `uv audit` (uv 0.11+); on older uv, substitute `uv export --no-hashes --no-emit-project | uvx pip-audit -r -`.
- `security`/`deadcode` run via `uv run --with` so bandit/vulture execute under the **project interpreter** (so version-specific syntax like `except*` parses) without becoming project dependencies.
- `bandit`/`vulture` are advisory-leaning; tune `--min-confidence` and add `# nosec` per the Rules.
