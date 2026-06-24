# Design: `code-metrics` insight skill

- **Date:** 2026-06-24
- **Status:** Approved design (pre-implementation)
- **Author:** sondeko (with Claude Code)
- **Related:** `python-health` skill (`tools/python_health/`) — this skill mirrors its scaffolding but is advisory-only and never gates.

## Goal

A reusable, on-demand skill that surveys a uv-based Python project's code-complexity
metrics and produces a **ranked list of refactor candidates** with an interpretation
of *why* each is a candidate. It answers "where does refactoring actually pay off?"
— not "did a gate pass?".

## Non-goals

- **Not a gate.** It never returns a non-zero "failed" status and never blocks a
  commit/CI. Cyclomatic *gating* is already a one-liner (`ruff` `C901` +
  `[tool.ruff.lint.mccabe] max-complexity`); that belongs in `python-health`, not here.
- **No code edits.** Report-only, like the report-kind gates in `python-health`.
- **No Maintainability Index.** MI is widely criticised (arbitrary/uncalibrated
  constants, LOC-dominated, averaging hides hotspots; radon's own docs call it
  "experimental"). Deliberately excluded.
- **No git-history trend tooling (wily).** Stale since 2023, needs a clean tree,
  history-heavy — overkill for a small/solo repo. (A churn overlay is noted under
  Future, but not in v1.)
- **No SaaS** (SonarQube/Qlty/Code Climate). Overkill unless the repo goes public.

## Background: why these two tools (the two-lens model)

The skill reports two complementary complexity lenses per function:

- **Cyclomatic complexity (CCN)** — independent paths through a function → a
  *testability* proxy. Blunt: over-flags wide-but-flat dispatch code.
- **Cognitive complexity** — penalises nesting and broken control flow, ignores flat
  `match`/`switch` and trivial accessors → a *readability* proxy.

Reporting both lets the skill separate "many branches but easy to read" (high CCN,
low cognitive — a flat dispatcher) from "genuinely tangled" (high cognitive). That
distinction is exactly the judgement "is this worth refactoring?" needs.

### Tool selection (empirically verified on this project, 2026-06-24)

| Lens | Tool | Why |
|---|---|---|
| Cyclomatic + size | **lizard** | Actively maintained; own tokenizing parser (no project import, no `pyproject` read); clean CSV with **line numbers**, CCN, NLOC, token & parameter counts; multi-language. Verified: parsed the whole `vspeech` package incl. `main.py`'s 3.11 `except*` with zero errors. |
| Cognitive | **complexipy** | Actively maintained Rust-backed cognitive-complexity tool; the live successor to the dead `flake8-cognitive-complexity`. JSON output via `--output-format json --output <file>`. Verified: parsed `except*` fine. |

**radon was rejected** after testing on this repo:
- Under the project interpreter (`uv run --with radon`) it crashes at startup —
  its `configparser`-based `FileConfig` reads `pyproject.toml` and chokes on the
  `%2B` in the pinned wheel URLs under `[tool.uv.sources]`
  (`ValueError: invalid interpolation syntax`).
- Under `uvx radon` (neutral cwd, avoids the crash) it runs on a non-project
  interpreter and reports `ERROR: invalid syntax` for `except*` files.
- It is in maintenance-mode (last release 6.0.1, 2023). lizard covers cyclomatic +
  LOC and is not fragile here, so radon is dropped entirely.

## Empirical gotchas baked into the design

1. **complexipy console encoding (Windows).** Its default rich output emits an emoji
   (`✅`) that crashes under a cp932/cp1252 console when captured by a subprocess
   (`UnicodeEncodeError: '✅'`). Mitigation: always invoke with `-q`
   `--output-format json --output <tmpfile>` and **read the JSON file**, never parse
   stdout; also set `PYTHONIOENCODING=utf-8` in the child env as belt-and-suspenders.
   (Same class of issue the `windows-shell-encoding` skill addresses.)
2. **complexipy has no line numbers** and names functions as `Class::method`. lizard
   provides the line numbers; the join keys off `(file, simple-name)`.
3. **Both tools run via `uvx`** (neither imports project code nor reads `pyproject`),
   so the skill needs no project venv and no `--extras` plumbing — unlike the
   env-sensitive `ty`/`pytest` gates in `python-health`.

## Architecture

Mirrors `python-health`'s structure and conventions.

### Placement / distribution
- **Source of truth (committed):** `tools/code_metrics/`
  - `SKILL.md`
  - `scripts/metrics.py`
  - `scripts/tests/test_metrics.py`
  - `references/metric-guide.md` (the two-lens model, thresholds, how to read buckets)
- **Installed (not committed):** copied to `~/.claude/skills/code-metrics/`; re-copy
  on change. Same workflow as `python-health`.
- **Generic across any uv project** — no vspeech-specific assumptions.

### Orchestrator (`metrics.py`) flow
1. Confirm a uv project (`pyproject.toml` exists). Read it with **`tomllib`**
   (not configparser — avoids the radon-class crash) to derive target packages.
2. **Derive targets** — a self-contained copy of `python-health`'s `derive_targets`
   precedence (skills install independently, so no shared import):
   build-backend `module-name` → `[project.scripts]` top package →
   `[tool.poetry].packages[].include` → normalized `[project].name` → `.`.
3. **Run analyzers** (from project root; absolute or package-relative paths):
   - `uvx lizard <pkgs> --csv`
   - `uvx complexipy <pkgs> -q --output-format json --output <tmpfile>`
     with `PYTHONIOENCODING=utf-8` in the child env.
   A missing tool / spawn failure is a **SKIP** for that lens (degrade, don't crash),
   matching `python-health`'s `classify`.
4. **Parse**:
   - lizard CSV (stdlib `csv`): columns are
     `nloc, ccn, token_count, param, length, "name@start-end@file", file, function_name, signature, start_line, end_line`.
   - complexipy JSON (stdlib `json`): list of
     `{complexity, file_name, function_name ("Class::method"), path, refactor_plans}`.
5. **Join** into unified records keyed off lizard (it has line numbers):
   `{file, function, line, ccn, nloc, params, cognitive|None}`.
   Match complexipy by `(normalized_path, simple_name)` where `simple_name` strips
   any `Class::` prefix from complexipy and matches lizard's `function_name`.
   Normalize path separators (`\` ↔ `/`). On simple-name collisions within a file,
   match best-effort and leave `cognitive=None` if ambiguous; never crash. Records
   present only in complexipy (lizard didn't list them) are appended with
   `ccn=None`. Unmatched functions are reported with the lens(es) available.
6. **Rank & bucket**:
   - Sort by `cognitive` desc, tiebreak `ccn` desc (None sorts last).
   - Thresholds (advisory bands): `ccn > 10` (watch) / `> 20` (high);
     `cognitive > 15` (complexipy default cap).
   - Buckets:
     - **both-high** (`ccn` high *and* `cognitive` high) → top refactor target.
     - **high-CCN-only** (`ccn` high, `cognitive` low) → likely a flat dispatcher;
       de-prioritize.
     - **high-cognitive-only** → "sneaky" target (nesting hides in few paths).
7. **Render**:
   - Human summary: ranked table `cog · ccn · nloc · params · function · file:line ·
     [bucket]` (top ~15 or all flagged) — no A–F letter grade, since radon (its
     source) is dropped; the bucket label carries the "how bad" signal instead.
     Followed by a short prose
     "refactor candidates & why" derived from the buckets.
   - `--json`: machine-readable list of the unified records + bucket labels,
     mirroring `python-health`'s `--json`.

### CLI flags
- `--root <dir>` (default `.`)
- `--json` (machine-readable; default is the human summary)
- `--top N` (default 15; `0` = all flagged)
- `--ccn-warn` / `--ccn-high` / `--cog-warn` (override thresholds; sensible defaults)

## Testing

`scripts/tests/test_metrics.py` mirrors `test_health.py`:
- Inject a fake command runner returning **captured fixture output** (a real
  lizard `--csv` block and a real complexipy JSON list).
- Assert the pure pipeline: `parse_lizard_csv` → `parse_complexipy_json` → `join` →
  `rank/bucket` → `render`. No live tool invocation in tests.
- Cover the join edge cases explicitly: path-separator normalization,
  `Class::method` stripping, a function present in only one lens, a simple-name
  collision, and a missing-tool SKIP.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| complexipy emoji crash on Windows console | `-q --output-format json --output <file>` + `PYTHONIOENCODING=utf-8`; read file, not stdout. |
| Join imperfect for same-named methods across classes | Best-effort `(file, simple-name)`; leave `cognitive=None` on ambiguity; report the gap rather than guess. |
| CCN (lizard, heuristic) differs slightly from AST-based CC | Acceptable — this is an advisory *ranking*, not a gate; documented in `references/`. |
| Tool API drift (complexipy is young) | Pin nothing in the skill, but tests use captured fixtures; the live invocation is isolated to one function so a flag change is a one-line fix. |

## Future (explicitly out of v1)

- **Churn overlay** ("hotspot ROI"): rank by complexity × git change-frequency to
  surface where complexity actually hurts. Powerful at team scale; deferred for solo.
- **Diff mode**: complexipy has a built-in `--diff <ref>`; could surface
  "complexity added by this branch" later.
