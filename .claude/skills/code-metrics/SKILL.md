---
name: code-metrics
description: Use to survey a uv-based Python project's complexity metrics and surface ranked refactor candidates — runs lizard (cyclomatic) + complexipy (cognitive) + a beniget DepDegree (def-use coupling) lens, joins them per function, and explains which functions are worth refactoring vs. merely branchy. Advisory only: never gates, never edits code. Use when asked where to refactor, to find complexity hotspots, to assess a module's maintainability, or to find functions with tangled/entangled mutable state.
---

# Code Metrics

Survey complexity with three complementary lenses and rank refactor candidates.
Advisory only — this never gates a commit and never edits source.

- **Cyclomatic (CCN, via lizard)** — independent paths → a testability proxy. Blunt;
  over-flags wide-but-flat dispatch.
- **Cognitive (via complexipy)** — penalizes nesting/broken flow, ignores flat
  `match` and trivial accessors → a readability proxy.
- **DepDegree (`dep`, via beniget)** — counts def-use dependency edges among a function's
  local variables → a data-flow-entanglement proxy. Catches the case the first two MISS:
  a flat loop threading many co-mutated locals (high `dep`, modest cog/ccn).

The first two are control-flow metrics; `dep` is the orthogonal data-flow axis. Together
they separate "many branches but easy to read" (high CCN, low cognitive — a flat
dispatcher) from "genuinely tangled" (high cognitive) from "entangled mutable state"
(high `dep`). See `references/metric-guide.md` for the model, thresholds, and tool notes.

## Procedure

1. Confirm the working directory is a uv project (`pyproject.toml` exists). If not,
   say so and stop.
2. Run it via the project's poe task (the orchestrator lives at `scripts/metrics.py`,
   wired as `[tool.poe.tasks].metrics`):
   `uv run poe metrics`
   - Machine-readable: `uv run poe metrics --json`. Its stdout is clean JSON; poe's
     `Poe =>` banner and any sync noise go to stderr, so pipe stdout only
     (`uv run poe metrics --json 2>/dev/null`).
   - Show more/all rows: append `--top 30` (or `--top 0` for all flagged).
   - Tune bands: `--ccn-warn`, `--ccn-high`, `--cog-warn`, `--dep-warn`.

   If a target project lacks the task, run the script directly. The `dep` lens needs
   `beniget` (a dev dependency here); when running outside this repo's venv, add it:
   `uv run --with beniget python scripts/metrics.py`. Or add the one-liner:
   `metrics = { cmd = "python scripts/metrics.py" }`.
3. Read the ranked table and the prose buckets:
   - **both-high** → top refactor target (tangled AND branchy).
   - **high-cognitive** → "sneaky" target (deep nesting hides in few paths).
   - **entangled** → high def-use coupling the cog/ccn lenses under-rank (flat loop,
     many co-mutated locals). The "Entangled state" line lists these — a genuinely new
     signal, not redundant with the two above.
   - **high-ccn** → likely fine (flat dispatch); de-prioritize — UNLESS it also appears in
     the entangled line, which refutes the "flat = simple" read.
4. Present the top candidates with `file:line`, the three scores, and a one-line
   reason each. Use `dep` to break ties the cognitive ranking gets wrong (e.g. a flat,
   stateful loop that out-ranks a deeply-nested-but-simple dispatcher). Do NOT edit code —
   this is input for the user's refactoring decision.

## Notes

- lizard and complexipy run via `uvx` (own parsers; they neither import the project nor
  read `pyproject`). The `dep` lens runs **in-process** via the `beniget` + `gast`
  libraries — it only parses source text (it does not import the project either), and gast
  parses 3.11 `except*` / `async def`. It computes `dep` over the package source files the
  script walks (skipping `__pycache__`/`.venv`/dotted dirs).
- complexipy output is read from a JSON file with `PYTHONIOENCODING=utf-8`, never from
  stdout (its rich/emoji stdout crashes a Windows cp932 console under capture).
- A missing analyzer is a SKIP for that lens (printed), not a failure — including
  `beniget` missing (the `dep` column shows `-`). The exit code is always 0.

## Rules

- Never edit source or project config. This skill only reports.
- Never present the metric as a verdict — CCN (lizard, heuristic) and cognitive
  (complexipy) are advisory rankings, not gates. The user decides what to refactor.
