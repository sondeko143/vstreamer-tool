---
name: code-metrics
description: Use to survey a uv-based Python project's complexity metrics and surface ranked refactor candidates — runs lizard (cyclomatic) + complexipy (cognitive), joins them per function, and explains which functions are worth refactoring vs. merely branchy. Advisory only: never gates, never edits code. Use when asked where to refactor, to find complexity hotspots, or to assess a module's maintainability.
---

# Code Metrics

Survey complexity with two complementary lenses and rank refactor candidates.
Advisory only — this never gates a commit and never edits source.

- **Cyclomatic (CCN, via lizard)** — independent paths → a testability proxy. Blunt;
  over-flags wide-but-flat dispatch.
- **Cognitive (via complexipy)** — penalizes nesting/broken flow, ignores flat
  `match` and trivial accessors → a readability proxy.

Reporting both separates "many branches but easy to read" (high CCN, low cognitive —
a flat dispatcher) from "genuinely tangled" (high cognitive). See
`references/metric-guide.md` for the model, thresholds, and tool notes.

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
   - Tune bands: `--ccn-warn`, `--ccn-high`, `--cog-warn`.

   If a target project lacks the task, run the script directly
   (`uv run python scripts/metrics.py`) or add the one-liner:
   `metrics = { cmd = "python scripts/metrics.py" }`.
3. Read the ranked table and the prose buckets:
   - **both-high** → top refactor target (tangled AND branchy).
   - **high-cognitive** → "sneaky" target (deep nesting hides in few paths).
   - **high-ccn** → likely fine (flat dispatch); de-prioritize.
4. Present the top candidates with `file:line`, both scores, and a one-line
   reason each. Do NOT edit code — this is input for the user's refactoring decision.

## Notes

- Both analyzers run via `uvx` (own parsers; they neither import the project nor read
  `pyproject`), so no project venv or extras are needed, and 3.11 `except*` parses.
- complexipy output is read from a JSON file with `PYTHONIOENCODING=utf-8`, never from
  stdout (its rich/emoji stdout crashes a Windows cp932 console under capture).
- A missing analyzer is a SKIP for that lens (printed), not a failure. The exit code is
  always 0.

## Rules

- Never edit source or project config. This skill only reports.
- Never present the metric as a verdict — CCN (lizard, heuristic) and cognitive
  (complexipy) are advisory rankings, not gates. The user decides what to refactor.
