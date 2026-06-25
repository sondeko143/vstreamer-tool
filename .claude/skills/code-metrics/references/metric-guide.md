# Metric guide

## The three-lens model

| Lens | Tool | Proxy for | Blind spot |
|------|------|-----------|------------|
| Cyclomatic (CCN) | lizard | testability (paths to cover) | over-counts flat `match`/dispatch |
| Cognitive | complexipy | readability (nesting/flow) | under-counts flat loops with many co-mutated locals |
| DepDegree (`dep`) | beniget | data-flow entanglement (def-use coupling) | scales with size; not a control-flow signal |

A function is a strong refactor target when CCN and cognitive are BOTH high. High CCN with
low cognitive is usually a flat dispatcher — readable, leave it. High cognitive with modest
CCN is the sneaky case: few paths, but deeply nested.

**The dep lens covers what the first two miss.** Cyclomatic and cognitive are both
control-flow metrics: they count branches and nesting. They under-rank a *flat* `while`
loop that threads ~8 co-mutated locals through a state machine (e.g. a VAD recorder) —
genuinely hard to read, but few branches and little nesting. DepDegree (Beyer & Fararooy,
ICPC 2010) counts the **def-use dependency edges** among a function's own local variables:
"more dependencies → more states to track → harder to understand." It is the only one of
the three that ranks `pyaudio_recording_worker` (dep 68) above `tts_worker` (dep 41), and
the only one that flags a pure data-flow hotspot like `pitch_extract` (cog 6, ccn 5 — both
within bands — but dep 42).

The `entangled` bucket / "Entangled state" prose line names functions whose dep is high but
whose cog/ccn do NOT already flag them — i.e. the findings the other two lenses would miss.
A high-CCN function that is *also* entangled is moved out of "Likely fine" (its flat-dispatch
verdict is refuted by hidden data coupling).

## Thresholds (advisory)

- `ccn_warn = 10` (radon "C" boundary), `ccn_high = 20`.
- `cog_warn = 15` (complexipy's default cap).
- `dep_warn = 40` (heuristic; dep scales with function size, so this is coarse — tune per repo).

Override with `--ccn-warn` / `--ccn-high` / `--cog-warn` / `--dep-warn`.

The report labels each function's cyclomatic band — `watch` (ccn > ccn_warn) or `high` (ccn > ccn_high) — surfaced as a `ccn_band` field in `--json` and a "Highest cyclomatic" line in the summary.

### How `dep` is computed

`beniget.DefUseChains` builds use-def chains over the (gast) AST; per function we sum
`len(def.users())` over the defs local to that function's scope (`chains.locals[node]`) —
the def-use edge count restricted to local variables. References to module globals,
imports and builtins are NOT a function's own locals, so they are excluded (their edges
attribute to the module scope), keeping `dep` a measure of *intra-function* entanglement.
Parameters count as defs (they are local data flow). beniget prints "unbound identifier"
notes to stderr for free names; the script mutes them.

## Why these tools

- **lizard** — actively maintained; own tokenizing parser, so it reads no `pyproject`
  (avoids the `configparser` crash radon hits on `%`-escaped wheel URLs) and parses
  3.11 `except*`. Emits CSV with line numbers, CCN, NLOC, token & parameter counts.
- **complexipy** — actively maintained cognitive-complexity tool; the live successor to
  the unmaintained `flake8-cognitive-complexity`.

## Deliberately excluded

- **Maintainability Index** — arbitrary/uncalibrated constants, LOC-dominated,
  averaging hides hotspots; radon's own docs call it "experimental".
- **radon** — maintenance-mode, and crashes reading this project's `pyproject.toml`.
- **wily / SaaS (SonarQube, Qlty)** — trend/dashboard machinery; overkill for a
  small/solo repo.

## Future options (not implemented)

- Churn overlay: rank by complexity × git change-frequency ("hotspot ROI").
- Diff mode: complexipy's built-in `--diff <ref>` for "complexity added by this branch".
