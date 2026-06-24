# Metric guide

## The two-lens model

| Lens | Tool | Proxy for | Blind spot |
|------|------|-----------|------------|
| Cyclomatic (CCN) | lizard | testability (paths to cover) | over-counts flat `match`/dispatch |
| Cognitive | complexipy | readability (nesting/flow) | younger metric, own implementation |

A function is a strong refactor target when BOTH are high. High CCN with low cognitive
is usually a flat dispatcher — readable, leave it. High cognitive with modest CCN is the
sneaky case: few paths, but deeply nested.

## Thresholds (advisory)

- `ccn_warn = 10` (radon "C" boundary), `ccn_high = 20`.
- `cog_warn = 15` (complexipy's default cap).

Override with `--ccn-warn` / `--ccn-high` / `--cog-warn`.

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
