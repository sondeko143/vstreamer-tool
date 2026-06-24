# Gate Catalog

Each gate is `(name, phase, check command, kind, advisory)`. `kind=fixable` gates auto-apply a fix and re-check; `kind=report` gates never edit files.

| Gate | Phase | Command | Kind | Notes |
|---|---|---|---|---|
| ruff-format | static | `uv run ruff format --check .` | fixable | fix: `ruff format .` |
| ruff-lint | static | `uv run ruff check .` | fixable | fix: `ruff check --fix .` (safe only; never `--unsafe-fixes`) |
| ty | static | `uv run ty check` | report | type errors are never auto-fixed |
| pytest-cov | tests | `uv run pytest --cov=<pkg> --cov-report=term-missing` | report | honors project `addopts` (e2e excluded); on pass the summary reports the coverage TOTAL % (via `parse_coverage`) |
| uv-lock-check | deps | `uv lock --check` | report | fix is `uv lock` (proposed, not auto-run) |
| pip-audit | deps | `uvx pip-audit -r <exported-reqs>` | report | prepared via `uv export`; surfaces known CVEs |
| outdated | deps | `uv pip list --outdated` | report (advisory) | informational; never blocks |
| bandit | extra | `uv run --with bandit bandit -q -r <pkg>` | report | security lint; runs under project interpreter so version-specific syntax (e.g. `except*`) parses |
| vulture | extra | `uv run --with vulture vulture <pkg> --min-confidence 80` | report (advisory) | dead-code; high min-confidence to cut false positives; runs under project interpreter so version-specific syntax (e.g. `except*`) parses |

## Extras injection (`--extras`)

`ty` and `pytest-cov` resolve project imports against the *installed* env, so on projects with optional/GPU extras they report false `unresolved-import` / collection errors unless the extras are present. Pass `--extras all` or `--extras whisper,rvc`; `apply_extras` injects the corresponding `--all-extras` / `--extra NAME` flags **after `uv run`** in every gate's check/fix/prepare command. `uv lock`, `uv export`, and `uvx` invocations are left untouched. Default (no `--extras`) is unchanged behavior — safe on platforms where the extras' wheels don't resolve.

## Target detection

`derive_targets` discovers which packages to scan in this precedence order:

1. `[tool.uv.build-backend].module-name` (string or list) — explicit uv build-backend declaration.
2. `[project.scripts]` top package — top-level module extracted from each entry-point (`pkg.module:func`).
3. `[tool.poetry].packages[].include` — Poetry-style package declarations.
4. Normalized project name — `[project].name` or `[tool.poetry].name`, with `-` → `_`.
5. Fallback `.` — whole tree (when no name or packages can be found).

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
