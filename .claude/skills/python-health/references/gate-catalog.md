# Gate Catalog

The gates are `poethepoet` tasks in the target project's `pyproject.toml` (see the **Task block** in `SKILL.md`). `poe check` runs them all report-only (collect-all); `poe fix` applies only the mechanical fixes. The skill, not a script, owns the triage.

| poe task | Phase | Command | Triage |
|---|---|---|---|
| `fmt-check` | static | `ruff format --check .` | mechanical → `poe fix` |
| `lint` | static | `ruff check .` | mechanical → `poe fix` (safe fixes only; never `--unsafe-fixes`) |
| `type` | static | `ty check` | report; never auto-fix |
| `test` | tests | `pytest --cov=<pkg> --cov-report=term-missing` | report; honors project `addopts` (e.g. e2e excluded). Read the coverage TOTAL % |
| `lock-check` | deps | `uv lock --check` | report; fix is usually `uv lock` (proposed, not auto-run) |
| `audit` | deps | `uv audit` | report; surface CVE + fixed version |
| `security` | extra | `uv run --with bandit bandit -q -r <pkg>` | report; runs under the project interpreter so version-specific syntax (e.g. `except*`) parses |
| `deadcode` | extra | `uv run --with vulture vulture <pkg> --min-confidence 80` | advisory; high min-confidence to cut false positives |

`fix` = `[fmt, lint-fix]`. `check` = the 8 report tasks with `ignore_fail = "return_non_zero"` (runs all, exits non-zero if any failed).

## Auto-fix policy

- **Auto-applied** (deterministic, reversible, no logic change): `poe fix` → `ruff format` + `ruff check --fix` (safe fixes).
- **Never auto-applied** (need human judgment): ty errors, test failures, dependency upgrades, `uv audit` / bandit findings, vulture deletions.

## Extras (GPU/ML projects)

`ty` and `test` resolve imports against the *installed* env, so projects with optional/GPU extras report false `unresolved-import` / collection errors unless the extras are present. Run `uv run --all-extras poe check` (or `--extra whisper --extra rvc`). The extras must ride on the `uv run` that launches `poe`; a bare `uv run` re-syncs to the default env and strips them, and a standalone `uv sync --all-extras` does not stick. Default (no extras) is safe on platforms where the extras' wheels don't resolve — report unresolvable imports as expected-absent/advisory.

## Target detection (done when setting up the task block)

Pick `<pkg>` for `test` / `security` / `deadcode` in this precedence order:

1. `[tool.uv.build-backend].module-name` (string or list).
2. `[project.scripts]` top package — top-level module of each entry-point (`pkg.module:func`).
3. `[tool.poetry].packages[].include` — Poetry-style declarations.
4. Normalized project name — `[project].name` or `[tool.poetry].name`, `-` → `_`.
5. Fallback `.` — whole tree.

## False-positive tuning

- bandit: scan only the package dir(s); test code is excluded by default. Suppress an accepted finding at the line with `# nosec <ID> - <reason>` — never disable the check globally.
- vulture: `--min-confidence 80` and treat as advisory; maintain a per-project allowlist if needed.

## uv version note

`uv audit` is native in uv 0.11+. On older uv, replace the `audit` task with `uv export --no-hashes --no-emit-project | uvx pip-audit -r -`.
