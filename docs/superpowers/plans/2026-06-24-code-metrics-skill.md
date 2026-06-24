# code-metrics Insight Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, advisory-only `code-metrics` skill that ranks a uv project's functions as refactor candidates using two complexity lenses (cyclomatic via lizard, cognitive via complexipy).

**Architecture:** A single stdlib-only orchestrator (`metrics.py`) reads `pyproject.toml` with `tomllib`, derives target packages, shells out to `uvx lizard --csv` and `uvx complexipy --output-format json`, then joins/ranks/buckets the results into a human summary or `--json`. It mirrors the existing `tools/python_health/` skill's structure and conventions but never gates and never edits code. The two analyzers run via `uvx` (own parsers, no project import, no `pyproject` read), so no project venv or `--extras` plumbing is needed.

**Tech Stack:** Python 3.11 (stdlib only: `argparse`, `csv`, `json`, `subprocess`, `tempfile`, `dataclasses`, `pathlib`, `tomllib`); external CLIs `lizard` and `complexipy` invoked via `uvx`; tests under `pytest`.

## Global Constraints

- **Stdlib only** in `metrics.py` — no third-party imports (analyzers are subprocesses). `tomllib` with a `tomli` fallback for Python < 3.11, exactly as `tools/python_health/scripts/health.py` does.
- **Advisory-only:** `main()` ALWAYS returns `0`. Never gate, never block, never edit source.
- **Analyzer invocation is fixed:**
  - lizard: `uvx lizard <pkgs> --csv` — parse stdout with stdlib `csv`.
  - complexipy: `uvx complexipy <pkgs> -q --output-format json --output <tmpfile>` with `PYTHONIOENCODING=utf-8` in the child env; **read the JSON file, never parse stdout** (its rich/emoji stdout crashes the Windows cp932 console).
- **Analyzer exit codes are ignored** except for "tool missing" (rc 127 / spawn failure → SKIP that lens). lizard and complexipy both exit non-zero merely when they find high-complexity functions — that is expected, not a failure.
- **Code style** matches `health.py`: `from __future__ import annotations`, **one import per line** (ruff `force-single-line`), full type annotations (ty-checked).
- **Import discipline:** each task adds to `metrics.py` ONLY the imports its new code uses (the project enforces ruff `I` + lint, so unused imports are `F401`/`I001` errors). Before committing each task, run `uv run ruff check --fix tools/code_metrics/` and `uv run ruff format tools/code_metrics/` — both must end clean.
- **Test import convention** (matches `tools/python_health/scripts/tests/test_health.py`): the test module imports the script via the package path `from tools.code_metrics.scripts import metrics` (NOT `sys.path.insert`). `tools/__init__.py` exists and pytest runs with `pythonpath = "."`, so this resolves and is ty-clean.
- **Join keys:** complexipy has no line numbers and names functions `Class::method`; lizard has line numbers and bare names. Join on `(normalized_forward_slash_path, simple_name)` where `simple_name` strips any `Class::` prefix. Line numbers come from lizard.
- **Thresholds (advisory bands):** `ccn_warn=10`, `ccn_high=20`, `cog_warn=15` (complexipy's default cap). Overridable via CLI.

---

## File Structure

Mirrors `tools/python_health/`:

- `tools/code_metrics/__init__.py` — empty package marker.
- `tools/code_metrics/scripts/__init__.py` — empty package marker.
- `tools/code_metrics/scripts/metrics.py` — the orchestrator (all logic).
- `tools/code_metrics/scripts/tests/__init__.py` — empty package marker.
- `tools/code_metrics/scripts/tests/test_metrics.py` — unit tests over captured fixtures (no live tool calls).
- `tools/code_metrics/SKILL.md` — the skill prose (procedure, how to read results, rules).
- `tools/code_metrics/references/metric-guide.md` — the two-lens model, thresholds, tool notes.

Install target (not committed): `~/.claude/skills/code-metrics/` (copy of the above), re-copied on change — same workflow as `python-health`.

---

## Task 1: Scaffolding, data model, path helpers, target detection

**Files:**
- Create: `tools/code_metrics/__init__.py` (empty)
- Create: `tools/code_metrics/scripts/__init__.py` (empty)
- Create: `tools/code_metrics/scripts/tests/__init__.py` (empty)
- Create: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `@dataclass FunctionMetric` with fields `file: str`, `function: str`, `line: int | None`, `ccn: int | None`, `nloc: int | None`, `params: int | None`, `cognitive: int | None`.
  - `@dataclass Targets` with `packages: list[str]`, `project_name: str`.
  - `normalize_path(p: str) -> str` (backslash → forward slash).
  - `simple_name(name: str) -> str` (strip `Class::` prefix → last segment).
  - `derive_targets(pyproject: dict) -> Targets` (same precedence as `health.py`).
  - `load_pyproject(root: Path) -> dict`.

- [ ] **Step 1: Create the empty package markers**

Create three empty files: `tools/code_metrics/__init__.py`, `tools/code_metrics/scripts/__init__.py`, `tools/code_metrics/scripts/tests/__init__.py` (each zero bytes).

- [ ] **Step 2: Write the failing test**

Create `tools/code_metrics/scripts/tests/test_metrics.py`:

```python
from tools.code_metrics.scripts import metrics


def test_normalize_path_converts_backslashes():
    assert metrics.normalize_path("vspeech\\lib\\command.py") == "vspeech/lib/command.py"


def test_simple_name_strips_class_prefix():
    assert metrics.simple_name("RVCModel::infer") == "infer"
    assert metrics.simple_name("process_command") == "process_command"


def test_derive_targets_prefers_build_backend_module_name():
    pyproject = {
        "project": {"name": "voicerecog"},
        "tool": {"uv": {"build-backend": {"module-name": ["vspeech"]}}},
    }
    targets = metrics.derive_targets(pyproject)
    assert targets.packages == ["vspeech"]


def test_derive_targets_falls_back_to_normalized_name():
    pyproject = {"project": {"name": "my-app"}}
    targets = metrics.derive_targets(pyproject)
    assert targets.packages == ["my_app"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'` (file not created yet).

- [ ] **Step 4: Write minimal implementation**

Create `tools/code_metrics/scripts/metrics.py` (import ONLY what Task 1 uses; later tasks add their own imports):

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionMetric:
    file: str
    function: str
    line: int | None
    ccn: int | None
    nloc: int | None
    params: int | None
    cognitive: int | None


@dataclass
class Targets:
    packages: list[str]
    project_name: str


def normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def simple_name(name: str) -> str:
    return name.rsplit("::", 1)[-1]


def derive_targets(pyproject: dict) -> Targets:
    project = pyproject.get("project", {})
    tool = pyproject.get("tool", {})
    poetry = tool.get("poetry", {})
    name = str(project.get("name") or poetry.get("name") or "")
    module_name = tool.get("uv", {}).get("build-backend", {}).get("module-name")

    packages: list[str] = []
    if isinstance(module_name, list):
        packages = [str(m) for m in module_name]
    elif isinstance(module_name, str):
        packages = [module_name]

    if not packages:
        for entry in project.get("scripts", {}).values():
            top = str(entry).split(":", 1)[0].split(".", 1)[0]
            if top and top not in packages:
                packages.append(top)

    if not packages:
        for pkg in poetry.get("packages", []):
            include = pkg.get("include") if isinstance(pkg, dict) else None
            if include and str(include) not in packages:
                packages.append(str(include))

    if not packages and name:
        packages = [name.replace("-", "_")]

    return Targets(packages=packages, project_name=name)


def load_pyproject(root: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # ty: ignore[unresolved-import]

    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add tools/code_metrics/__init__.py tools/code_metrics/scripts/__init__.py tools/code_metrics/scripts/tests/__init__.py tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): scaffolding, data model, target detection"
```

---

## Task 2: Parse lizard CSV

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: `FunctionMetric`, `normalize_path` (Task 1).
- Produces: `parse_lizard_csv(text: str) -> list[FunctionMetric]`. lizard `--csv` rows are headerless and positional: `nloc, ccn, token_count, param, length, "name@start-end@file", file, function_name, signature, start_line, end_line`. Returns `FunctionMetric` with `cognitive=None`; non-numeric/short rows are skipped.

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py` (top-of-file fixture, then the test):

```python
LIZARD_CSV = (
    '66,21,474,2,66,"process_command@30-95@vspeech\\lib\\command.py",'
    '"vspeech\\lib\\command.py","process_command","process_command( c )",30,95\n'
    '13,13,120,1,13,"operation_to_event@97-130@vspeech\\shared_context.py",'
    '"vspeech\\shared_context.py","operation_to_event","operation_to_event( op )",97,130\n'
    '8,3,60,1,8,"draw_text@41-70@vspeech\\worker\\subtitle.py",'
    '"vspeech\\worker\\subtitle.py","draw_text","draw_text( s )",41,70\n'
    '5,2,30,1,5,"helper@10-15@vspeech\\lib\\ami.py",'
    '"vspeech\\lib\\ami.py","helper","helper( x )",10,15\n'
)


def test_parse_lizard_csv_extracts_fields():
    rows = metrics.parse_lizard_csv(LIZARD_CSV)
    assert len(rows) == 4
    pc = next(r for r in rows if r.function == "process_command")
    assert pc.file == "vspeech/lib/command.py"
    assert pc.ccn == 21
    assert pc.nloc == 66
    assert pc.params == 2
    assert pc.line == 30
    assert pc.cognitive is None


def test_parse_lizard_csv_skips_non_numeric_rows():
    rows = metrics.parse_lizard_csv("not,a,real,row\n" + LIZARD_CSV)
    assert len(rows) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py::test_parse_lizard_csv_extracts_fields -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'parse_lizard_csv'`.

- [ ] **Step 3: Write minimal implementation**

First add the imports this code uses to the top of `metrics.py` (one per line; `uv run ruff check --fix` will sort them): `import csv` and `import io`. Then add:

```python
def parse_lizard_csv(text: str) -> list[FunctionMetric]:
    out: list[FunctionMetric] = []
    for cols in csv.reader(io.StringIO(text)):
        if len(cols) < 11:
            continue
        try:
            nloc = int(cols[0])
            ccn = int(cols[1])
            params = int(cols[3])
            line = int(cols[9])
        except ValueError:
            continue  # header / malformed row
        out.append(
            FunctionMetric(
                file=normalize_path(cols[6]),
                function=cols[7],
                line=line,
                ccn=ccn,
                nloc=nloc,
                params=params,
                cognitive=None,
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k lizard_csv -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): parse lizard CSV into FunctionMetric"
```

---

## Task 3: Parse complexipy JSON and build the cognitive index

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: `normalize_path`, `simple_name` (Task 1).
- Produces:
  - `parse_complexipy_json(text: str) -> list[tuple[str, str, int]]` — list of `(normalized_path, simple_name, cognitive)`.
  - `build_cognitive_index(rows: list[tuple[str, str, int]]) -> dict[tuple[str, str], int | None]` — keyed by `(path, simple_name)`; value is the cognitive score, or `None` when the same key carries conflicting values (ambiguous).

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py`:

```python
COMPLEXIPY_JSON = json.dumps(
    [
        {"complexity": 28, "file_name": "command.py",
         "function_name": "process_command", "path": "vspeech/lib/command.py",
         "refactor_plans": []},
        {"complexity": 3, "file_name": "shared_context.py",
         "function_name": "operation_to_event", "path": "vspeech/shared_context.py",
         "refactor_plans": []},
        {"complexity": 18, "file_name": "subtitle.py",
         "function_name": "draw_text", "path": "vspeech/worker/subtitle.py",
         "refactor_plans": []},
        {"complexity": 1, "file_name": "ami.py",
         "function_name": "helper", "path": "vspeech/lib/ami.py",
         "refactor_plans": []},
        {"complexity": 12, "file_name": "vc.py",
         "function_name": "RVCModel::orphan", "path": "vspeech/worker/vc.py",
         "refactor_plans": []},
    ]
)


def test_parse_complexipy_json_strips_class_and_normalizes():
    rows = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    assert ("vspeech/worker/vc.py", "orphan", 12) in rows
    assert ("vspeech/lib/command.py", "process_command", 28) in rows


def test_build_cognitive_index_marks_conflicts_ambiguous():
    rows = [
        ("a.py", "run", 5),
        ("a.py", "run", 9),
        ("b.py", "go", 7),
    ]
    index = metrics.build_cognitive_index(rows)
    assert index[("a.py", "run")] is None  # conflicting -> ambiguous
    assert index[("b.py", "go")] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k complexipy -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'parse_complexipy_json'`.

- [ ] **Step 3: Write minimal implementation**

First add `import json` to the top of `metrics.py` (if not already present). Then add:

```python
def parse_complexipy_json(text: str) -> list[tuple[str, str, int]]:
    data = json.loads(text)
    out: list[tuple[str, str, int]] = []
    for item in data:
        path = normalize_path(str(item["path"]))
        name = simple_name(str(item["function_name"]))
        out.append((path, name, int(item["complexity"])))
    return out


def build_cognitive_index(
    rows: list[tuple[str, str, int]],
) -> dict[tuple[str, str], int | None]:
    index: dict[tuple[str, str], int | None] = {}
    for path, name, cog in rows:
        key = (path, name)
        if key in index and index[key] != cog:
            index[key] = None  # conflicting same-name entries -> ambiguous
        elif key not in index:
            index[key] = cog
    return index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k complexipy -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): parse complexipy JSON + cognitive index"
```

---

## Task 4: Join the two lenses

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: `FunctionMetric`, `build_cognitive_index` (Tasks 1, 3), `replace` (dataclasses).
- Produces: `join_metrics(lizard_metrics: list[FunctionMetric], cog_rows: list[tuple[str, str, int]]) -> list[FunctionMetric]`. Each lizard metric gets `cognitive` filled from the index (or `None` if unmatched/ambiguous). complexipy-only functions (no lizard match) are appended with `ccn=None`, `nloc=None`, `params=None`, `line=None`, deduplicated by key.

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py`:

```python
def test_join_fills_cognitive_and_appends_orphans():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)

    pc = next(m for m in joined if m.function == "process_command")
    assert pc.ccn == 21 and pc.cognitive == 28

    orphan = next(m for m in joined if m.function == "orphan")
    assert orphan.ccn is None
    assert orphan.line is None
    assert orphan.cognitive == 12
    assert orphan.file == "vspeech/worker/vc.py"

    # exactly one orphan appended (the 4 lizard rows + 1 complexipy-only)
    assert len(joined) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py::test_join_fills_cognitive_and_appends_orphans -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'join_metrics'`.

- [ ] **Step 3: Write minimal implementation**

First add `from dataclasses import replace` to the top of `metrics.py`. Then add:

```python
def join_metrics(
    lizard_metrics: list[FunctionMetric],
    cog_rows: list[tuple[str, str, int]],
) -> list[FunctionMetric]:
    index = build_cognitive_index(cog_rows)
    lizard_keys = {(m.file, m.function) for m in lizard_metrics}

    out = [
        replace(m, cognitive=index.get((m.file, m.function)))
        for m in lizard_metrics
    ]

    appended: set[tuple[str, str]] = set()
    for path, name, _cog in cog_rows:
        key = (path, name)
        if key in lizard_keys or key in appended:
            continue
        appended.add(key)
        out.append(
            FunctionMetric(
                file=path,
                function=name,
                line=None,
                ccn=None,
                nloc=None,
                params=None,
                cognitive=index.get(key),
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py::test_join_fills_cognitive_and_appends_orphans -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): join cyclomatic + cognitive lenses"
```

---

## Task 5: Thresholds, bucketing, and ranking

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: `FunctionMetric` (Task 1).
- Produces:
  - `@dataclass Thresholds` with `ccn_warn: int = 10`, `ccn_high: int = 20`, `cog_warn: int = 15`.
  - `bucket(m: FunctionMetric, t: Thresholds) -> str` → one of `"both-high"`, `"high-ccn"`, `"high-cognitive"`, `"ok"`. A lens flags when its value is not `None` and strictly greater than its warn threshold.
  - `rank_metrics(metrics: list[FunctionMetric]) -> list[FunctionMetric]` → sorted by `cognitive` desc then `ccn` desc, treating `None` as `-1` (sorts last).

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py`:

```python
def test_bucket_classifies_each_case():
    t = metrics.Thresholds()
    both = metrics.FunctionMetric("f.py", "a", 1, 21, 60, 2, 28)
    ccn_only = metrics.FunctionMetric("f.py", "b", 1, 13, 13, 1, 3)
    cog_only = metrics.FunctionMetric("f.py", "c", 1, 3, 8, 1, 18)
    ok = metrics.FunctionMetric("f.py", "d", 1, 2, 5, 1, 1)
    assert metrics.bucket(both, t) == "both-high"
    assert metrics.bucket(ccn_only, t) == "high-ccn"
    assert metrics.bucket(cog_only, t) == "high-cognitive"
    assert metrics.bucket(ok, t) == "ok"


def test_rank_orders_by_cognitive_then_ccn_none_last():
    a = metrics.FunctionMetric("f.py", "a", 1, 5, 5, 1, 28)
    b = metrics.FunctionMetric("f.py", "b", 1, 9, 9, 1, 18)
    c = metrics.FunctionMetric("f.py", "c", 1, 30, 9, 1, None)
    ranked = metrics.rank_metrics([c, b, a])
    assert [m.function for m in ranked] == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k "bucket or rank" -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'Thresholds'`.

- [ ] **Step 3: Write minimal implementation**

Add to `metrics.py`:

```python
@dataclass
class Thresholds:
    ccn_warn: int = 10
    ccn_high: int = 20
    cog_warn: int = 15


def bucket(m: FunctionMetric, t: Thresholds) -> str:
    ccn_flag = m.ccn is not None and m.ccn > t.ccn_warn
    cog_flag = m.cognitive is not None and m.cognitive > t.cog_warn
    if ccn_flag and cog_flag:
        return "both-high"
    if ccn_flag:
        return "high-ccn"
    if cog_flag:
        return "high-cognitive"
    return "ok"


def rank_metrics(metrics_list: list[FunctionMetric]) -> list[FunctionMetric]:
    def key(m: FunctionMetric) -> tuple[int, int]:
        return (
            m.cognitive if m.cognitive is not None else -1,
            m.ccn if m.ccn is not None else -1,
        )

    return sorted(metrics_list, key=key, reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k "bucket or rank" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): thresholds, bucketing, ranking"
```

---

## Task 6: Render the summary and JSON

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: `FunctionMetric`, `Thresholds`, `bucket`, `rank_metrics` (Tasks 1, 5), `asdict` (dataclasses).
- Produces:
  - `render_summary(metrics_list: list[FunctionMetric], t: Thresholds, top: int) -> str` — a ranked table of flagged functions (or top-N when nothing is flagged), followed by prose grouping by bucket. `top=0` means "all".
  - `metrics_to_json(metrics_list: list[FunctionMetric], t: Thresholds) -> str` — JSON array of ranked records, each augmented with a `"bucket"` field.

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py`:

```python
def test_render_summary_flags_and_explains():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    out = metrics.render_summary(joined, metrics.Thresholds(), top=15)
    assert "process_command" in out
    assert "both-high" in out
    assert "draw_text" in out  # high-cognitive
    # de-prioritized flat dispatcher is named in the "likely fine" prose
    assert "operation_to_event" in out


def test_metrics_to_json_is_valid_and_has_bucket():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    payload = json.loads(metrics.metrics_to_json(joined, metrics.Thresholds()))
    assert payload[0]["function"] == "process_command"
    assert payload[0]["bucket"] == "both-high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k "render or to_json" -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'render_summary'`.

- [ ] **Step 3: Write minimal implementation**

First add `from dataclasses import asdict` to the top of `metrics.py`. Then add:

```python
def _fmt(value: int | None) -> str:
    return "-" if value is None else str(value)


def render_summary(
    metrics_list: list[FunctionMetric], t: Thresholds, top: int
) -> str:
    ranked = rank_metrics(metrics_list)
    flagged = [m for m in ranked if bucket(m, t) != "ok"]
    shown = flagged if flagged else ranked
    if top and len(shown) > top:
        shown = shown[:top]

    lines = [
        "",
        "=== code-metrics: refactor candidates ===",
        f"{'cog':>4} {'ccn':>4} {'nloc':>5} {'par':>4}  function (file:line)  [bucket]",
    ]
    for m in shown:
        loc = f"{m.file}:{m.line}" if m.line is not None else m.file
        lines.append(
            f"{_fmt(m.cognitive):>4} {_fmt(m.ccn):>4} {_fmt(m.nloc):>5} "
            f"{_fmt(m.params):>4}  {m.function} ({loc})  [{bucket(m, t)}]"
        )

    both = [m.function for m in shown if bucket(m, t) == "both-high"]
    cog_only = [m.function for m in shown if bucket(m, t) == "high-cognitive"]
    ccn_only = [m.function for m in shown if bucket(m, t) == "high-ccn"]
    lines.append("")
    if both:
        lines.append("Top targets (tangled AND branchy): " + ", ".join(both))
    if cog_only:
        lines.append("Sneaky (deep nesting, few paths): " + ", ".join(cog_only))
    if ccn_only:
        lines.append(
            "Likely fine (wide but flat dispatch; de-prioritize): "
            + ", ".join(ccn_only)
        )
    if not (both or cog_only or ccn_only):
        lines.append("No functions exceed thresholds — within complexity bands.")
    return "\n".join(lines)


def metrics_to_json(metrics_list: list[FunctionMetric], t: Thresholds) -> str:
    ranked = rank_metrics(metrics_list)
    payload = [{**asdict(m), "bucket": bucket(m, t)} for m in ranked]
    return json.dumps(payload, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k "render or to_json" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): render summary + JSON output"
```

---

## Task 7: Collection layer and CLI (`main`)

**Files:**
- Modify: `tools/code_metrics/scripts/metrics.py`
- Test: `tools/code_metrics/scripts/tests/test_metrics.py`

**Interfaces:**
- Consumes: all prior functions.
- Produces:
  - `CommandRunner = Callable[[list[str], dict[str, str] | None], tuple[int, str, str]]`.
  - `is_missing(rc: int, err: str) -> bool` — True for command-not-found / spawn failure (rc 127 or matching stderr text).
  - `subprocess_runner(cmd: list[str], env_extra: dict[str, str] | None = None) -> tuple[int, str, str]`.
  - `collect_lizard(run: CommandRunner, pkgs: list[str]) -> str | None` — runs `uvx lizard <pkgs> --csv`; returns stdout, or `None` if the tool is missing. Exit code otherwise ignored.
  - `collect_complexipy(run: CommandRunner, pkgs: list[str], out_path: str) -> str | None` — runs `uvx complexipy <pkgs> -q --output-format json --output <out_path>` with `PYTHONIOENCODING=utf-8`; returns the file contents (read regardless of exit code), or `None` if the tool is missing or the file is absent.
  - `main(argv: list[str] | None = None) -> int` — wires it all; always returns `0`.

- [ ] **Step 1: Write the failing test**

Add to `test_metrics.py`:

```python
def test_collect_complexipy_reads_file_despite_nonzero_exit(tmp_path):
    out_file = tmp_path / "cx.json"

    def fake_run(cmd, env_extra=None):
        # complexipy exits 1 when it finds high-complexity functions
        out_file.write_text(COMPLEXIPY_JSON, encoding="utf-8")
        assert env_extra == {"PYTHONIOENCODING": "utf-8"}
        return 1, "", ""

    text = metrics.collect_complexipy(fake_run, ["vspeech"], str(out_file))
    assert text is not None
    assert "process_command" in text


def test_collect_lizard_returns_none_when_missing():
    def fake_run(cmd, env_extra=None):
        return 127, "", "command not found: uvx"

    assert metrics.collect_lizard(fake_run, ["vspeech"]) is None


def test_main_runs_advisory_and_returns_zero(tmp_path, capsys, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n', encoding="utf-8"
    )

    def fake_runner(cmd, env_extra=None):
        if "lizard" in cmd:
            return 0, LIZARD_CSV, ""
        if "complexipy" in cmd:
            idx = cmd.index("--output") + 1
            Path(cmd[idx]).write_text(COMPLEXIPY_JSON, encoding="utf-8")
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(metrics, "subprocess_runner", fake_runner)
    rc = metrics.main(["--root", str(tmp_path)])
    assert rc == 0
    assert "process_command" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -k "collect or main" -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'collect_complexipy'`.

- [ ] **Step 3: Write minimal implementation**

First add the imports this code uses to the top of `metrics.py`: `import argparse`, `import os`, `import subprocess`, `import tempfile`, and `from collections.abc import Callable`. Then add:

```python
CommandRunner = Callable[[list[str], dict[str, str] | None], tuple[int, str, str]]


def is_missing(rc: int, err: str) -> bool:
    e = (err or "").lower()
    return (
        rc == 127
        or "command not found" in e
        or "failed to spawn" in e
        or "no such file" in e
    )


def subprocess_runner(
    cmd: list[str], env_extra: dict[str, str] | None = None
) -> tuple[int, str, str]:
    env = {**os.environ, **env_extra} if env_extra else None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    return proc.returncode, proc.stdout, proc.stderr


def collect_lizard(run: CommandRunner, pkgs: list[str]) -> str | None:
    rc, out, err = run(["uvx", "lizard", *pkgs, "--csv"], None)
    if is_missing(rc, err):
        return None
    return out


def collect_complexipy(
    run: CommandRunner, pkgs: list[str], out_path: str
) -> str | None:
    rc, _out, err = run(
        [
            "uvx",
            "complexipy",
            *pkgs,
            "-q",
            "--output-format",
            "json",
            "--output",
            out_path,
        ],
        {"PYTHONIOENCODING": "utf-8"},
    )
    if is_missing(rc, err):
        return None
    try:
        return Path(out_path).read_text(encoding="utf-8")
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code-metrics")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--ccn-warn", type=int, default=10)
    parser.add_argument("--ccn-high", type=int, default=20)
    parser.add_argument("--cog-warn", type=int, default=15)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    os.chdir(root)
    targets = derive_targets(load_pyproject(root))
    pkgs = targets.packages or ["."]
    thresholds = Thresholds(args.ccn_warn, args.ccn_high, args.cog_warn)

    lizard_text = collect_lizard(subprocess_runner, pkgs)
    with tempfile.TemporaryDirectory() as td:
        cx_path = os.path.join(td, "complexipy.json")
        cx_text = collect_complexipy(subprocess_runner, pkgs, cx_path)

    lizard_metrics = parse_lizard_csv(lizard_text) if lizard_text else []
    cog_rows = parse_complexipy_json(cx_text) if cx_text else []
    joined = join_metrics(lizard_metrics, cog_rows)

    if args.json:
        print(metrics_to_json(joined, thresholds))
    else:
        print(render_summary(joined, thresholds, args.top))
        if lizard_text is None:
            print("[SKIP] lizard unavailable — cyclomatic lens missing")
        if cx_text is None:
            print("[SKIP] complexipy unavailable — cognitive lens missing")
    return 0  # advisory: never gate


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `uv run pytest tools/code_metrics/scripts/tests/test_metrics.py -v`
Expected: PASS (all tests, including `collect`/`main`).

- [ ] **Step 5: Lint, format, and type-check the new module**

Run: `uv run ruff format tools/code_metrics/ && uv run ruff check tools/code_metrics/ && uv run ty check tools/code_metrics/scripts/metrics.py`
Expected: format clean, no lint errors, no type errors. Fix any reported issue before committing.

- [ ] **Step 6: Commit**

```bash
git add tools/code_metrics/scripts/metrics.py tools/code_metrics/scripts/tests/test_metrics.py
git commit -m "feat(code-metrics): collection layer + advisory CLI"
```

---

## Task 8: Skill prose, reference guide, install, and real smoke test

**Files:**
- Create: `tools/code_metrics/SKILL.md`
- Create: `tools/code_metrics/references/metric-guide.md`

**Interfaces:**
- Consumes: the finished `metrics.py` CLI (Task 7).
- Produces: a runnable skill at `~/.claude/skills/code-metrics/` and verified output on this repo.

- [ ] **Step 1: Write `SKILL.md`**

Create `tools/code_metrics/SKILL.md`:

```markdown
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
2. Run the orchestrator from the project root:
   `uv run --with tomli python <skill-dir>/scripts/metrics.py --root .`
   - Machine-readable: add `--json`.
   - Show more/all rows: `--top 30` (or `--top 0` for all flagged).
   - Tune bands: `--ccn-warn`, `--ccn-high`, `--cog-warn`.
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
```

- [ ] **Step 2: Write `references/metric-guide.md`**

Create `tools/code_metrics/references/metric-guide.md`:

```markdown
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
```

- [ ] **Step 3: Install the skill to the user skills dir**

Run (PowerShell):

```powershell
$dst = "$env:USERPROFILE\.claude\skills\code-metrics"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item -Recurse -Force "tools\code_metrics\*" $dst
Get-ChildItem -Recurse $dst | Select-Object FullName
```

Expected: `SKILL.md`, `scripts/metrics.py`, `references/metric-guide.md` present under the destination. (Do not copy `__pycache__`; if present, it is harmless.)

- [ ] **Step 4: Real smoke test on this repo**

Run from the project root:

```bash
uv run --with tomli python ~/.claude/skills/code-metrics/scripts/metrics.py --root .
```

Expected: a ranked table where `process_command (vspeech/lib/command.py:30)` appears with `ccn 21` and a cognitive score, bucketed `both-high` or `high-cognitive`; no traceback; exits 0. Also run with `--json` and confirm valid JSON.

- [ ] **Step 5: Commit**

```bash
git add tools/code_metrics/SKILL.md tools/code_metrics/references/metric-guide.md
git commit -m "feat(code-metrics): skill prose, reference guide, install"
```

---

## Self-Review notes (for the implementer)

- The `metrics` module is imported via `sys.path.insert` in the test (no package install), matching how `test_health.py` imports `health`. Confirm `tools/python_health/scripts/tests/test_health.py` for the exact pattern if anything is unclear.
- Keep imports one-per-line (ruff `force-single-line`) — the `from dataclasses import X` lines are intentionally split.
- `main()` is the only place `subprocess_runner` is referenced by name (so the `test_main_*` monkeypatch works); do not inline it.
- After Task 7, the whole suite plus `ruff`/`ty` must be green before Task 8.
