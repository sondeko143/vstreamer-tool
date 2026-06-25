from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
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


@dataclass
class Thresholds:
    ccn_warn: int = 10
    ccn_high: int = 20
    cog_warn: int = 15


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


def join_metrics(
    lizard_metrics: list[FunctionMetric],
    cog_rows: list[tuple[str, str, int]],
) -> list[FunctionMetric]:
    index = build_cognitive_index(cog_rows)
    lizard_keys = {(m.file, m.function) for m in lizard_metrics}

    out = [
        replace(m, cognitive=index.get((m.file, m.function))) for m in lizard_metrics
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


def ccn_band(m: FunctionMetric, t: Thresholds) -> str:
    if m.ccn is None:
        return "n/a"
    if m.ccn > t.ccn_high:
        return "high"
    if m.ccn > t.ccn_warn:
        return "watch"
    return "ok"


def rank_metrics(metrics_list: list[FunctionMetric]) -> list[FunctionMetric]:
    def key(m: FunctionMetric) -> tuple[int, int]:
        return (
            m.cognitive if m.cognitive is not None else -1,
            m.ccn if m.ccn is not None else -1,
        )

    return sorted(metrics_list, key=key, reverse=True)


def _fmt(value: int | None) -> str:
    return "-" if value is None else str(value)


def render_summary(metrics_list: list[FunctionMetric], t: Thresholds, top: int) -> str:
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
    high_ccn = [m.function for m in shown if ccn_band(m, t) == "high"]
    if high_ccn:
        lines.append(f"Highest cyclomatic (ccn > {t.ccn_high}): " + ", ".join(high_ccn))
    return "\n".join(lines)


def metrics_to_json(metrics_list: list[FunctionMetric], t: Thresholds) -> str:
    ranked = rank_metrics(metrics_list)
    payload = [
        {**asdict(m), "bucket": bucket(m, t), "ccn_band": ccn_band(m, t)}
        for m in ranked
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
