from __future__ import annotations

import csv
import io
import json
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
