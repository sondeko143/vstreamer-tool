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
