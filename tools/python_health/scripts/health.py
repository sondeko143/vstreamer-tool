from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Targets:
    packages: list[str]
    project_name: str


def derive_targets(pyproject: dict) -> Targets:
    project = pyproject.get("project", {})
    name = str(project.get("name", ""))
    tool = pyproject.get("tool", {})
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

    if not packages and name:
        packages = [name.replace("-", "_")]

    return Targets(packages=packages, project_name=name)
