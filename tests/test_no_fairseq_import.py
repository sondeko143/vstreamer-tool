"""runtime に fairseq を二度と入れないための構造ゲート。

fairseq は requires-python 引き上げの唯一の障害（上流は 0.12.2 で凍結、
リポジトリは 2026-03-20 に archived）。import が復活したら即座に落とす。
"""

import ast
from pathlib import Path

VSPEECH_DIR = Path(__file__).resolve().parents[1] / "vspeech"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_vspeech_never_imports_fairseq():
    offenders = []
    for py_file in sorted(VSPEECH_DIR.rglob("*.py")):
        for module in _imported_modules(py_file):
            if module == "fairseq" or module.startswith("fairseq."):
                offenders.append(f"{py_file.relative_to(VSPEECH_DIR.parent)}: {module}")
    assert not offenders, "fairseq import leaked back into the runtime:\n" + "\n".join(
        offenders
    )
