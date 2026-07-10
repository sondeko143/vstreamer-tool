"""runtime に重い ML フレームワークを二度と入れないための構造ゲート。

- fairseq: requires-python 引き上げの唯一の障害（上流は 0.12.2 で凍結、リポジトリは
  2026-03-20 に archived）。spec ① で撤去。
- transformers: uv.lock に載るだけで `uv audit` に 3 件の advisory を持ち込む。
  spec ② で content encoder を ONNX 化して撤去。

どちらも offline ツール (scripts/convert_hubert.py, scripts/export_hubert_onnx.py) では
使ってよい。禁じるのは `vspeech/` 配下、すなわち runtime だけ。
"""

import ast
from pathlib import Path

import pytest

VSPEECH_DIR = Path(__file__).resolve().parents[1] / "vspeech"

FORBIDDEN = ("fairseq", "transformers")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_vspeech_never_imports(forbidden: str):
    offenders = []
    for py_file in sorted(VSPEECH_DIR.rglob("*.py")):
        for module in _imported_modules(py_file):
            if module == forbidden or module.startswith(f"{forbidden}."):
                offenders.append(f"{py_file.relative_to(VSPEECH_DIR.parent)}: {module}")
    assert not offenders, (
        f"{forbidden} import leaked back into the runtime:\n" + "\n".join(offenders)
    )


def test_the_gate_would_catch_a_regression(tmp_path):
    """ゲートが空虚でないこと: 実際に禁止 import を含むファイルを検出できる。"""
    leaked = tmp_path / "leak.py"
    leaked.write_text("from transformers import HubertModel\n", encoding="utf-8")
    assert "transformers" in list(_imported_modules(leaked))
