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


def _is_forbidden(module: str, forbidden: str) -> bool:
    """`forbidden` 本体か、そのサブモジュールか。

    `fairseq_utils` のような別モジュールを巻き込まないよう、サブモジュール判定には
    必ずドットを付ける（`startswith("fairseq")` では誤検出する）。
    """
    return module == forbidden or module.startswith(f"{forbidden}.")


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_vspeech_never_imports(forbidden: str):
    offenders = []
    for py_file in sorted(VSPEECH_DIR.rglob("*.py")):
        for module in _imported_modules(py_file):
            if _is_forbidden(module, forbidden):
                offenders.append(f"{py_file.relative_to(VSPEECH_DIR.parent)}: {module}")
    assert not offenders, (
        f"{forbidden} import leaked back into the runtime:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    ("module", "forbidden", "expected"),
    [
        ("fairseq", "fairseq", True),
        ("fairseq.data", "fairseq", True),
        ("transformers.models.hubert", "transformers", True),
        ("fairseq_utils", "fairseq", False),
        ("torch", "fairseq", False),
        ("torch", "transformers", False),
    ],
)
def test_is_forbidden_predicate(module: str, forbidden: str, expected: bool):
    """述語そのものを固定する。

    `test_vspeech_never_imports` はいま何も import していない runtime に対して走るので、
    述語が壊れても緑のままになる。ここだけが述語を検査している。サブモジュール判定を
    落とす退行 (`fairseq.data` を見逃す) と、ドットを落とす退行 (`fairseq_utils` を
    誤検出する) の両方を捕まえる。
    """
    assert _is_forbidden(module, forbidden) is expected


def test_the_gate_would_catch_a_regression(tmp_path):
    """AST 走査と述語が実際に繋がっていること（end-to-end）。"""
    leaked = tmp_path / "leak.py"
    leaked.write_text("from transformers import HubertModel\n", encoding="utf-8")
    modules = list(_imported_modules(leaked))
    assert "transformers" in modules
    assert any(_is_forbidden(module, "transformers") for module in modules)
