"""A structural gate that keeps heavy ML frameworks out of the runtime for good.

- fairseq: the sole obstacle to raising requires-python (upstream is frozen at 0.12.2 and
  the repository was archived on 2026-03-20). Removed in spec 1.
- transformers: merely appearing in uv.lock brings three advisories into `uv audit`.
  Removed in spec 2 by moving the content encoder to ONNX.
- pydantic_settings: its provider barrel imports every backend (AWS / Azure / GCP Secret
  Manager, CLI, dotenv, YAML) unconditionally, which costs +13.7 MB RSS / +176 modules to
  import in isolation — though only 32 modules / ~1.6 MB of that was unique to it on the real
  startup path, the rest being shared with grpc and google-cloud. Removed in ADR-0066 by
  taking configuration from the `--config` file only.

Both are fine in the offline tools (scripts/convert_hubert.py,
scripts/export_hubert_onnx.py). What is forbidden is only `vspeech/`, i.e. the runtime.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

VSPEECH_DIR = Path(__file__).resolve().parents[1] / "vspeech"

FORBIDDEN = ("fairseq", "transformers", "pydantic_settings")


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
    """Whether it is `forbidden` itself, or a submodule of it.

    Always include the dot in the submodule test so that an unrelated module such as
    `fairseq_utils` is not swept in (`startswith("fairseq")` false-positives on it).
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
    """Pin the predicate itself.

    `test_vspeech_never_imports` runs against a runtime that currently imports none of
    them, so it stays green even if the predicate breaks. This is the only place that
    checks the predicate. It catches both a regression that drops the submodule test
    (missing `fairseq.data`) and one that drops the dot (false-positiving on
    `fairseq_utils`).
    """
    assert _is_forbidden(module, forbidden) is expected


def test_the_gate_would_catch_a_regression(tmp_path):
    """The AST walk and the predicate really are wired together (end-to-end)."""
    leaked = tmp_path / "leak.py"
    leaked.write_text("from transformers import HubertModel\n", encoding="utf-8")
    modules = list(_imported_modules(leaked))
    assert "transformers" in modules
    assert any(_is_forbidden(module, "transformers") for module in modules)


def test_consumer_path_is_torch_free():
    """The role=consumer (playback-only) modules pull in no torch at all (ADR-0055).

    A sys.modules check within this process is contaminated by test order (if an earlier
    test imported torch it would pass falsely), so the check runs in a pristine child
    process.
    """
    code = (
        "import sys\n"
        "import vspeech.stream_vc.consumer\n"
        "import vspeech.stream_vc.udp\n"
        "import vspeech.stream_vc.jitter\n"
        "import vspeech.stream_vc.wire\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_the_entry_point_never_loads_pydantic_settings():
    """Nothing on the startup path drags the env-config machinery back in (ADR-0066).

    The AST gate above only sees `vspeech/`; this catches a transitive import
    through a dependency. A sys.modules check inside the test process would be
    contaminated by test order, so it runs in a pristine child process.
    """
    code = (
        "import sys\n"
        "import vspeech.main\n"
        "assert 'pydantic_settings' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
