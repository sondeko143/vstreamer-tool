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
- torchaudio: it pulls in torch, so keeping it is keeping torch's +476.7 MB RSS / +3.17 s
  of startup. Its only use here was a resampler duplicating the in-house polyphase FIR
  the device boundaries already run. Removed in ADR-0082.
- torch: the RVC conversion path (HuBERT features, f0, the RVC synthesizer's `infer`)
  now binds `OrtValue`s directly and runs on numpy + onnxruntime-native, with no
  framework tensor anywhere on the path. Removed in ADR-0081.

They are all fine in the offline tools (scripts/convert_hubert.py,
scripts/export_hubert_onnx.py, scripts/export_fcpe_onnx.py). What is forbidden there is
only `vspeech/`, i.e. the runtime.

For torch the import gate is not sufficient, which is why this file also gates the
**dependency table** (the second half, below -- ADR-0084). `ctranslate2` is a core
dependency and does `try: import torch`, so a torch that is merely *installed* is loaded
into every pipeline whether or not any line of this repo imports it -- +476.7 MB RSS and
+3.17 s of startup, i.e. the entire benefit of ADR-0080. One `uv add`, or one dependency
that grows a transitive edge to torch, would restore that with every import gate still
green.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VSPEECH_DIR = REPO_ROOT / "vspeech"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

FORBIDDEN = ("fairseq", "transformers", "pydantic_settings", "torch", "torchaudio")

# Distribution names, not import names -- what is gated below is what gets installed.
# faiss-cpu is here although nothing ever imported it: it was in the rvc extra, so it was
# installed, and ADR-0080 removed it for that reason alone.
FORBIDDEN_DISTRIBUTIONS = ("torch", "torchaudio", "faiss-cpu")


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
        ("torchaudio", "torchaudio", True),
        ("torchaudio.transforms", "torchaudio", True),
        # `torch` and `torchaudio` are both forbidden now, but as two separate
        # entries in FORBIDDEN, and neither is a submodule of the other -- pin the
        # dot boundary between the two similarly-named packages in both directions.
        ("torch", "torchaudio", False),
        ("torchaudio", "torch", False),
        ("torch", "torch", True),
        ("torch.cuda", "torch", True),
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


def test_device_layer_is_torch_free():
    """Resolving a device and opening an ONNX session pull in no torch (ADR-0078).

    This is the invariant the whisper pipeline's memory and startup time rest on: the
    transcription worker infers with ctranslate2 and needs torch for nothing, but it
    resolves a GPU through this layer. While that layer spoke `torch.device`, whisper
    paid 477MB of RSS and 3.2s of startup for one integer.

    The check targets the device layer rather than `vspeech.worker.transcription`
    because that worker defers both `faster_whisper` and `get_device` into the function
    body -- importing the module proves nothing.
    """
    code = (
        "import sys\n"
        "import vspeech.lib.cuda_driver\n"
        "import vspeech.lib.cuda_util\n"
        "import vspeech.lib.onnx_session\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _canonical(name: str) -> str:
    """A distribution name in the one spelling everything else is compared against.

    PEP 503 normalisation: runs of `-`, `_` and `.` collapse to a single `-`, and case is
    folded. Without it `Torch`, `faiss_cpu` and `faiss.cpu` would each walk past the gate.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """The distribution name out of a PEP 508 requirement string.

    Everything after the name is a version specifier, an extras list, a `; marker` or a
    `@ url` -- all of which have to be stripped, since `torch @ https://.../torch-...whl`
    is a perfectly ordinary way to add torch back.
    """
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return _canonical(match.group(1)) if match else ""


def _declared_distributions(pyproject: dict[str, Any]) -> set[str]:
    """Every distribution this project asks for **by name**, across all tables.

    Extras and dependency groups included: an extra is still installed by
    `uv sync --all-extras`, which is what this project's own docs tell people to run.
    """
    project = pyproject.get("project", {})
    requirements: list[Any] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    for group in pyproject.get("dependency-groups", {}).values():
        # A group entry can be `{include-group = "..."}` rather than a requirement string.
        requirements += [entry for entry in group if isinstance(entry, str)]
    return {name for entry in requirements if (name := _requirement_name(entry))}


def _locked_distributions(lock: dict[str, Any]) -> set[str]:
    """Every distribution in the resolved set, transitive edges included.

    This is the half that cannot be fooled: uv.lock lists what would actually be
    installed, so a dependency that grows an edge to torch shows up here even though no
    table in pyproject.toml mentions it.
    """
    return {_canonical(package["name"]) for package in lock.get("package", [])}


@pytest.mark.parametrize("distribution", FORBIDDEN_DISTRIBUTIONS)
def test_no_forbidden_distribution_is_declared(distribution: str):
    """Nothing asks for it directly. Catches the `uv add torch` before the lock is even
    regenerated, and names the culprit table."""
    with open(PYPROJECT_TOML, "rb") as f:
        pyproject = tomllib.load(f)
    assert _canonical(distribution) not in _declared_distributions(pyproject), (
        f"{distribution} is back in pyproject.toml. ctranslate2 loads torch whenever it "
        "is installed, so declaring it costs 477MB and 3.2s per pipeline even if no code "
        "imports it (ADR-0080)."
    )


@pytest.mark.parametrize("distribution", FORBIDDEN_DISTRIBUTIONS)
def test_no_forbidden_distribution_is_in_the_resolved_set(distribution: str):
    """Nor does anything pull it in transitively.

    The lock is the resolved set for this project's only environment
    (`sys_platform == 'win32'`), so a package listed here is a package `uv sync` installs.
    """
    with open(UV_LOCK, "rb") as f:
        lock = tomllib.load(f)
    assert _canonical(distribution) not in _locked_distributions(lock), (
        f"{distribution} is back in uv.lock. If no table declares it, some dependency "
        "grew an edge to it -- find that edge rather than relaxing this test (ADR-0080)."
    )


def test_the_lock_gate_catches_a_purely_transitive_edge():
    """The end-to-end proof that the lock half adds something the pyproject half cannot.

    The synthetic lock declares nothing, exactly as a transitive edge would: torch is
    present only because something else resolved to it.
    """
    lock = tomllib.loads(
        '[[package]]\nname = "some-dep"\nversion = "1.0"\n'
        '[[package]]\nname = "Torch"\nversion = "2.13.0"\n'
    )
    assert "torch" in _locked_distributions(lock)


def test_the_declaration_gate_catches_every_table_a_dependency_can_hide_in():
    """A requirement is a requirement wherever it is written, and in whatever spelling."""
    pyproject = tomllib.loads(
        "[project]\n"
        'dependencies = ["Torch >= 2.13"]\n'
        "[project.optional-dependencies]\n"
        'rvc = ["torchaudio @ https://example.invalid/torchaudio-2.11.0.whl"]\n'
        "[dependency-groups]\n"
        "dev = [\"faiss_cpu ; sys_platform == 'win32'\"]\n"
    )
    assert _declared_distributions(pyproject) >= {"torch", "torchaudio", "faiss-cpu"}


def test_the_declaration_gate_does_not_false_positive_on_a_similar_name():
    """`torchvision` is not `torch`; the comparison is on whole canonical names."""
    pyproject = tomllib.loads('[project]\ndependencies = ["torchvision>=0.1"]\n')
    declared = _declared_distributions(pyproject)
    assert declared == {"torchvision"}
    assert "torch" not in declared


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
