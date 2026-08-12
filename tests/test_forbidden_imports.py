"""Two structural gates on what the runtime is allowed to re-acquire.

The first half bans module names from `vspeech/`. The offline tools in `scripts/` may
import whatever they need -- what is gated is the runtime. The second half (ADR-0084) bans
three distributions from the dependency table and from uv.lock's resolved set, an entirely
different failure: a merely *installed* torch is picked up by ctranslate2 whether or not a
line of this repo imports it.

**No reason is written out in this file.** Each entry in `FORBIDDEN` carries the paths of the
ADRs that hold its reason, every failure message prints them, and a test below checks each
path leads to a document that really discusses the name. Reasons used to be copied into this
docstring, and that is exactly what failed: the world moved on and the gate went on
enforcing grounds that no longer held (`fairseq`), or that held only for whichever version a
resolver happened to land on (`transformers`) -- both measured in ADR-0086.

**A name belongs in `FORBIDDEN` only if its return would leave every other gate green.**
That is the criterion ADR-0086 applied by injection, name by name:

- If the package cannot be installed without ADR-0084's table gate below firing, then its
  import into `vspeech/` cannot be made to work either, and a broken import announces
  itself -- pytest aborts collection, or the entry-point smoke tests in `tests/test_main.py`
  fail. Those names came off the list; there was nothing left for them to catch.
- If the package installs with every table still green, its import into a lazily-loaded
  module is silent. The outcome gate in `tests/test_runtime_footprint.py` cannot see it
  either, since that measures only what `import vspeech.main` loads. Those names stayed:
  this list is the only thing standing between the runtime and their return.
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

# Module name -> the ADRs that hold the reason it is banned. The reason lives there and only
# there; printing these paths is how a red gate leads a reader to it.
#
# Two each, because the reason has two halves and either alone misleads. The first is the
# decision that took the module out of the runtime. The second is always ADR-0086, which is
# where "and no other gate would catch it coming back" lives -- that is what keeps the name
# on *this list* rather than leaving it to the dependency-table or outcome gates. ADR-0022 in
# particular states its grounds as advisories in the lock, which ADR-0086 measured to be
# version-dependent, so a reader sent only there would land on a rationale this project no
# longer rests on.
FORBIDDEN = {
    "transformers": (
        "docs/adr/0022-hubert-onnx-runtime.md",
        "docs/adr/0086-forbidden-name-list-by-what-else-catches-it.md",
    ),
    "pydantic_settings": (
        "docs/adr/0066-config-input-file-only.md",
        "docs/adr/0086-forbidden-name-list-by-what-else-catches-it.md",
    ),
}

# Taken off that list by ADR-0086's inventory. Recorded here rather than only in the ADR so
# that whoever next reaches for this list can see what guards these names now, and redo the
# judgement instead of guessing at it:
#   torch, torchaudio -- ADR-0084's table gate below, both halves proved by injecting each
#     into pyproject.toml and into uv.lock. What backs those two names up *inside*
#     `vspeech/` is narrower than "the suite breaks loudly", and the difference matters
#     before anyone leans on it. All of the following was injected and measured in this
#     repo (2026-08-12, Python 3.14.5 / win32, `uv sync --all-extras`, torch not installed,
#     exit codes read from a redirected file):
#       - The loud case ADR-0086 measured is an *unconditional* module-level `import torch`
#         in a module the suite imports. Collection aborts, or the `python -m vspeech`
#         smoke tests fail.
#       - A *guarded* one is invisible to the suite. `try: import torch / except
#         ImportError: torch = None` -- the very idiom ctranslate2 uses to pick torch up --
#         put into `vspeech/lib/rvc.py` left the full suite at exit 0 (1178 passed). So did
#         an `import torch` in a function body no test executes
#         (`vspeech/worker/vc.py::rvc_worker`, 1178 passed), and one in a new module under
#         `vspeech/` that no test imports (1178 passed). ADR-0086's census -- 55 modules,
#         52 of them reached at collection -- is a snapshot, and no test pins it. (An
#         import in a function the suite *does* execute still fails loudly: the same line
#         in `rvc.py::load_hubert_model` gave 6 failed, 3 errors.)
#       - What catches all three is `uv run ty check` (exit 1,
#         `error[unresolved-import]`), which lives in `poe check` and **not** in the suite.
#         If the gate being relied on is pytest, these shapes are not in it.
#     Residual ADR-0084 deliberately does not cover: a torch a developer installed by hand
#     is not policed at all (ADR-0084 rejected reading the installed environment, because
#     the offline `uv run --with` overlay legitimately has one). There the import resolves,
#     so ty falls silent as well -- verified with an installed stand-in package: both the
#     function-body form and `except ImportError: pass` gave ty exit 0. (Only the
#     `except ImportError: <name> = None` spelling stays red there, and incidentally, on
#     `invalid-assignment` rather than on the import.) **A hand-installed torch combined
#     with a guarded or function-body import is caught by nothing here.** It is only the
#     unconditional form that the next person to run the suite in a venv without torch
#     catches -- loudly, but later than this list would have.
#   fairseq -- the same table gate, one edge away: `uv add fairseq` in this repo resolves
#     0.12.2 and drags torch 2.13.0 and torchaudio 2.11.0 into uv.lock, which fires it
#     (measured). Two residuals. The edge only goes back so far: `fairseq<0.12` still
#     resolves onto torch (0.11.1), but `fairseq==0.6.2` takes this repo's lock from 86 to
#     92 packages with no torch in them and slips past -- though that is a 2019 release with
#     no HuBERT in it, i.e. nothing the offline converter here could be pointed at. And the
#     edge is *removable by hardening*: putting `constraint-dependencies = ["torch<0"]` in
#     `[tool.uv]` makes a plain `uv add fairseq` backtrack silently to that same 0.6.2 and
#     exit 0, with every gate green (both measured). If that constraint is ever added, put
#     fairseq back on the list above -- ADR-0086 records why.

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
        f"{forbidden} import leaked back into the runtime:\n"
        + "\n".join(offenders)
        + "\nIt was taken out on purpose and no other gate catches it coming back; "
        + f"the reasons are in {' and '.join(FORBIDDEN[forbidden])}."
    )


@pytest.mark.parametrize(
    ("forbidden", "adr"),
    [(name, adr) for name, adrs in sorted(FORBIDDEN.items()) for adr in adrs],
)
def test_every_ban_points_at_a_reason_that_exists(forbidden: str, adr: str):
    """A rotted pointer is worse than no pointer at all.

    Since nothing here says *why* a name is banned, a red gate is worth exactly as much as
    this path. Requiring the document to mention the name as well catches a pointer aimed at
    the wrong ADR, which a plain file-exists check would wave straight through.
    """
    document = REPO_ROOT / adr
    assert document.is_file(), (
        f"the ban on {forbidden} points at {adr}, which does not exist. The reason has to "
        "stay reachable from the failure message; an ADR is never deleted, so find where it "
        "moved (or which ADR superseded it) rather than dropping the pointer."
    )
    spellings = {forbidden, forbidden.replace("_", "-")}
    text = document.read_text(encoding="utf-8")
    assert any(spelling in text for spelling in spellings), (
        f"the ban on {forbidden} points at {adr}, which never mentions it. A reader who "
        "follows that path lands on the wrong decision."
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
        # Neither `torch` nor `torchaudio` is in FORBIDDEN any more (ADR-0086), and they
        # stay here as data because they are the sharpest prefix collision available: one
        # name is a prefix of the other without being its parent package. Pin the dot
        # boundary between them in both directions.
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


def test_the_entry_point_never_loads_a_forbidden_module():
    """Nothing on the startup path drags one back in through a dependency.

    The AST gate above reads `vspeech/` and nothing else, so an import arriving through a
    dependency is invisible to it. `tests/test_runtime_footprint.py` would notice this one
    too, but it reports whatever it finds as an unrecognised newcomer and offers to
    re-record its baseline; for a module that was taken out on purpose, the answer is the
    ADR, not a new baseline. A sys.modules check inside the test process would be
    contaminated by test order, so it runs in a pristine child process.
    """
    names = sorted(FORBIDDEN)
    code = (
        "import sys\n"
        "import vspeech.main\n"
        f"leaked = [name for name in {names!r} if name in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "a module the runtime is not allowed to load reached the startup path.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
        "Where each of these was decided: "
        + "; ".join(f"{name} -> {' and '.join(FORBIDDEN[name])}" for name in names)
    )
