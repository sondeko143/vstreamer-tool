"""Pin `create_session`'s execution-provider choice and where sessions are constructed.

The RVC decoder, HuBERT and RMVPE paths all share this single implementation, so this file
protects all three.
"""

import ast
import platform
from pathlib import Path

import pytest

import vspeech.lib.onnx_session as onnx_session
from vspeech.lib.cuda_util import Device


def _capture(monkeypatch, cuda_available: bool):
    """Replace InferenceSession and capture the providers / provider_options passed in.

    `cuda_available` drives onnxruntime's own provider list, which is what decides the
    EP since ADR-0078 -- not torch.

    `preload_dlls` is replaced too: it has a real side effect (loading CUDA DLLs into
    the process) that these tests must not trigger. The calls it receives are recorded
    under `"preloads"`, and the once-per-process latch is reset so each test starts from
    a fresh process state.
    """
    captured: dict = {"preloads": []}

    def fake_session(path, sess_options, providers, provider_options):
        captured["path"] = path
        captured["providers"] = providers
        captured["provider_options"] = provider_options
        return object()

    available = ["CPUExecutionProvider"]
    if cuda_available:
        available.insert(0, "CUDAExecutionProvider")
    monkeypatch.setattr(onnx_session, "InferenceSession", fake_session)
    monkeypatch.setattr(onnx_session, "get_available_providers", lambda: available)
    monkeypatch.setattr(
        onnx_session,
        "preload_dlls",
        lambda **kwargs: captured["preloads"].append(kwargs),
    )
    monkeypatch.setattr(onnx_session, "_cuda_libraries_loaded", False)
    return captured


def test_cpu_device_never_gets_the_cuda_ep(tmp_path, monkeypatch):
    """Even when CUDA is available, a CPU device must not get the CUDA EP.

    The CUDA EP uses TF32 for fp32 matmuls. HuBERT's features degrade from max_abs
    1.010e-05 to 2.625e-03, which fails the fp32 gate (1e-4) in
    `tests/test_hubert_equivalence.py`.
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cpu"))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]


def test_cuda_device_gets_the_cuda_ep_first(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cuda", 3))

    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert captured["provider_options"][0]["device_id"] == 3


def test_a_bare_cuda_device_yields_device_id_zero(tmp_path, monkeypatch):
    """`Device("cuda")` has index None. None must never be passed to ORT.

    This test is the only thing pinning the `device.index if ... else 0` guard against a
    leaking None.
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cuda"))

    assert captured["provider_options"][0]["device_id"] == 0
    assert captured["provider_options"][0]["device_id"] is not None


def test_cpu_only_box_never_gets_the_cuda_ep(tmp_path, monkeypatch):
    """onnxruntime without a CUDA EP gets the CPU EP only, even for a cuda device.

    Asking ORT what it can actually provide is the point of ADR-0078: a build without
    the CUDA EP would previously still have been asked for it whenever torch happened
    to see a GPU.
    """
    captured = _capture(monkeypatch, cuda_available=False)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cuda", 0))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]


def test_a_cuda_session_preloads_the_cuda_libraries(tmp_path, monkeypatch):
    """The CUDA EP links against cuBLAS/cuFFT/cuDNN, which no longer arrive with torch.

    Without this, `onnxruntime_providers_cuda.dll` fails to load and onnxruntime falls
    back to `CPUExecutionProvider` without raising (ADR-0083).
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cuda", 0))

    assert [(p["cuda"], p["cudnn"]) for p in captured["preloads"]] == [
        (True, False),
        (False, True),
    ]


def test_a_cpu_session_does_not_load_the_cuda_libraries(tmp_path, monkeypatch):
    """A CPU device must not pay for the CUDA libraries -- on a CUDA-capable box too.

    `cuda_available=True` is the point: with it False both guards would block the
    preload and the test could not tell which one did the work.
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cpu"))

    assert captured["preloads"] == []


def test_a_box_without_the_cuda_ep_does_not_load_the_cuda_libraries(
    tmp_path, monkeypatch
):
    """The other guard: a cuda device on a build whose CUDA EP is missing."""
    captured = _capture(monkeypatch, cuda_available=False)

    onnx_session.create_session(tmp_path / "m.onnx", Device("cuda", 0))

    assert captured["preloads"] == []


def test_the_cuda_probe_names_the_generation_this_build_wants():
    """The probe has to carry onnxruntime's CUDA major, not a wildcard."""
    from onnxruntime import cuda_version

    major = cuda_version.split(".")[0]
    assert onnx_session._cuda_probe() == f"cublasLt64_{major}.dll"


def test_a_co_installed_cuda_generation_does_not_win_the_directory(
    tmp_path, monkeypatch
):
    """NVIDIA gives each CUDA generation its own directory under `nvidia/`.

    A version-agnostic glob returns whichever sorts first, which is unrelated to the
    generation onnxruntime was built against -- so `preload_dlls` would be handed a
    directory holding none of the files it wants and load nothing at all.
    """
    root = tmp_path / "nvidia"
    for generation, dll in (
        ("cu12", "cublasLt64_12.dll"),
        ("cu13", "cublasLt64_13.dll"),
        ("cu14", "cublasLt64_14.dll"),
    ):
        bin_dir = root / generation / "bin" / "x86_64"
        bin_dir.mkdir(parents=True)
        (bin_dir / dll).write_bytes(b"")
    monkeypatch.setattr(onnx_session, "_nvidia_roots", lambda: [str(root)])

    assert onnx_session._nvidia_wheel_dir("cublasLt64_13.dll") == str(
        root / "cu13" / "bin" / "x86_64"
    )
    # The hazard this guards against: the wildcard picks the oldest generation present.
    assert onnx_session._nvidia_wheel_dir("cublasLt64_*.dll") == str(
        root / "cu12" / "bin" / "x86_64"
    )


def test_the_cuda_libraries_are_loaded_once_per_process(tmp_path, monkeypatch):
    """RVC opens three sessions (decoder, HuBERT, f0); the DLLs load on the first."""
    captured = _capture(monkeypatch, cuda_available=True)

    for _ in range(3):
        onnx_session.create_session(tmp_path / "m.onnx", Device("cuda", 0))

    assert len(captured["preloads"]) == 2


def test_each_preload_is_pointed_at_a_directory_that_has_what_it_wants():
    """The pinned nvidia wheels must sit where `preload_dlls` is told to look.

    `preload_dlls` is given one directory per half and looks for every file it wants
    directly inside it, so a wheel that moves its layout silently supplies nothing --
    it only prints, and the pipeline dies later at `check_cuda_provider`. onnxruntime's
    own list is the authority on which files those are.

    Skipped when the wheels are absent (an install without the `rvc` extra).
    """
    if platform.system() != "Windows":
        pytest.skip("the pinned wheels and this DLL list are Windows-only")
    from onnxruntime import _get_nvidia_dll_paths

    cuda_dir = onnx_session._nvidia_wheel_dir(onnx_session._cuda_probe())
    cudnn_dir = onnx_session._nvidia_wheel_dir(onnx_session._CUDNN_PROBE)
    if cuda_dir is None or cudnn_dir is None:
        pytest.skip("nvidia CUDA wheels are not installed")

    # cuDNN ships this one only from 9.23; onnxruntime treats it as optional.
    optional = {"cudnn_engines_tensor_ir64_9.dll"}
    missing = [
        str(Path(directory) / relative[-1])
        for directory, cuda, cudnn in (
            (cuda_dir, True, False),
            (cudnn_dir, False, True),
        )
        for relative in _get_nvidia_dll_paths(True, cuda=cuda, cudnn=cudnn)
        if relative[-1] not in optional
        and not (Path(directory) / relative[-1]).is_file()
    ]
    assert missing == []


def _inference_session_construction_sites() -> list[str]:
    """The file names under `vspeech/` that construct an `InferenceSession(...)`."""
    vspeech_dir = Path(__file__).resolve().parents[2] / "vspeech"
    sites = []
    for py_file in sorted(vspeech_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "InferenceSession"
            ):
                sites.append(py_file.name)
    return sites


def test_only_one_place_builds_a_device_aware_session():
    """Session construction that chooses the CUDA EP exists in exactly one place,
    `onnx_session.py`.

    A duplicate could be written under any name, so this counts `InferenceSession`
    construction sites rather than function names.

    `vad.py` is the exception. Silero VAD pins `providers=["CPUExecutionProvider"]`, takes
    no device, and has no EP-selection logic.
    """
    assert sorted(set(_inference_session_construction_sites())) == [
        "onnx_session.py",
        "vad.py",
    ]


def test_rvc_uses_the_shared_factory():
    """`rvc` has no implementation of its own and calls the shared function."""
    import vspeech.lib.rvc as rvc

    assert rvc.create_session is onnx_session.create_session
