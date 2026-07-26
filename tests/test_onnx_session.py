"""Pin `create_session`'s execution-provider choice and where sessions are constructed.

The RVC decoder, HuBERT and RMVPE paths all share this single implementation, so this file
protects all three.
"""

import ast
from pathlib import Path

import torch

import vspeech.lib.onnx_session as onnx_session


def _capture(monkeypatch, cuda_available: bool):
    """Replace InferenceSession and capture the providers / provider_options passed in."""
    captured: dict = {}

    def fake_session(path, sess_options, providers, provider_options):
        captured["path"] = path
        captured["providers"] = providers
        captured["provider_options"] = provider_options
        return object()

    monkeypatch.setattr(onnx_session, "InferenceSession", fake_session)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    return captured


def test_cpu_device_never_gets_the_cuda_ep(tmp_path, monkeypatch):
    """Even when CUDA is available, a CPU device must not get the CUDA EP.

    The CUDA EP uses TF32 for fp32 matmuls. HuBERT's features degrade from max_abs
    1.010e-05 to 2.625e-03, which fails the fp32 gate (1e-4) in
    `tests/test_hubert_equivalence.py`.
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cpu"))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]


def test_cuda_device_gets_the_cuda_ep_first(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cuda", 3))

    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert captured["provider_options"][0]["device_id"] == 3


def test_a_bare_cuda_device_yields_device_id_zero(tmp_path, monkeypatch):
    """`torch.device("cuda")` has index None. None must never be passed to ORT.

    This test is the only thing pinning the `device.index if ... else 0` guard against a
    leaking None.
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cuda"))

    assert captured["provider_options"][0]["device_id"] == 0
    assert captured["provider_options"][0]["device_id"] is not None


def test_cpu_only_box_never_gets_the_cuda_ep(tmp_path, monkeypatch):
    """With no CUDA available, even a requested cuda device gets the CPU EP only."""
    captured = _capture(monkeypatch, cuda_available=False)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cuda", 0))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]


def _inference_session_construction_sites() -> list[str]:
    """The file names under `vspeech/` that construct an `InferenceSession(...)`."""
    vspeech_dir = Path(__file__).resolve().parents[1] / "vspeech"
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
