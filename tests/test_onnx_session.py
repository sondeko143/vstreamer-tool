"""`create_session` の execution provider 選択と、その生成箇所を固定する。

RVC decoder / HuBERT / RMVPE の 3 経路がこの 1 実装を共有するので、ここが 3 つとも守る。
"""

import ast
from pathlib import Path

import torch

import vspeech.lib.onnx_session as onnx_session


def _capture(monkeypatch, cuda_available: bool):
    """InferenceSession を差し替え、渡された providers / provider_options を捕まえる。"""
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
    """CUDA が使えても device が CPU なら CUDA EP を積まないこと。

    CUDA EP は fp32 の行列積に TF32 を使う。HuBERT の特徴量は max_abs が 1.010e-05 から
    2.625e-03 へ悪化し、`tests/test_hubert_equivalence.py` の fp32 ゲート (1e-4) が落ちる。
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
    """`torch.device("cuda")` は index が None。ORT に None を渡してはいけない。

    このテストだけが `device.index if ... else 0` の None 漏れ対策を固定している。
    """
    captured = _capture(monkeypatch, cuda_available=True)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cuda"))

    assert captured["provider_options"][0]["device_id"] == 0
    assert captured["provider_options"][0]["device_id"] is not None


def test_cpu_only_box_never_gets_the_cuda_ep(tmp_path, monkeypatch):
    """CUDA が無ければ、cuda device を要求されても CPU EP のみ。"""
    captured = _capture(monkeypatch, cuda_available=False)

    onnx_session.create_session(tmp_path / "m.onnx", torch.device("cuda", 0))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["provider_options"] == [{}]


def _inference_session_construction_sites() -> list[str]:
    """`vspeech/` 配下で `InferenceSession(...)` を組み立てているファイル名。"""
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
    """CUDA EP を選ぶセッション生成は `onnx_session.py` の 1 箇所だけ。

    複製は任意の名前で書けるので、関数名ではなく `InferenceSession` の構築箇所を数える。

    `vad.py` は例外。Silero VAD は `providers=["CPUExecutionProvider"]` 固定で device を
    取らず、EP 選択のロジックを持たない。
    """
    assert sorted(set(_inference_session_construction_sites())) == [
        "onnx_session.py",
        "vad.py",
    ]


def test_rvc_uses_the_shared_factory():
    """`rvc` が自前の実装を持たず、共有の関数を呼んでいること。"""
    import vspeech.lib.rvc as rvc

    assert rvc.create_session is onnx_session.create_session
