"""onnxruntime セッションの execution provider 選択を固定する。

このモジュールは RVC decoder / HuBERT / RMVPE の 3 セッションすべてが通る唯一の入口。
かつては `rvc.create_session` と `pitch_extract.create_rmvpe_session` に同じ 20 行が
重複しており、前者だけが device を尊重するよう直されて後者が取り残された。
実装を 1 本にしたので、このテストが 3 経路すべてを守る。
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

    以前は `torch.cuda.is_available()` だけで判定しており、config で CPU を指定しても
    GPU で走っていた。HuBERT の fp32 等価ゲートはこの差 (CUDA EP の TF32, max_abs
    2.6e-3) で落ちる。
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

    `rvc.create_session` と `pitch_extract.create_rmvpe_session` は同じ 20 行の複製で、
    前者だけが device を尊重するよう直され、後者が取り残された。**複製こそが原因**なので、
    「create_session が 2 つある」ではなく「InferenceSession を組み立てる場所が増えた」を
    検出する。新しい複製はどんな名前で書かれてもここで落ちる。

    `vad.py` は例外。Silero VAD は意図的に `providers=["CPUExecutionProvider"]` 固定で、
    device 引数を取らない（EP 選択のロジックを持たない）ので、複製ではない。
    """
    assert sorted(set(_inference_session_construction_sites())) == [
        "onnx_session.py",
        "vad.py",
    ]


def test_rvc_uses_the_shared_factory():
    """`rvc` が自前の実装を持たず、共有の関数を呼んでいること。"""
    import vspeech.lib.rvc as rvc

    assert rvc.create_session is onnx_session.create_session
