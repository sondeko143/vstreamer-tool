"""onnxruntime セッションの execution provider 選択を固定する。

このモジュールは RVC decoder / HuBERT / RMVPE の 3 セッションすべてが通る唯一の入口。
かつては `rvc.create_session` と `pitch_extract.create_rmvpe_session` に同じ 20 行が
重複しており、前者だけが device を尊重するよう直されて後者が取り残された。
実装を 1 本にしたので、このテストが 3 経路すべてを守る。
"""

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


def test_rvc_and_pitch_extract_share_one_implementation():
    """`rvc.create_session` と `pitch_extract` が同じ関数を指していること。

    片方だけを直したことで生まれたバグを二度と作らないための構造ゲート。
    """
    import vspeech.lib.pitch_extract as pitch_extract
    import vspeech.lib.rvc as rvc

    assert rvc.create_session is onnx_session.create_session
    assert pitch_extract.create_session is onnx_session.create_session
    assert not hasattr(pitch_extract, "create_rmvpe_session")
