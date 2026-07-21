from typing import cast

import numpy as np
import pytest
import torch
from onnxruntime import InferenceSession

from vspeech.config import F0ExtractorType
from vspeech.lib.pitch_extract import pitch_extract


class FakeRmvpeSession:
    """Stands in for a waveform-input rmvpe.onnx InferenceSession.

    Mirrors the yxlllc/VCClient-style export: consumes ``waveform`` + ``threshold``
    and returns the f0 output first, followed by an unused ``uv`` mask (the yxlllc
    export emits both; the w-okada re-export emits only the single f0 named
    ``pitchf``). Records calls so the test can pin the input contract and that we
    read output index 0 rather than depending on the f0/pitchf output name.
    """

    def __init__(self, f0: np.ndarray):
        self._f0 = f0
        self.calls: list[tuple[object, dict]] = []

    def run(self, output_names, input_feed):
        self.calls.append((output_names, dict(input_feed)))
        uv = np.zeros_like(self._f0)
        return [self._f0, uv]


def test_pitch_extract_rmvpe_routes_to_session_and_returns_aligned_pitch():
    sr = 16000
    window = 160
    audio = torch.zeros(sr, dtype=torch.float32)
    p_len = sr // window  # 100 frames at a 10ms hop
    # rmvpe.onnx emits f0 in Hz with unvoiced frames already zeroed; shape (1, N).
    fake_f0 = np.full((1, p_len), 220.0, dtype=np.float32)
    session = FakeRmvpeSession(fake_f0)

    f0_coarse, f0bak = pitch_extract(
        audio,
        f0_up_key=0,
        sr=sr,
        window=window,
        f0_extractor=F0ExtractorType.rmvpe,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )

    # Routed to the onnx session exactly once. We request all outputs (None) and
    # read index 0, so the f0/pitchf output-name difference between exports
    # doesn't matter.
    assert len(session.calls) == 1
    output_names, input_feed = session.calls[0]
    assert output_names is None
    # waveform fed as a batched (1, N) float32 tensor; threshold as a (1,) array.
    assert input_feed["waveform"].dtype == np.float32
    assert input_feed["waveform"].ndim == 2
    assert input_feed["waveform"].shape[0] == 1
    assert input_feed["threshold"].shape == (1,)

    # f0 returned in Hz (no transpose), length aligned to the frame count,
    # coarse bins within the valid 1..255 range.
    assert f0bak.shape == (p_len,)
    np.testing.assert_allclose(f0bak, 220.0, rtol=1e-5)
    assert f0_coarse.shape == (p_len,)
    assert f0_coarse.min() >= 1
    assert f0_coarse.max() <= 255


def test_pitch_extract_rmvpe_requires_session():
    with pytest.raises(ValueError):
        pitch_extract(
            torch.zeros(16000, dtype=torch.float32),
            f0_up_key=0,
            sr=16000,
            window=160,
            f0_extractor=F0ExtractorType.rmvpe,
            f0_session=None,
            silence_front=0,
        )


class FakeFcpeSession:
    """Stands in for a waveform-only fcpe.onnx returning a 3-D ``(1, T, 1)`` f0.

    The real fcpe.onnx bakes threshold/sample_rate/decoder_mode into the graph
    and emits f0 with shape ``(1, T, 1)``; the runtime feeds only ``waveform``.
    Records calls so tests can pin the input contract, the output-index-0 read,
    and the min-length pad guard. Takes a 1-D f0 and returns it as ``(1, T, 1)``.
    """

    def __init__(self, f0_1d: np.ndarray):
        self._f0 = f0_1d.reshape(1, -1, 1).astype(np.float32)
        self.calls: list[tuple[object, dict]] = []

    def run(self, output_names, input_feed):
        self.calls.append((output_names, dict(input_feed)))
        return [self._f0]


def test_pitch_extract_fcpe_routes_to_session_waveform_only():
    sr = 16000
    window = 160
    audio = torch.zeros(sr, dtype=torch.float32)
    p_len = sr // window
    # 実 fcpe.onnx は p_len 以上のフレームを返す (baked hop=160 == window)。realistic に +2。
    session = FakeFcpeSession(np.full(p_len + 2, 220.0, dtype=np.float32))

    f0_coarse, f0bak = pitch_extract(
        audio,
        f0_up_key=0,
        sr=sr,
        window=window,
        f0_extractor=F0ExtractorType.fcpe,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )

    assert len(session.calls) == 1
    output_names, input_feed = session.calls[0]
    assert output_names is None
    # fcpe.onnx takes ONLY waveform (threshold/sr/decoder baked at export).
    assert set(input_feed) == {"waveform"}
    assert input_feed["waveform"].dtype == np.float32
    assert input_feed["waveform"].ndim == 2
    assert input_feed["waveform"].shape[0] == 1

    # 3-D (1, T, 1) を .squeeze() で 1-D に潰し、フレーム数は p_len 以上。
    assert f0bak.ndim == 1
    assert len(f0bak) >= p_len
    np.testing.assert_allclose(f0bak, 220.0, rtol=1e-5)
    assert f0_coarse.min() >= 1
    assert f0_coarse.max() <= 255


def test_pitch_extract_fcpe_single_frame_does_not_collapse_to_0d():
    # T=1 -> (1,1,1).squeeze() は 0-d になり、呼び出し側の f0[:p_len] が IndexError に
    # なる。atleast_1d が 1-D を保証することを固定する。
    session = FakeFcpeSession(np.array([220.0], dtype=np.float32))
    _coarse, f0bak = pitch_extract(
        torch.zeros(16000, dtype=torch.float32),
        f0_up_key=0,
        sr=16000,
        window=160,
        f0_extractor=F0ExtractorType.fcpe,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )
    assert f0bak.ndim == 1
    assert f0bak.shape == (1,)


def test_pitch_extract_fcpe_pads_short_input_to_min_samples():
    # 焼込み reflect-pad は最小長を要求する。短い入力は onnx に渡す前に底上げされる。
    from vspeech.lib.pitch_extract import FCPE_MIN_SAMPLES

    session = FakeFcpeSession(np.full(3, 220.0, dtype=np.float32))
    pitch_extract(
        torch.zeros(100, dtype=torch.float32),
        f0_up_key=0,
        sr=16000,
        window=160,
        f0_extractor=F0ExtractorType.fcpe,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )
    assert session.calls[0][1]["waveform"].shape[-1] >= FCPE_MIN_SAMPLES


def test_pitch_extract_fcpe_requires_session():
    with pytest.raises(ValueError):
        pitch_extract(
            torch.zeros(16000, dtype=torch.float32),
            f0_up_key=0,
            sr=16000,
            window=160,
            f0_extractor=F0ExtractorType.fcpe,
            f0_session=None,
            silence_front=0,
        )
