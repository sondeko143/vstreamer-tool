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
        rmvpe_session=cast(InferenceSession, session),
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
            rmvpe_session=None,
            silence_front=0,
        )
