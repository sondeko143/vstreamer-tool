from typing import cast

import numpy as np
import torch
from onnxruntime import InferenceSession
from torch.nn import functional

from vspeech.config import RvcConfig
from vspeech.config import RvcQuality
from vspeech.lib.rvc import _align_pitch_to_feats
from vspeech.lib.rvc import _is_model_half
from vspeech.lib.rvc import _pad_input_to_block
from vspeech.lib.rvc import _postprocess
from vspeech.lib.rvc import _quality_padding
from vspeech.lib.rvc import _select_pitch


def test_pad_input_to_block_rounds_up_to_128_and_left_pads():
    raw = np.arange(1, 201, dtype=np.int16)  # 200 samples; block target is 256
    out = _pad_input_to_block(raw.tobytes())
    # Preserved original quirk: when unaligned it PREPENDS `input_size` (256)
    # zeros to the FULL signal, so the result is 256 + 200 = 456 (it does not
    # pad *to* 256). The tail holds the original normalized samples.
    assert out.shape[0] == 456
    np.testing.assert_allclose(out[-200:], raw.astype(np.float32) / 32768.0, rtol=1e-6)
    np.testing.assert_array_equal(out[:256], np.zeros(256))


def test_pad_input_to_block_already_aligned_no_pad():
    raw = np.ones(128, dtype=np.int16)
    out = _pad_input_to_block(raw.tobytes())
    assert out.shape[0] == 128
    np.testing.assert_allclose(out, np.ones(128, dtype=np.float32) / 32768.0, rtol=1e-6)


def test_quality_padding_zero_is_noop():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.zero)
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, 16000, 40000)
    assert t_pad_tgt == 0
    assert audio_pad.shape == (10,)
    np.testing.assert_array_equal(audio_pad.numpy(), audio.squeeze(0).numpy())


def test_quality_padding_positive_reflects():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.one)
    vsr, tsr = 16000, 40000
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, vsr, tsr)
    sec = (1 * (10 - 1)) / vsr  # repeat=1
    assert t_pad_tgt == round(tsr * sec)
    expected = functional.pad(audio, (9, 9), mode="reflect").squeeze(0)
    np.testing.assert_array_equal(audio_pad.numpy(), expected.numpy())
    assert audio_pad.shape[0] == 10 + 2 * 9


class _FakeInput:
    def __init__(self, type_str: str):
        self.type = type_str


class _FakeSession:
    def __init__(self, type_str: str):
        self._inputs = [_FakeInput(type_str)]

    def get_inputs(self):
        return self._inputs


def test_is_model_half_float_is_false():
    session = cast(InferenceSession, _FakeSession("tensor(float)"))
    assert _is_model_half(session) is False


def test_is_model_half_float16_is_true():
    session = cast(InferenceSession, _FakeSession("tensor(float16)"))
    assert _is_model_half(session) is True


def test_align_pitch_to_feats_trims_tail():
    pitch = torch.arange(10).view(1, -1)
    pitchf = torch.arange(10).view(1, -1).float()
    p, pf = _align_pitch_to_feats(pitch, pitchf, 4)
    assert p is not None and pf is not None
    np.testing.assert_array_equal(p.numpy(), np.array([[6, 7, 8, 9]]))
    np.testing.assert_array_equal(
        pf.numpy(), np.array([[6, 7, 8, 9]], dtype=np.float32)
    )


def test_align_pitch_to_feats_none_passthrough():
    assert _align_pitch_to_feats(None, None, 4) == (None, None)


def test_postprocess_no_trim_when_zero():
    audio1 = torch.arange(6, dtype=torch.int16)
    out = _postprocess(audio1, 0)
    np.testing.assert_array_equal(out, np.arange(6, dtype=np.int16))


def test_postprocess_trims_both_ends():
    audio1 = torch.arange(10, dtype=torch.int16)
    out = _postprocess(audio1, 2)
    np.testing.assert_array_equal(out, np.arange(10, dtype=np.int16)[2:-2])


def test_select_pitch_disabled_returns_none():
    audio_pad = torch.zeros(16000, dtype=torch.float32)
    result = _select_pitch(
        audio_pad=audio_pad,
        rvc_config=RvcConfig(),
        f0_enabled=False,
        p_len=10,
        device=torch.device("cpu"),
        rmvpe_session=None,
    )
    assert result == (None, None)
