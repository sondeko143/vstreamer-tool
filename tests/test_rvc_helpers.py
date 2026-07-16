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
from vspeech.lib.rvc import _to_int16


def test_pad_input_to_block_rounds_up_to_128_and_left_pads():
    raw = np.arange(1, 201, dtype=np.int16)  # 200 samples; next 128-multiple is 256
    out = _pad_input_to_block(raw.tobytes())
    # Left-pads UP TO the next multiple of 128 (256): prepends only the
    # remainder (56) zeros, NOT a full extra block. Padding more than the
    # remainder roughly doubles the signal duration (the RVC backlog bug).
    assert out.shape[0] == 256
    np.testing.assert_array_equal(out[:56], np.zeros(56))
    np.testing.assert_allclose(out[-200:], raw.astype(np.float32) / 32768.0, rtol=1e-6)


def test_pad_input_to_block_already_aligned_no_pad():
    raw = np.ones(128, dtype=np.int16)
    out = _pad_input_to_block(raw.tobytes())
    assert out.shape[0] == 128
    np.testing.assert_allclose(out, np.ones(128, dtype=np.float32) / 32768.0, rtol=1e-6)


def test_quality_padding_zero_is_noop():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.zero)
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, 40000)
    assert t_pad_tgt == 0
    assert audio_pad.shape == (10,)
    np.testing.assert_array_equal(audio_pad.numpy(), audio.squeeze(0).numpy())


def test_quality_padding_positive_reflects():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.one)
    tsr = 40000
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, tsr)
    # input pad is repeat*(N-1) samples at the 16k internal rate
    assert t_pad_tgt == round(9 * tsr / 16000)
    expected = functional.pad(audio, (9, 9), mode="reflect").squeeze(0)
    np.testing.assert_array_equal(audio_pad.numpy(), expected.numpy())
    assert audio_pad.shape[0] == 10 + 2 * 9


def test_quality_padding_output_pad_independent_of_original_rate():
    # The audio reaching _quality_padding is already resampled to the 16k
    # internal rate, so the output-side pad must scale by target_sr / 16000 --
    # the remote's original capture rate must not change it.
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.one)
    _, t_pad_tgt = _quality_padding(audio, cfg, 48000)
    assert t_pad_tgt == round(9 * 48000 / 16000)  # 27, not 54


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


def test_to_int16_saturates_out_of_range():
    # RVC/vocoder output can overshoot +-1.0; the int16 cast MUST clamp first.
    # An unclamped cast wraps modulo 2**16 (e.g. 1.05 -> -31131), flipping a
    # peak's sign into a loud click. Clamping saturates to the rail instead.
    vals = torch.tensor([-1.5, -1.0, 0.0, 1.0, 1.05, 1.5])
    out = _to_int16(vals)
    assert out.dtype == torch.int16
    assert out[2].item() == 0
    assert out[3].item() == 32767  # 1.0 * 32767.5 -> clamp 32767
    assert out[4].item() == 32767  # 1.05 would WRAP to -31131 unclamped
    assert out[5].item() == 32767  # 1.5 saturates high
    assert out[0].item() == -32768  # -1.5 saturates low
    # a full block of overshoot must all saturate high, never sign-flip negative
    assert int(_to_int16(torch.full((64,), 1.3)).min().item()) == 32767


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


def test_element_type_maps_supported_dtypes():
    import numpy as np
    import torch

    from vspeech.lib.rvc import _element_type

    assert _element_type(torch.float16) is np.float16
    assert _element_type(torch.float32) is np.float32
    assert _element_type(torch.int64) is np.int64


def test_element_type_rejects_unsupported_dtype():
    import pytest
    import torch

    from vspeech.lib.rvc import _element_type

    with pytest.raises(ValueError, match="Unsupported dtype"):
        _element_type(torch.bfloat16)


def test_ort_output_to_torch_falls_back_to_numpy():
    """dlpack が使えない ORT 値でも numpy 経由で torch tensor を返すこと。"""
    import numpy as np
    import torch

    from vspeech.lib.rvc import _ort_output_to_torch

    # 繰り延べ: このスタブは _ortvalue も to_dlpack も持たないので、内側の
    # `except AttributeError` から想定外の AttributeError で外側 `except Exception`
    # に落ちて numpy fallback に至る。「dlpack が無い」のか「dlpack が壊れている」のかを
    # このテストは区別できない。
    class _NoDlpack:
        def numpy(self):
            return np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    out = _ort_output_to_torch(_NoDlpack(), torch.device("cpu"))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 3)
    assert out.dtype == torch.float32
    assert out[0, 1, 2].item() == 5.0


def test_get_device_treats_gpu_id_zero_as_a_real_device(monkeypatch):
    """`gpu_id = 0` は「未設定」ではなく cuda:0。

    「未設定」を表すのは `None`（`gpu_id: int | None = None`）。`if gpu_id and ...` と
    書くと 0 が falsy で弾かれ、`config.toml.example` が載せている `gpu_id = 0` の構成が
    CPU device に落ちる。すると `check_cuda_provider` が vc worker の起動時に落ちる。
    """
    import torch

    import vspeech.lib.cuda_util as cuda_util

    class _Prop:
        name = "FakeGPU"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Prop())

    device, name = cuda_util.get_device(0, "")
    assert device == torch.device("cuda", 0)
    assert name == "FakeGPU"


def test_get_device_falls_back_to_cpu_when_gpu_id_is_none(monkeypatch):
    import torch

    import vspeech.lib.cuda_util as cuda_util

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, name = cuda_util.get_device(None, "")
    assert device == torch.device("cpu")
    assert name == "cpu"
