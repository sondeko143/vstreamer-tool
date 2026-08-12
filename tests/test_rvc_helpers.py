from typing import cast

import numpy as np
from onnxruntime import InferenceSession

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


def test_to_hubert_rate_passes_16k_through_untouched():
    """16kHz in means no filtering at all -- the same pass-through torchaudio had.

    Filtering a signal that is already at the model's rate would change every
    utterance captured at 16kHz for no reason.
    """
    from vspeech.lib.rvc import _to_hubert_rate

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(1024).astype(np.float32)
    out = _to_hubert_rate(audio, 16000)
    np.testing.assert_array_equal(out, audio)


def test_to_hubert_rate_uses_the_shared_polyphase_resampler():
    """Non-16k input goes through vspeech.lib.resample and nothing else (ADR-0082).

    Asserted bit-exactly against that module rather than with a tolerance: a
    tolerance would still pass if a second, differently-tuned resampler appeared.
    """
    from vspeech.lib.resample import PolyphaseResampler
    from vspeech.lib.rvc import _to_hubert_rate

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(48000).astype(np.float32)
    out = _to_hubert_rate(audio, 48000)
    expected = PolyphaseResampler(48000, 16000).resample_full(audio)
    np.testing.assert_array_equal(out, expected)
    assert out.shape[0] == 16000


def test_to_hubert_rate_normalises_the_float64_of_a_padded_block():
    """_pad_input_to_block returns float64 whenever it prepends zeros.

    Both branches must still hand back float32. The resampling branch would get there
    on its own (`PolyphaseResampler.process` coerces), but the passthrough branch
    returns the array it was given, so without the cast a 16kHz float64 utterance
    would escape as float64.
    """
    from vspeech.lib.rvc import _pad_input_to_block
    from vspeech.lib.rvc import _to_hubert_rate

    padded = _pad_input_to_block(np.ones(200, dtype=np.int16).tobytes())
    assert padded.dtype == np.float64  # the premise of this test, not an assumption

    resampled = _to_hubert_rate(padded, 48000)
    assert resampled.dtype == np.float32
    assert resampled.shape[0] == round(256 * 16000 / 48000)

    passthrough = _to_hubert_rate(padded, 16000)
    assert passthrough.dtype == np.float32
    assert passthrough.shape[0] == 256


def test_quality_padding_zero_is_noop():
    audio = np.arange(10, dtype=np.float32)
    cfg = RvcConfig(quality=RvcQuality.zero)
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, 40000)
    assert t_pad_tgt == 0
    assert audio_pad.shape == (10,)
    np.testing.assert_array_equal(audio_pad, audio)


def test_quality_padding_positive_reflects():
    audio = np.arange(10, dtype=np.float32)
    cfg = RvcConfig(quality=RvcQuality.one)
    tsr = 40000
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, tsr)
    # input pad is repeat*(N-1) samples at the 16k internal rate
    assert t_pad_tgt == round(9 * tsr / 16000)
    # The reflection itself, spelled out rather than delegated to np.pad, so this also
    # pins that the edge sample is NOT repeated -- the one way numpy's "reflect" could
    # have differed from the torch pad it replaced (ADR-0081).
    expected = np.concatenate(
        [audio[1:][::-1], audio, audio[:-1][::-1]], dtype=np.float32
    )
    np.testing.assert_array_equal(audio_pad, expected)
    assert audio_pad.shape[0] == 10 + 2 * 9


def test_quality_padding_output_pad_independent_of_original_rate():
    # The audio reaching _quality_padding is already resampled to the 16k
    # internal rate, so the output-side pad must scale by target_sr / 16000 --
    # the remote's original capture rate must not change it.
    audio = np.arange(10, dtype=np.float32)
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
    pitch = np.arange(10, dtype=np.int64).reshape(1, -1)
    pitchf = np.arange(10, dtype=np.float32).reshape(1, -1)
    p, pf = _align_pitch_to_feats(pitch, pitchf, 4)
    assert p is not None and pf is not None
    np.testing.assert_array_equal(p, np.array([[6, 7, 8, 9]]))
    np.testing.assert_array_equal(pf, np.array([[6, 7, 8, 9]], dtype=np.float32))


def test_align_pitch_to_feats_none_passthrough():
    assert _align_pitch_to_feats(None, None, 4) == (None, None)


def test_postprocess_no_trim_when_zero():
    audio1 = np.arange(6, dtype=np.int16)
    out = _postprocess(audio1, 0)
    np.testing.assert_array_equal(out, np.arange(6, dtype=np.int16))


def test_postprocess_trims_both_ends():
    audio1 = np.arange(10, dtype=np.int16)
    out = _postprocess(audio1, 2)
    np.testing.assert_array_equal(out, np.arange(10, dtype=np.int16)[2:-2])


def test_to_int16_saturates_out_of_range():
    # RVC/vocoder output can overshoot +-1.0; the int16 cast MUST clip first.
    # An unclipped cast wraps modulo 2**16 (e.g. 1.05 -> -31131), flipping a
    # peak's sign into a loud click. Clipping saturates to the rail instead.
    vals = np.array([-1.5, -1.0, 0.0, 1.0, 1.05, 1.5], dtype=np.float32)
    out = _to_int16(vals)
    assert out.dtype == np.int16
    assert out[2] == 0
    assert out[3] == 32767  # 1.0 * 32767.5 -> clip 32767
    assert out[4] == 32767  # 1.05 would WRAP to -31131 unclipped
    assert out[5] == 32767  # 1.5 saturates high
    assert out[0] == -32768  # -1.5 saturates low
    # a full block of overshoot must all saturate high, never sign-flip negative
    assert int(_to_int16(np.full(64, 1.3, dtype=np.float32)).min()) == 32767


def test_to_int16_does_not_saturate_a_float16_decoder_output():
    """Characterization of the defect recorded in `_to_int16`'s docstring.

    Every fp16 RVC decoder emits float16, and 32767.0 is not representable there: it
    rounds to 32768.0, so the upper bound of the clip *is* 32768.0 and a full-scale
    sample casts to -32768. This is what the torch implementation did too (measured bit
    for bit over 2M samples), so it is pinned rather than fixed here -- fixing it moves
    in-range samples by up to 1 LSB and so cannot ride along with a bit-exactness gate.
    Delete this test when that fix lands.
    """
    with np.errstate(over="ignore"):
        out = _to_int16(np.array([1.0, 1.5, -1.5], dtype=np.float16))
    assert out[0] == -32768  # NOT 32767: the clip did not saturate
    assert out[1] == -32768
    assert out[2] == -32768  # the low rail is exact, so this one is genuinely clipped
    # ... while one LSB below full scale still behaves.
    assert _to_int16(np.array([0.999, -0.999], dtype=np.float16))[0] > 0


def test_select_pitch_disabled_returns_none():
    audio_pad = np.zeros(16000, dtype=np.float32)
    result = _select_pitch(
        audio_pad=audio_pad,
        rvc_config=RvcConfig(),
        f0_enabled=False,
        p_len=10,
        f0_session=None,
    )
    assert result == (None, None)


# Device resolution moved to tests/test_cuda_util.py along with the code (ADR-0078).
# Both behaviours these tests pinned -- `gpu_id = 0` is a real device, and no GPU
# setting means CPU -- are covered there, without torch.


def test_ort_device_id_defaults_a_bare_cuda_device_to_zero():
    """`Device("cuda")` has index None, and ORT needs a concrete ordinal.

    Passing None through would bind the inputs and the output on a device id ORT cannot
    resolve; 0 is the same default `create_session` picks, so the values land on the card
    the session runs on.

    [Open, deferred 2026-08-12 -- outside ADR-0080's scope] The last assertion pins a call
    site that does not exist. `_run_on_device` returns through `session.run` whenever
    `device.type != "cuda"`, so `_ort_device_id` is never reached with a CPU device and no
    production behaviour depends on what it answers there. Keeping it freezes an
    implementation detail (that the helper is total rather than rejecting a non-CUDA
    device) for nothing, which is the kind of assertion that later blocks a refactor it
    was never protecting. Deferred rather than deleted because "pin less" and "pin that
    the helper does not raise" are both defensible, and choosing is a judgement about how
    this helper should behave -- not a fix.
    """
    from vspeech.lib.cuda_util import Device
    from vspeech.lib.rvc import _ort_device_id

    assert _ort_device_id(Device("cuda")) == 0
    assert _ort_device_id(Device("cuda", 0)) == 0
    assert _ort_device_id(Device("cuda", 1)) == 1
    assert _ort_device_id(Device("cpu")) == 0
