import os
from pathlib import Path

import numpy as np
import pytest

from vspeech.lib.vad import apply_vad_gate
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_gate_mask
from vspeech.lib.vad import speech_probs


def test_should_skip_when_no_window_reaches_threshold():
    probs = np.array([0.1, 0.2, 0.05, 0.3])
    skip, ratio = should_skip_vc(probs, threshold=0.5, min_speech_ratio=0.1)
    assert skip
    assert ratio == 0.0


def test_should_not_skip_when_enough_speech_windows():
    probs = np.array([0.9, 0.8, 0.1, 0.1])
    skip, ratio = should_skip_vc(probs, threshold=0.5, min_speech_ratio=0.1)
    assert not skip
    assert ratio == 0.5


def test_should_skip_empty_probs():
    skip, ratio = should_skip_vc(np.zeros(0), threshold=0.5, min_speech_ratio=0.1)
    assert skip
    assert ratio == 0.0


def test_gate_mask_full_speech_is_all_ones():
    probs = np.full(10, 0.9)
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=0.0, min_gain=0.0)
    np.testing.assert_array_equal(mask, np.ones(10))


def test_gate_mask_nonspeech_gets_min_gain():
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.9])
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=0.0, min_gain=0.25)
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.25, 0.25, 1.0])


def test_gate_mask_pad_dilates_speech_regions():
    # 1 窓 = 32ms。pad_ms=32 -> speech の前後 1 窓ずつ開く。
    probs = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=32.0, min_gain=0.0)
    np.testing.assert_array_equal(mask, [0.0, 1.0, 1.0, 1.0, 0.0])


def test_apply_vad_gate_mutes_nonspeech_half():
    out = (np.ones(1000) * 10000).astype(np.int16)
    gains = np.array([1.0, 0.0])
    res = apply_vad_gate(out, gains)
    assert res.dtype == np.int16
    assert res.shape[0] == out.shape[0]
    assert res[0] == 10000
    assert res[-1] == 0


def test_apply_vad_gate_empty_gains_returns_output_unchanged():
    out = (np.ones(100) * 5000).astype(np.int16)
    res = apply_vad_gate(out, np.zeros(0))
    np.testing.assert_array_equal(res, out)


class _StubSession:
    """Duck-typed InferenceSession: records feeds, returns a fixed prob and
    increments the state by 1 per call so state threading is observable."""

    def __init__(self, prob: float = 0.7):
        self.feeds: list[dict] = []
        self._prob = prob

    def run(self, output_names, input_feed):
        self.feeds.append({k: np.copy(v) for k, v in input_feed.items()})
        state = input_feed["state"] + 1.0
        return np.array([[self._prob]], dtype=np.float32), state


def test_speech_probs_window_count_and_feed_shapes():
    session = _StubSession()
    audio = np.zeros(512 * 3 + 100, dtype=np.float32)
    probs = speech_probs(session, audio)
    # 3 full windows + 1 zero-padded tail window
    assert probs.shape[0] == 4
    np.testing.assert_allclose(probs, 0.7)
    first = session.feeds[0]
    assert first["input"].shape == (1, 512 + 64)
    assert first["state"].shape == (2, 1, 128)
    assert first["sr"] == 16000


def test_speech_probs_carries_context_and_state():
    session = _StubSession()
    audio = np.arange(1024, dtype=np.float32)
    speech_probs(session, audio)
    second = session.feeds[1]
    # window 2 の context = window 1 の末尾 64 サンプル
    np.testing.assert_array_equal(
        second["input"][0, :64], np.arange(448, 512, dtype=np.float32)
    )
    # state はスタブが 1 ずつ加算 -> 2 呼び出し目にはゼロ +1 が届いている
    np.testing.assert_array_equal(second["state"], np.ones((2, 1, 128)))


def test_speech_probs_empty_audio_returns_empty():
    session = _StubSession()
    probs = speech_probs(session, np.zeros(0, dtype=np.float32))
    assert probs.shape[0] == 0
    assert not session.feeds


_VAD_MODEL_ENV = "VSPEECH_VAD_MODEL"
_vad_model = os.environ.get(_VAD_MODEL_ENV)
VAD_MODEL = Path(_vad_model) if _vad_model else None

requires_vad_model = pytest.mark.skipif(
    VAD_MODEL is None or not VAD_MODEL.exists(),
    reason=f"${_VAD_MODEL_ENV} not set or model missing",
)


@requires_vad_model
def test_real_model_silence_and_noise_score_low():
    pytest.importorskip("onnxruntime")
    from vspeech.lib.vad import create_vad_session

    assert VAD_MODEL is not None
    session = create_vad_session(VAD_MODEL)
    silence = np.zeros(16000, dtype=np.float32)
    assert float(speech_probs(session, silence).max()) < 0.3
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    assert float(np.mean(speech_probs(session, noise) >= 0.5)) < 0.1


def test_create_vad_session_missing_file_fails_loudly():
    pytest.importorskip("onnxruntime")
    from vspeech.lib.vad import create_vad_session

    with pytest.raises(FileNotFoundError) as excinfo:
        create_vad_session(Path("./no-such-vad-model.onnx"))
    assert "silero" in str(excinfo.value).lower()


def test_vc_config_vad_defaults_are_off_and_sane():
    from vspeech.config import VcConfig

    config = VcConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1
    assert config.vad_speech_pad_ms == 100.0
    assert config.vad_min_gain == 0.0
