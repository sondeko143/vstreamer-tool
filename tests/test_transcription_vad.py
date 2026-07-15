from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from vspeech.config import SampleFormat
from vspeech.config import TranscriptionConfig
from vspeech.shared_context import SoundInput

if TYPE_CHECKING:
    from onnxruntime import InferenceSession


def test_transcription_config_vad_defaults_are_off_and_sane():
    from vspeech.config import TranscriptionConfig

    config = TranscriptionConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1


def test_create_session_none_when_gate_disabled():
    from vspeech.config import TranscriptionConfig
    from vspeech.worker.transcription import create_transcription_vad_session

    assert create_transcription_vad_session(TranscriptionConfig()) is None


class _StubVad:
    """Duck-typed InferenceSession: returns a fixed speech prob per window and
    advances the recurrent state so state threading stays observable."""

    def __init__(self, prob: float):
        self._prob = prob

    def run(self, output_names, input_feed):
        state = input_feed["state"] + 1.0
        return np.array([[self._prob]], dtype=np.float32), state


def _sound_16k_int16(seconds: float = 1.0) -> SoundInput:
    n = int(seconds * 16000)
    data = np.zeros(n, dtype=np.int16).tobytes()
    return SoundInput(data=data, rate=16000, format=SampleFormat.INT16, channels=1)


async def test_vad_skip_disabled_returns_false():
    from vspeech.worker.transcription import vad_should_skip

    assert (
        await vad_should_skip(None, _sound_16k_int16(), TranscriptionConfig(), "")
        is False
    )


async def test_vad_skip_low_speech_returns_true():
    from vspeech.worker.transcription import vad_should_skip

    cfg = TranscriptionConfig(
        vad_gate=True, vad_threshold=0.5, vad_min_speech_ratio=0.1
    )
    session = cast("InferenceSession", _StubVad(0.1))
    assert await vad_should_skip(session, _sound_16k_int16(), cfg, "") is True


async def test_vad_pass_high_speech_returns_false():
    from vspeech.worker.transcription import vad_should_skip

    cfg = TranscriptionConfig(
        vad_gate=True, vad_threshold=0.5, vad_min_speech_ratio=0.1
    )
    session = cast("InferenceSession", _StubVad(0.9))
    assert await vad_should_skip(session, _sound_16k_int16(), cfg, "") is False


async def test_vad_exception_passes_through_ungated():
    from vspeech.worker.transcription import vad_should_skip

    class _Boom:
        def run(self, *args):
            raise RuntimeError("boom")

    cfg = TranscriptionConfig(vad_gate=True)
    session = cast("InferenceSession", _Boom())
    assert await vad_should_skip(session, _sound_16k_int16(), cfg, "") is False


async def test_vad_skip_records_transc_skip_telemetry():
    from vspeech.lib.telemetry import telemetry
    from vspeech.worker.transcription import vad_should_skip

    cfg = TranscriptionConfig(
        vad_gate=True, vad_threshold=0.5, vad_min_speech_ratio=0.1
    )
    session = cast("InferenceSession", _StubVad(0.1))
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    try:
        assert await vad_should_skip(session, _sound_16k_int16(), cfg, "t") is True
        summary = telemetry.summary()
        assert "transc_skip" in summary
        assert summary["transc_skip"]["count"] == 1.0
    finally:
        telemetry.reset()
        telemetry.configure(enabled=False, max_samples=5000)


async def test_vad_gate_failure_fails_open_and_dedups_warning():
    from vspeech.worker import transcription as tx
    from vspeech.worker.transcription import vad_should_skip

    class _Boom:
        def run(self, *args):
            raise RuntimeError("boom")

    cfg = TranscriptionConfig(vad_gate=True)
    session = cast("InferenceSession", _Boom())
    tx._vad_gate_warned.discard("RuntimeError")
    try:
        # First failure fails open (True never returned) and remembers the error
        # type; a second identical failure still fails open without re-warning.
        assert await vad_should_skip(session, _sound_16k_int16(), cfg, "") is False
        assert "RuntimeError" in tx._vad_gate_warned
        assert await vad_should_skip(session, _sound_16k_int16(), cfg, "") is False
    finally:
        tx._vad_gate_warned.discard("RuntimeError")
