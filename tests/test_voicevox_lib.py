import importlib
import sys
import types
from pathlib import Path

import pytest

from vspeech.config import VoicevoxParam


class _FakeAudioQuery:
    def __init__(self):
        self.speed_scale = 0.0
        self.pitch_scale = 0.0
        self.intonation_scale = 0.0
        self.volume_scale = 0.0
        self.pre_phoneme_length = 0.0
        self.post_phoneme_length = 0.0


class _FakeStyle:
    def __init__(self, id):
        self.id = id


class _FakeCharacter:
    def __init__(self, styles):
        self.styles = styles


class _FakeModel:
    def __init__(self, metas):
        self.metas = metas

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


# vvm ファイル名 -> metas
METAS_BY_NAME: dict[str, list] = {}


class _FakeVoiceModelFile:
    @staticmethod
    def open(path):
        return _FakeModel(METAS_BY_NAME[Path(path).name])


class _FakeSynthesizer:
    def __init__(self, *_args, **_kwargs):
        self.loaded = []
        self.last_audio_query = None
        self.last_synth_style_id = None

    def load_voice_model(self, model):
        self.loaded.append(model)

    def create_audio_query(self, text, style_id):
        self.last_text = text
        self.last_query_style_id = style_id
        return _FakeAudioQuery()

    def synthesis(self, audio_query, style_id):
        self.last_audio_query = audio_query
        self.last_synth_style_id = style_id
        return b"RIFF" + b"\x00" * 40 + b"WAVEPAYLOAD"


class _FakeOnnxruntime:
    @staticmethod
    def load_once(*_args, **_kwargs):
        return object()


class _FakeOpenJtalk:
    def __init__(self, *_args, **_kwargs):
        pass


@pytest.fixture
def voicevox_module(monkeypatch):
    core = types.ModuleType("voicevox_core")
    blocking = types.ModuleType("voicevox_core.blocking")
    blocking.Onnxruntime = _FakeOnnxruntime
    blocking.OpenJtalk = _FakeOpenJtalk
    blocking.Synthesizer = _FakeSynthesizer
    blocking.VoiceModelFile = _FakeVoiceModelFile
    monkeypatch.setitem(sys.modules, "voicevox_core", core)
    monkeypatch.setitem(sys.modules, "voicevox_core.blocking", blocking)
    monkeypatch.delitem(sys.modules, "vspeech.lib.voicevox", raising=False)
    module = importlib.import_module("vspeech.lib.voicevox")
    yield module
    monkeypatch.delitem(sys.modules, "vspeech.lib.voicevox", raising=False)


def _make_models(tmp_path: Path):
    METAS_BY_NAME.clear()
    (tmp_path / "0.vvm").write_bytes(b"")
    (tmp_path / "1.vvm").write_bytes(b"")
    METAS_BY_NAME["0.vvm"] = [_FakeCharacter([_FakeStyle(3)])]
    METAS_BY_NAME["1.vvm"] = [_FakeCharacter([_FakeStyle(7)])]


def test_lazy_load_loads_correct_model_once(voicevox_module, tmp_path):
    _make_models(tmp_path)
    vvox = voicevox_module.Voicevox(tmp_path, tmp_path)
    assert not vvox.is_model_loaded(3)
    vvox.load_model(3)
    assert vvox.is_model_loaded(3)
    assert len(vvox.synthesizer.loaded) == 1
    vvox.load_model(3)  # 2 回目は再ロードしない
    assert len(vvox.synthesizer.loaded) == 1


def test_unknown_style_raises(voicevox_module, tmp_path):
    _make_models(tmp_path)
    vvox = voicevox_module.Voicevox(tmp_path, tmp_path)
    with pytest.raises(ValueError):
        vvox.load_model(999)


def test_tts_sets_params_and_returns_bytes(voicevox_module, tmp_path):
    _make_models(tmp_path)
    vvox = voicevox_module.Voicevox(tmp_path, tmp_path)
    params = VoicevoxParam(speed_scale=1.5, pitch_scale=0.3)
    wav = vvox.voicevox_tts("こんにちは", 3, params)
    assert wav.startswith(b"RIFF")
    assert vvox.synthesizer.last_audio_query.speed_scale == 1.5
    assert vvox.synthesizer.last_audio_query.pitch_scale == 0.3
    assert vvox.synthesizer.last_synth_style_id == 3
