import sys
import types
from asyncio import Queue
from asyncio import wait_for
from pathlib import Path
from uuid import uuid4

import pytest

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import VoicevoxConfig
from vspeech.config import VoicevoxParam
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.tts import voicevox_worker


class FakeVoicevox:
    instances: list[FakeVoicevox] = []

    def __init__(self, open_jtalk_dict_dir=None, model_dir=None, onnxruntime_path=None):
        self.init_args = (open_jtalk_dict_dir, model_dir, onnxruntime_path)
        self.load_calls: list[int] = []
        self.tts_calls: list[dict] = []
        FakeVoicevox.instances.append(self)

    def load_model(self, style_id):
        self.load_calls.append(style_id)

    def is_model_loaded(self, style_id):
        return style_id in self.load_calls

    def voicevox_tts(self, text, speaker_id, params):
        self.tts_calls.append(
            {"text": text, "speaker_id": speaker_id, "params": params}
        )
        return b"RIFF" + b"\x00" * 40 + b"PCMDATA"


@pytest.fixture
def fake_voicevox(monkeypatch):
    FakeVoicevox.instances = []
    module = types.ModuleType("vspeech.lib.voicevox")
    setattr(module, "Voicevox", FakeVoicevox)
    monkeypatch.setitem(sys.modules, "vspeech.lib.voicevox", module)
    return FakeVoicevox


async def _put(queue, text, params):
    await queue.put(
        WorkerInput(
            input_id=uuid4(),
            current_event=EventAddress(EventType.tts, params=params),
            following_events=[],
            text=text,
            sound=SoundInput.invalid(),
            file_path="",
            filters=[],
        )
    )


async def _run_one(cfg, queue):
    return await wait_for(anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 10)


async def test_worker_constructs_with_config_paths(fake_voicevox):
    cfg = VoicevoxConfig(
        speaker_id=2, model_dir=Path("M"), onnxruntime_path=Path("ORT")
    )
    queue = Queue()
    await _put(queue, "テスト", Params())
    await _run_one(cfg, queue)
    inst = fake_voicevox.instances[0]
    assert inst.init_args == (cfg.openjtalk_dir, cfg.model_dir, cfg.onnxruntime_path)


async def test_worker_route_speaker_id_overrides_config(fake_voicevox):
    cfg = VoicevoxConfig(speaker_id=2)
    queue = Queue()
    await _put(queue, "テスト", Params(speaker_id=5))
    await _run_one(cfg, queue)
    inst = fake_voicevox.instances[0]
    assert inst.load_calls == [5]
    assert inst.tts_calls[0]["speaker_id"] == 5


async def test_worker_uses_config_speaker_when_no_param(fake_voicevox):
    cfg = VoicevoxConfig(speaker_id=2)
    queue = Queue()
    await _put(queue, "テスト", Params())
    await _run_one(cfg, queue)
    inst = fake_voicevox.instances[0]
    assert inst.tts_calls[0]["speaker_id"] == 2


async def test_worker_route_params_override(fake_voicevox):
    cfg = VoicevoxConfig(
        speaker_id=2, params=VoicevoxParam(speed_scale=1.0, pitch_scale=0.0)
    )
    queue = Queue()
    await _put(queue, "テスト", Params(speed=1.7, pitch=0.4))
    await _run_one(cfg, queue)
    sent = fake_voicevox.instances[0].tts_calls[0]["params"]
    assert sent.speed_scale == 1.7
    assert sent.pitch_scale == 0.4


async def test_worker_output_sound_shape(fake_voicevox):
    cfg = VoicevoxConfig(speaker_id=2)
    queue = Queue()
    await _put(queue, "テスト", Params())
    output = await _run_one(cfg, queue)
    assert output.sound is not None
    assert output.sound.rate == 24000
    assert output.sound.format == SampleFormat.INT16
    assert output.sound.channels == 1
    assert output.sound.data == b"PCMDATA"
    assert output.text == "テスト"


async def test_worker_swallows_value_error(monkeypatch):
    class RaiseOnBad(FakeVoicevox):
        def voicevox_tts(self, text, speaker_id, params):
            if text == "bad":
                raise ValueError("boom")
            return super().voicevox_tts(text, speaker_id, params)

    RaiseOnBad.instances = []
    module = types.ModuleType("vspeech.lib.voicevox")
    setattr(module, "Voicevox", RaiseOnBad)
    monkeypatch.setitem(sys.modules, "vspeech.lib.voicevox", module)

    cfg = VoicevoxConfig(speaker_id=2)
    queue = Queue()
    await _put(queue, "bad", Params())
    await _put(queue, "good", Params())
    output = await wait_for(anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 10)
    assert output.text == "good"
