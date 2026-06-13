import os
from asyncio import Queue
from asyncio import wait_for
from pathlib import Path
from uuid import uuid4

import pytest

from vspeech.config import VoicevoxConfig
from vspeech.shared_context import EventAddress
from vspeech.shared_context import EventType
from vspeech.shared_context import Params
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.tts import voicevox_worker

ASSET_ROOT = Path(os.environ.get("VSPEECH_VVOX_ASSETS", "tests/assets/voicevox"))


def _first(glob_iter):
    return next(iter(sorted(glob_iter)), None)


def _resolve_assets():
    ort_env = os.environ.get("VSPEECH_VVOX_ONNXRUNTIME")
    dict_env = os.environ.get("VSPEECH_VVOX_DICT")
    models_env = os.environ.get("VSPEECH_VVOX_MODEL_DIR")
    ort = (
        Path(ort_env)
        if ort_env
        else _first(ASSET_ROOT.glob("**/voicevox_onnxruntime*"))
    )
    dic = Path(dict_env) if dict_env else _first(ASSET_ROOT.glob("**/open_jtalk_dic_*"))
    models = Path(models_env) if models_env else (ASSET_ROOT / "models" / "vvms")
    return ort, dic, models


_ORT, _DICT, _MODELS = _resolve_assets()
_ASSETS_READY = bool(
    _ORT
    and _ORT.exists()
    and _DICT
    and _DICT.exists()
    and _MODELS.exists()
    and any(_MODELS.glob("*.vvm"))
)

pytestmark = [
    pytest.mark.voicevox_e2e,
    pytest.mark.skipif(
        not _ASSETS_READY,
        reason="VOICEVOX 実資産が見つかりません (make voicevox-assets で取得)",
    ),
]


async def test_voicevox_e2e_synthesis():
    # 実資産がある環境でのみ実行。voicevox_core が import される。
    from vspeech.lib.voicevox import Voicevox

    style_index = Voicevox._build_style_index(_MODELS.expanduser())
    assert style_index, "vvm から style が見つかりません"
    style_id = sorted(style_index)[0]

    cfg = VoicevoxConfig(
        speaker_id=style_id,
        openjtalk_dir=_DICT,
        model_dir=_MODELS,
        onnxruntime_path=_ORT,
    )
    queue = Queue()
    await queue.put(
        WorkerInput(
            input_id=uuid4(),
            current_event=EventAddress(EventType.tts, params=Params()),
            following_events=[],
            text="テストです",
            sound=SoundInput.invalid(),
            file_path="",
            filters=[],
        )
    )
    output = await wait_for(anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 60)
    assert output.sound is not None
    assert output.sound.rate == 24000
    assert len(output.sound.data) > 0
