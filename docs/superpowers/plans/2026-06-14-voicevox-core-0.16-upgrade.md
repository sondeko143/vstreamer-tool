# voicevox-core 0.16.4 アップグレード Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** voicevox-core を pydantic 非依存の 0.16.4 へ上げ、pydantic v2 化のブロッカーを除去する（pydantic 本体は v1 維持）。

**Architecture:** モノリシックな `VoicevoxCore`（0.14）を、0.16 の `voicevox_core.blocking`（`Onnxruntime`/`OpenJtalk`/`Synthesizer`/`VoiceModelFile`）に書き換える。実行時資産（onnxruntime/辞書/.vvm モデル）は wheel 非同梱になるため config で各パスを指す。テストはワーカーをモックで完全自動検証する層と、実資産を使う E2E を専用 pytest マーカーで分離する層の2段構え。

**Tech Stack:** Python 3.11, uv, pydantic v1, voicevox-core 0.16.4, pytest（asyncio_mode=auto）, ruff, ty。

参照スペック: [docs/superpowers/specs/2026-06-14-voicevox-core-0.16-upgrade-design.md](../specs/2026-06-14-voicevox-core-0.16-upgrade-design.md)

作業ブランチ: `voicevox-core-0.16-upgrade`（作成済み）。

---

## ファイル構成

- Modify: `pyproject.toml` — voicevox-core wheel URL 更新、pytest marker/addopts 追加。
- Generated: `uv.lock`, `requirements-pod.txt` — 再生成。
- Modify: `vspeech/config.py` — `VoicevoxConfig` に `model_dir` / `onnxruntime_path` 追加、`openjtalk_dir` 既定更新。
- Rewrite: `vspeech/lib/voicevox.py` — 0.16 blocking API ラッパー。
- Modify: `vspeech/worker/tts.py` — `voicevox_worker` の `Voicevox(...)` コンストラクタ引数。
- Modify: `gui/gui.py` — voicevox タブに `model_dir` / `onnxruntime_path` 入力欄。
- Modify: `config.toml.example`, `CLAUDE.md`, `Makefile` — ドキュメント／資産取得ターゲット。
- Create: `tests/test_voicevox_config.py` — config フィールドのテスト。
- Create: `tests/test_voicevox_lib.py` — lib のラッパーロジック（fake voicevox_core 注入）。
- Create: `tests/test_tts_worker.py` — voicevox_worker のオーケストレーション（fake `vspeech.lib.voicevox`）。
- Create: `tests/test_voicevox_e2e.py` — 実資産 E2E（`voicevox_e2e` マーカー、資産が無ければ skip）。

---

## Task 1: voicevox-core を 0.16.4 へ bump し、ブロッカー解消を確認

**Files:**
- Modify: `pyproject.toml`（`[tool.uv.sources]` の voicevox-core）
- Generated: `uv.lock`, `requirements-pod.txt`

- [ ] **Step 1: pyproject.toml の wheel URL を変更**

`pyproject.toml` の `[tool.uv.sources]` 内、次の行を置換する。

旧:
```toml
voicevox-core = { url = "https://github.com/VOICEVOX/voicevox_core/releases/download/0.14.3/voicevox_core-0.14.3+cuda-cp38-abi3-win_amd64.whl" }
```
新:
```toml
voicevox-core = { url = "https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-win_amd64.whl" }
```

`[project.optional-dependencies]` の `voicevox = ["voicevox-core ; sys_platform == 'win32'"]` は**変更しない**（win32 マーカー維持）。

- [ ] **Step 2: ロックを再生成**

Run: `uv lock`
Expected: 成功して `uv.lock` が更新される。エラー（解決不能）にならないこと。

- [ ] **Step 3: ブロッカー解消を検証**

Run: `grep -n -A8 'name = "voicevox-core"' uv.lock`
Expected:
- `voicevox-core` のバージョンが `0.16.4`。
- その `dependencies` / `requires-dist` ブロックに **`pydantic` の行が無い**（0.16 は pydantic 非依存）。

Run: `grep -n -A2 'name = "pydantic"' uv.lock`
Expected: `pydantic` 本体は引き続き `1.10.x`（`>=1.10.7,<2` で解決）。voicevox-core から pydantic への依存が消えていること。

- [ ] **Step 4: 依存を同期（Windows 開発環境）**

Run: `uv sync --extra voicevox`
Expected: 成功。voicevox-core 0.16.4 が入る。
（注: 非 Windows 環境では voicevox-core は win32 マーカーで除外されるため入らない。その場合は `uv sync` のみでよい。）

- [ ] **Step 5: Docker 用 requirements を再生成**

Run: `make`
Expected: `requirements-pod.txt` が更新される。`grep -n 'pydantic' requirements-pod.txt` で pydantic が 1.10.x のままであること（Linux pod は voicevox 非搭載のまま）。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock requirements-pod.txt
git commit -m "chore: voicevox-core を 0.16.4 へ更新 (pydantic 依存を除去)"
```

---

## Task 2: VoicevoxConfig に model_dir / onnxruntime_path を追加

**Files:**
- Modify: `vspeech/config.py`（`VoicevoxConfig`, 約 298-301 行）
- Test: `tests/test_voicevox_config.py`

- [ ] **Step 1: 失敗するテストを書く**

Create `tests/test_voicevox_config.py`:
```python
from pathlib import Path

from vspeech.config import VoicevoxConfig


def test_voicevox_config_new_fields_defaults():
    cfg = VoicevoxConfig()
    assert cfg.model_dir == Path("./voicevox/models/vvms")
    assert cfg.onnxruntime_path is None


def test_voicevox_config_parse_obj_roundtrip():
    cfg = VoicevoxConfig.parse_obj(
        {
            "speaker_id": 3,
            "openjtalk_dir": "d",
            "model_dir": "m",
            "onnxruntime_path": "o.dll",
        }
    )
    assert cfg.speaker_id == 3
    assert cfg.model_dir == Path("m")
    assert cfg.onnxruntime_path == Path("o.dll")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `uv run pytest tests/test_voicevox_config.py -v`
Expected: FAIL（`AttributeError` または `model_dir` が存在しない）。

- [ ] **Step 3: VoicevoxConfig を更新**

`vspeech/config.py` の `VoicevoxConfig` を次に置換する。

旧:
```python
class VoicevoxConfig(BaseModel):
    speaker_id: int = 1
    params: VoicevoxParam = Field(default_factory=VoicevoxParam)
    openjtalk_dir: Path = Path("./voicevox_core/open_jtalk_dic_utf_8-1.11")
```
新:
```python
class VoicevoxConfig(BaseModel):
    speaker_id: int = 1
    params: VoicevoxParam = Field(default_factory=VoicevoxParam)
    openjtalk_dir: Path = Path("./voicevox/dict/open_jtalk_dic_utf_8-1.11")
    model_dir: Path = Field(default=Path("./voicevox/models/vvms"))
    onnxruntime_path: Path | None = Field(
        default=None,
        description="voicevox_onnxruntime ライブラリの実パス (onnxruntime-gpu とは別物)",
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `uv run pytest tests/test_voicevox_config.py -v`
Expected: PASS（2 件）。

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py tests/test_voicevox_config.py
git commit -m "feat: VoicevoxConfig に model_dir/onnxruntime_path を追加"
```

---

## Task 3: lib/voicevox.py を 0.16 blocking API に書き換え（ロジックを fake で検証）

**Files:**
- Rewrite: `vspeech/lib/voicevox.py`
- Test: `tests/test_voicevox_lib.py`

このモジュールはネイティブ `voicevox_core` を import する。テストでは `voicevox_core` と
`voicevox_core.blocking` を fake モジュールとして `sys.modules` に注入し、我々のラッパーロジック
（style 索引構築・遅延ロード・パラメータ設定）を実資産なしで検証する。

- [ ] **Step 1: 失敗するテストを書く**

Create `tests/test_voicevox_lib.py`:
```python
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

    def __exit__(self, *exc):
        return False


# vvm ファイル名 -> metas
METAS_BY_NAME: dict[str, list] = {}


class _FakeVoiceModelFile:
    @staticmethod
    def open(path):
        return _FakeModel(METAS_BY_NAME[Path(path).name])


class _FakeSynthesizer:
    def __init__(self, *args, **kwargs):
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
    def load_once(*args, **kwargs):
        return object()


class _FakeOpenJtalk:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def voicevox_module(monkeypatch):
    core = types.ModuleType("voicevox_core")
    core.AccelerationMode = types.SimpleNamespace(AUTO="AUTO")
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `uv run pytest tests/test_voicevox_lib.py -v`
Expected: FAIL（現行の `Voicevox` は 0.14 API で、`Voicevox(tmp_path, tmp_path)` の 2 引数や
`synthesizer` 属性を持たないため）。

- [ ] **Step 3: lib/voicevox.py を全面書き換え**

Replace the entire contents of `vspeech/lib/voicevox.py` with:
```python
from pathlib import Path

from voicevox_core import AccelerationMode
from voicevox_core.blocking import Onnxruntime
from voicevox_core.blocking import OpenJtalk
from voicevox_core.blocking import Synthesizer
from voicevox_core.blocking import VoiceModelFile

from vspeech.config import VoicevoxParam


class Voicevox:
    def __init__(
        self,
        open_jtalk_dict_dir: Path,
        model_dir: Path,
        onnxruntime_path: Path | None = None,
    ) -> None:
        if onnxruntime_path is not None:
            onnxruntime = Onnxruntime.load_once(
                filename=str(onnxruntime_path.expanduser())
            )
        else:
            onnxruntime = Onnxruntime.load_once()
        self.synthesizer = Synthesizer(
            onnxruntime,
            OpenJtalk(str(open_jtalk_dict_dir.expanduser())),
            acceleration_mode=AccelerationMode.AUTO,
        )
        self.model_dir = model_dir.expanduser()
        self._style_index: dict[int, Path] = self._build_style_index(self.model_dir)
        self._loaded: set[int] = set()

    @staticmethod
    def _build_style_index(model_dir: Path) -> dict[int, Path]:
        index: dict[int, Path] = {}
        for vvm_path in sorted(model_dir.glob("*.vvm")):
            with VoiceModelFile.open(str(vvm_path)) as model:
                for character in model.metas:
                    for style in character.styles:
                        index[int(style.id)] = vvm_path
        return index

    def load_model(self, style_id: int) -> None:
        if style_id in self._loaded:
            return
        vvm_path = self._style_index.get(style_id)
        if vvm_path is None:
            raise ValueError(f"no voice model found for style_id={style_id}")
        try:
            with VoiceModelFile.open(str(vvm_path)) as model:
                self.synthesizer.load_voice_model(model)
        except Exception as e:
            raise ValueError(e)
        self._loaded.add(style_id)

    def is_model_loaded(self, style_id: int) -> bool:
        return style_id in self._loaded

    def voicevox_tts(self, text: str, speaker_id: int, params: VoicevoxParam) -> bytes:
        try:
            audio_query = self.synthesizer.create_audio_query(text, speaker_id)
            for key, value in params:
                setattr(audio_query, key, value)
            return self.synthesizer.synthesis(audio_query, speaker_id)
        except Exception as e:
            raise ValueError(e)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `uv run pytest tests/test_voicevox_lib.py -v`
Expected: PASS（3 件）。

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/voicevox.py tests/test_voicevox_lib.py
git commit -m "feat: lib/voicevox を 0.16 blocking API へ書き換え"
```

---

## Task 4: voicevox_worker を新コンストラクタへ更新（モックでオーケストレーション検証）

**Files:**
- Modify: `vspeech/worker/tts.py`（`voicevox_worker`, 約 48-51 行）
- Test: `tests/test_tts_worker.py`

ワーカーは `voicevox_worker` 内で `from vspeech.lib.voicevox import Voicevox` を遅延 import する。
テストでは `vspeech.lib.voicevox` モジュール自体を fake に差し替えるため、`voicevox_core` 未インストール
環境でも動作する（CI 可）。

- [ ] **Step 1: 失敗するテストを書く**

Create `tests/test_tts_worker.py`:
```python
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
    instances: list["FakeVoicevox"] = []

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
    module.Voicevox = FakeVoicevox
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
    return await wait_for(
        anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 10
    )


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
    module.Voicevox = RaiseOnBad
    monkeypatch.setitem(sys.modules, "vspeech.lib.voicevox", module)

    cfg = VoicevoxConfig(speaker_id=2)
    queue = Queue()
    await _put(queue, "bad", Params())
    await _put(queue, "good", Params())
    output = await wait_for(
        anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 10
    )
    assert output.text == "good"
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `uv run pytest tests/test_tts_worker.py -v`
Expected: `test_worker_constructs_with_config_paths` が FAIL（現行ワーカーは
`Voicevox(vvox_config.openjtalk_dir)` の 1 引数のみで、`init_args` が
`(openjtalk_dir, None, None)` になる）。他のテストは PASS（既存挙動の回帰ガード）。

- [ ] **Step 3: voicevox_worker のコンストラクタ呼び出しを更新**

`vspeech/worker/tts.py` の `voicevox_worker` 冒頭、次の行を置換する。

旧:
```python
    vvox = Voicevox(vvox_config.openjtalk_dir)
```
新:
```python
    vvox = Voicevox(
        vvox_config.openjtalk_dir,
        vvox_config.model_dir,
        vvox_config.onnxruntime_path,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `uv run pytest tests/test_tts_worker.py -v`
Expected: PASS（6 件）。

- [ ] **Step 5: Commit**

```bash
git add vspeech/worker/tts.py tests/test_tts_worker.py
git commit -m "feat: voicevox_worker を model_dir/onnxruntime_path 対応に更新"
```

---

## Task 5: E2E テストのフレームワーク（pytest マーカー + 資産取得ターゲット）

**Files:**
- Modify: `pyproject.toml`（`[tool.pytest.ini_options]`）
- Modify: `Makefile`
- Test: `tests/test_voicevox_e2e.py`

- [ ] **Step 1: pytest マーカーと addopts を登録**

`pyproject.toml` の `[tool.pytest.ini_options]` を次に置換する。

旧:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = "."
asyncio_mode = "auto"
```
新:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = "."
asyncio_mode = "auto"
addopts = "-m 'not voicevox_e2e'"
markers = [
    "voicevox_e2e: VOICEVOX 実合成 E2E（実行時資産が必要。通常実行では除外）",
]
```

- [ ] **Step 2: E2E テストを作成**

Create `tests/test_voicevox_e2e.py`:
```python
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
    dic = (
        Path(dict_env)
        if dict_env
        else _first(ASSET_ROOT.glob("**/open_jtalk_dic_*"))
    )
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
    output = await wait_for(
        anext(voicevox_worker(vvox_config=cfg, in_queue=queue)), 60
    )
    assert output.sound is not None
    assert output.sound.rate == 24000
    assert len(output.sound.data) > 0
```

- [ ] **Step 3: 通常実行で E2E が除外＆スキップされることを確認**

Run: `uv run pytest tests/test_voicevox_e2e.py -v`
Expected: addopts により deselect される（`1 deselected` 等）。明示選択しても資産が無ければ skip:
Run: `uv run pytest -m voicevox_e2e tests/test_voicevox_e2e.py -v`
Expected: SKIPPED（reason: VOICEVOX 実資産が見つかりません）。

- [ ] **Step 4: 資産取得用の Makefile ターゲットを追加**

`Makefile` の末尾（`clean` の前）に次を追加する。`download-windows-x64.exe` は VOICEVOX 公式の
ダウンローダ。GPU を使う場合はデバイス指定が必要（フラグは `./download-voicevox.exe --help` で確認）。

```makefile
.PHONY: voicevox-assets
voicevox-assets:
	curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe -o download-voicevox.exe
	./download-voicevox.exe -o tests/assets/voicevox --exclude c-api
	@echo "GPU を使う場合は --device (例: cuda) を付けて再実行してください。フラグは ./download-voicevox.exe --help を参照。"
```

`.gitignore` に資産ディレクトリとダウンローダを無視させる（既に `*.wav` 等は無視されているが、
バイナリ資産は追跡しない）。`.gitignore` の末尾に追記:
```
/tests/assets/voicevox/
/download-voicevox.exe
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile tests/test_voicevox_e2e.py .gitignore
git commit -m "test: VOICEVOX 実合成 E2E を voicevox_e2e マーカーで追加"
```

---

## Task 6: GUI に model_dir / onnxruntime_path 入力欄を追加

**Files:**
- Modify: `gui/gui.py`（`draw_voicevox_tab`, 約 867-898 行）

GUI には自動テストが無い（Tkinter）。検証は `ruff` / `ty` と、必要なら `gui` extra での手動起動。

- [ ] **Step 1: draw_voicevox_tab を更新**

`gui/gui.py` の `draw_voicevox_tab` 内、`openjtalk_dir` の `draw_tb` から `speaker_id` の `draw_sb` までを
次に置換する。

旧:
```python
        self.draw_tb(tab_frame, config_name=f"{prefix}.openjtalk_dir").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.speaker_id",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=0, row=1, sticky=EW)
        params = list(get_type_hints(VoicevoxParam).keys())
        chunked_list: List[List[str]] = list()
        for i in range(0, len(params), max_columns):
            chunked_list.append(params[i : i + max_columns])
        for row, params_chunk in enumerate(chunked_list):
            for column, param_name in enumerate(params_chunk):
                self.draw_sb(
                    frame=tab_frame,
                    config_name=f"{prefix}.params.{param_name}",
                    from_=-1.0,
                    to=2.0,
                    increment=0.01,
                ).grid(column=column, row=row + 2, sticky=EW)
```
新:
```python
        self.draw_tb(tab_frame, config_name=f"{prefix}.openjtalk_dir").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.model_dir").grid(
            column=0, row=1, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.onnxruntime_path").grid(
            column=0, row=2, columnspan=max_columns, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.speaker_id",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=0, row=3, sticky=EW)
        params = list(get_type_hints(VoicevoxParam).keys())
        chunked_list: List[List[str]] = list()
        for i in range(0, len(params), max_columns):
            chunked_list.append(params[i : i + max_columns])
        for row, params_chunk in enumerate(chunked_list):
            for column, param_name in enumerate(params_chunk):
                self.draw_sb(
                    frame=tab_frame,
                    config_name=f"{prefix}.params.{param_name}",
                    from_=-1.0,
                    to=2.0,
                    increment=0.01,
                ).grid(column=column, row=row + 4, sticky=EW)
```

- [ ] **Step 2: Lint / 型チェック**

Run: `uv run ruff check gui/gui.py && uv run ty check`
Expected: エラーなし。

- [ ] **Step 3: Commit**

```bash
git add gui/gui.py
git commit -m "feat(gui): voicevox タブに model_dir/onnxruntime_path を追加"
```

---

## Task 7: ドキュメント更新（config.toml.example / CLAUDE.md）

**Files:**
- Modify: `config.toml.example`（`[voicevox]` セクション, 98-100 行）
- Modify: `CLAUDE.md`（Platform constraints の項）

- [ ] **Step 1: config.toml.example を更新**

`config.toml.example` の `[voicevox]` セクションを次に置換する。

旧:
```toml
[voicevox]
speaker_id = 2
openjtalk_dir = "./open_jtalk_dic_utf_8-1.11"
```
新:
```toml
[voicevox]
speaker_id = 2
# 0.16 では onnxruntime / 辞書 / .vvm モデルは wheel 同梱ではない。
# 公式ダウンローダ (make voicevox-assets) で取得し、各パスを指定する。
openjtalk_dir = "./voicevox/dict/open_jtalk_dic_utf_8-1.11"
model_dir = "./voicevox/models/vvms"
# voicevox 専用ビルドの onnxruntime。whisper/rvc の onnxruntime-gpu とは別物。
# 明示しないと誤った onnxruntime DLL を読み込む恐れがある。
onnxruntime_path = "./voicevox/onnxruntime/lib/voicevox_onnxruntime.dll"
```

- [ ] **Step 2: CLAUDE.md の Platform constraints を更新**

`CLAUDE.md` の Platform constraints 文中、`voicevox-core` に関する記述に次の一文を追記する
（`[tool.uv.sources]` を説明している文の直後）。

追記する文:
```markdown
`voicevox-core` is pinned to **0.16.4** (`cp310-abi3`, pydantic-free). Its ONNX Runtime, OpenJTalk dictionary, and `.vvm` voice models are **not** bundled in the wheel — fetch them with the VOICEVOX downloader (`make voicevox-assets`) and point `voicevox.openjtalk_dir` / `model_dir` / `onnxruntime_path` at them. VOICEVOX uses its own `voicevox_onnxruntime` build, distinct from the `onnxruntime-gpu` used by whisper/rvc; set `onnxruntime_path` explicitly so the correct DLL is loaded.
```

- [ ] **Step 3: Commit**

```bash
git add config.toml.example CLAUDE.md
git commit -m "docs: voicevox 0.16 の資産取得とパス設定を記載"
```

---

## Task 8: フル検証

**Files:** なし（検証のみ）

- [ ] **Step 1: フォーマット & lint**

Run: `uv run ruff format . && uv run ruff check .`
Expected: 変更なし or 自動修正済み、エラーなし。

- [ ] **Step 2: 型チェック**

Run: `uv run ty check`
Expected: エラーなし。

- [ ] **Step 3: テスト（通常実行 = E2E 除外）**

Run: `uv run pytest -v`
Expected: 全 PASS（`test_voicevox_config` / `test_voicevox_lib` / `test_tts_worker` を含む）、
`test_voicevox_e2e` は deselected。既存テスト（`test_event_chains` / `test_worker_input` /
`test_transcription_worker`）も緑のまま。

- [ ] **Step 4: ブロッカー解消の最終確認**

Run: `grep -n -A8 'name = "voicevox-core"' uv.lock`
Expected: 0.16.4、pydantic 依存なし。これで pydantic v2 化の前提が整ったことを記録する。

- [ ] **Step 5: （ユーザ環境）実合成 E2E**

Run（資産取得後）: `make voicevox-assets` → `uv run pytest -m voicevox_e2e -v`
Expected: PASS（実 WAV が生成される）。
注: このステップは Windows + GPU + 実資産が必要で、実装担当エージェントの環境では実行不可。
ユーザが手元で実施する。

---

## 完了条件

- `uv.lock` で voicevox-core が 0.16.4、pydantic 依存が消えている。
- `uv run pytest`（通常実行）が全緑。
- `ruff` / `ty` が緑。
- 実合成 E2E はユーザ環境で `uv run pytest -m voicevox_e2e` により確認可能。
