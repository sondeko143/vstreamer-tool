# Pydantic v1 → v2 移行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vspeech / gui を Pydantic v1 から v2 ネイティブ API へ全面移行する。

**Architecture:** 依存を `pydantic>=2` + `pydantic-settings` に上げ、v1 API（`BaseSettings`/`root_validator`/`parse_obj`/`.dict()`/`.json()`/`orm_mode`/`from_orm`/`allow_population_by_field_name`/`Field(env=)`/`json_encoders`）を v2 API（`SettingsConfigDict`/`model_validator`/`model_validate`/`model_dump`/`model_dump_json`/`from_attributes`/`populate_by_name`/`AliasChoices`/`ConfigDict(json_encoders=)`）へ置換。pydantic は単一バージョンしか入れられないため、依存 bump 後はコード変換が全て終わるまでテストスイートは赤になる（移行ブランチ上の想定挙動）。

**Tech Stack:** Python 3.11, uv, pydantic v2, pydantic-settings v2, pytest（asyncio_mode=auto）, ruff, ty。

参照スペック: [docs/superpowers/specs/2026-06-14-pydantic-v2-migration-design.md](../specs/2026-06-14-pydantic-v2-migration-design.md)

作業ブランチ: `pydantic-v2-migration`（作成済み、spec コミット済み `5cad8af`）。

---

## 重要な前提（全タスク共通）

- **pydantic は単一バージョン**。Task 1 で v2 を入れた瞬間、未変換ファイルの v1 API は全て壊れる。よって Task 2〜5 は「該当モジュールが import でき／対象テストが通る」ことを個別検証とし、**フルスイート緑は Task 6** で達成する。中間コミットがスイート赤なのは織り込み済み（ブランチは Task 6 まで未マージ）。
- 変換順序の依存: `config.py` ← `shared_context.py` ← `audio.py`/`ami.py`/`tts.py`/`gui.py` ← `tests`。この順で変換する。
- `voicevox_core` / `pyaudio` / `ttkbootstrap` は base venv に未インストールの optional extra。よって `vspeech.lib.voicevox` / `vspeech.lib.audio` / `gui.gui` は base 環境では import できない（pydantic とは無関係）。これらは grep ベース＋可能な範囲の import で検証する。
- ty は既に ~73 件の既存診断があり green gate ではない。新規の**実害**（本物の型エラー）のみ気にする。
- `uv run` は base 環境を自動同期して実行できる（`--extra voicevox` 同期は VS Code のファイルロックで失敗するので使わない）。

---

## ファイル構成（変更対象）

- Modify: `pyproject.toml`（pydantic>=2、pydantic-settings 追加）、`uv.lock`、`requirements-pod.txt`（再生成）
- Modify: `vspeech/config.py`（BaseSettings→pydantic-settings、model_config、AliasChoices、protected_namespaces、parse_obj/dict 置換、cli= 削除）
- Modify: `vspeech/shared_context.py`（Params/SoundInput の model_config、root_validator→model_validator、from_orm→model_validate、filters→list[str]）
- Modify: `vspeech/lib/audio.py`、`vspeech/lib/ami.py`（alias_generator の model_config、parse_obj→model_validate）
- Modify: `vspeech/worker/tts.py`（dict→model_dump）、`gui/gui.py`（json→model_dump_json）
- Modify: `tests/test_voicevox_config.py`（parse_obj→model_validate）
- Create: `tests/test_config_secret.py`（SecretStr round-trip）
- Modify: `CLAUDE.md`（v2 前提へ。今回からリポジトリ追跡）

---

## Task 1: 依存を pydantic v2 + pydantic-settings へ

**Files:** `pyproject.toml`, `uv.lock`, `requirements-pod.txt`

- [ ] **Step 1: pyproject.toml の依存を変更**

`[project] dependencies` の `"pydantic>=1.10.7,<2",` を次の2行に置換:
```toml
    "pydantic>=2,<3",
    "pydantic-settings>=2,<3",
```

- [ ] **Step 2: ロック再生成**

Run: `uv lock`
Expected: 成功。`grep -nE 'name = "pydantic(-settings)?"' uv.lock` で `pydantic` が `2.x`、`pydantic-settings` が `2.x` で解決されること。

- [ ] **Step 3: pod requirements 再生成**

Run: `make` （無ければ `uv export --no-default-groups --extra voicevox --no-hashes --no-emit-project -o requirements-pod.txt`）
Expected: `requirements-pod.txt` に `pydantic==2.*` と `pydantic-settings==2.*` が現れる。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock requirements-pod.txt
git commit -m "chore: pydantic を v2 系へ更新し pydantic-settings を追加"
```

注: この時点で `uv run pytest` は赤（v1 コードが v2 上で動かない）。Task 6 まで赤のまま進む。

---

## Task 2: vspeech/config.py を v2 へ

**Files:** `vspeech/config.py`

- [ ] **Step 1: import を更新**

先頭の import 群を次のように変更する。
削除: `from pydantic import BaseSettings`
追加（一行ずつ、ruff force-single-line）:
```python
from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
```
`from pydantic import BaseModel` / `Field` / `SecretStr` は維持。

- [ ] **Step 2: RecordingConfig の未使用 `cli=` kwarg を削除**

`RecordingConfig` の4フィールドから `cli=(...)` 引数のみ削除（`default=`/`description=` は残す）:
```python
    channels: int = Field(default=1, description="recording channels")
    interval_sec: float = Field(default=0.1, description="recording interval sec.")
    silence_threshold: int = Field(
        default=-40,
        description="voice detection volume percentage (approx)",
    )
    max_recording_sec: float = Field(
        default=5,
        description="max wav file length to process",
    )
```

- [ ] **Step 3: protected_namespaces を付与**

`model_` 接頭辞フィールドを持つ2モデルに `model_config` を追加。
`RvcConfig` のクラス本体先頭（`model_file:` の直前）に:
```python
    model_config = ConfigDict(protected_namespaces=())
```
`VoicevoxConfig` のクラス本体先頭（`speaker_id:` の直前）に:
```python
    model_config = ConfigDict(protected_namespaces=())
```

- [ ] **Step 4: Config(BaseSettings) を v2 化**

(a) `listen_port` を AliasChoices に:
```python
    listen_port: int = Field(
        default=8080, validation_alias=AliasChoices("listen_port", "PORT")
    )
```
(b) ネスト `class Config(BaseSettings.Config): ...`（env_prefix/env_nested_delimiter/encode_secret/json_encoders）を削除し、代わりにフィールド群の直後へ:
```python
    model_config = SettingsConfigDict(
        env_prefix="vspeech_",
        env_nested_delimiter="__",
        json_encoders={SecretStr: lambda v: v.get_secret_value()},
    )
```
(c) `read_config_from_file` の `Config.parse_obj(config_obj)` → `Config.model_validate(config_obj)`
(d) `export_to_toml` の `self.dict()` → `self.model_dump()`

- [ ] **Step 5: 検証（import 構築 + 警告ゼロ + grep）**

Run: `uv run python -W error::UserWarning -c "from vspeech.config import Config; c = Config(); print(c.model_validate({'listen_port': 9001}).listen_port)"`
Expected: `9001` が出力され、**警告でエラーにならない**こと（protected_namespaces 対応で警告ゼロ／AliasChoices で field-name 入力が通る）。

Run (env override 確認): `PORT=7777 uv run python -c "from vspeech.config import Config; print(Config().listen_port)"`
Expected: `7777`（PORT env が効く）。もし効かない場合は AliasChoices と env_prefix の相互作用が原因なので、`AliasChoices` の順序や `populate_by_name=True` 追加で dict 入力（`listen_port`）と env（`PORT`）の両立を確認する。

Run: `uv run ruff check vspeech/config.py && uv run ruff format vspeech/config.py`
Run (残存 v1 API 無し): `grep -nE 'BaseSettings\.Config|parse_obj|\.dict\(|env=|json_encoders|cli=' vspeech/config.py`
Expected: `json_encoders` の1件（model_config 内）以外に v1 API が無いこと（`parse_obj`/`.dict(`/`env=`/`cli=`/`BaseSettings.Config` はゼロ）。

- [ ] **Step 6: Commit**

```bash
git add vspeech/config.py
git commit -m "refactor: config.py を pydantic v2 (pydantic-settings) へ移行"
```

---

## Task 3: vspeech/shared_context.py を v2 へ（コア検証ポイント）

**Files:** `vspeech/shared_context.py`

- [ ] **Step 1: import を更新**

削除: `from pydantic import root_validator`、`from typing import Iterable`（filters を list 化するため不要に）。
追加: `from pydantic import ConfigDict`、`from pydantic import model_validator`。
（`from typing import Any` / `from typing import cast` が新バリデータで未使用になる場合は削除。判断は Step 6 の ruff で確定。）

- [ ] **Step 2: Params の Config を v2 化**

`Params` の
```python
    class Config:
        allow_population_by_field_name = True
```
を
```python
    model_config = ConfigDict(populate_by_name=True)
```
に置換（クラス本体先頭に置く）。

- [ ] **Step 3: SoundInput の Config を v2 化**

`SoundInput` の
```python
    class Config:
        orm_mode = True
```
を
```python
    model_config = ConfigDict(from_attributes=True)
```
に置換。

- [ ] **Step 4: WorkerInput の filters と root_validator を v2 化**

(a) `filters: Iterable[str]` → `filters: list[str]`
(b) 壊れた validator
```python
    @classmethod
    @root_validator(pre=False, skip_on_failure=True)
    def root_validator(cls, values: dict[str, Any]):
        sound = cast(SoundInput, values.get("sound"))
        event = cast(EventType, values.get("current_event"))
        if is_sound_event(event) and sound.is_invalid():
            raise ValueError("sound input is invalid")
        if event == EventType.reload and not values.get("file_path"):
            raise ValueError("file_path is required")
        return values
```
を次に置換（`mode="after"` のインスタンスメソッド。`current_event` は `EventAddress` だが `EventAddress.__eq__` が `EventType` 比較に対応しているので `is_sound_event` / `== EventType.reload` はそのまま機能する）:
```python
    @model_validator(mode="after")
    def _validate_input(self):
        if is_sound_event(self.current_event) and self.sound.is_invalid():
            raise ValueError("sound input is invalid")
        if self.current_event == EventType.reload and not self.file_path:
            raise ValueError("file_path is required")
        return self
```

- [ ] **Step 5: from_orm を model_validate に**

`from_output` / `from_command` 内の2箇所:
```python
                sound=SoundInput.from_orm(output.sound)
```
```python
                sound=SoundInput.from_orm(command.operand.sound),
```
をそれぞれ `SoundInput.model_validate(output.sound)` / `SoundInput.model_validate(command.operand.sound)` に置換（`SoundInput` は from_attributes=True なので属性アクセスで検証できる）。

- [ ] **Step 6: 検証（コアテスト緑）**

Run: `uv run ruff check vspeech/shared_context.py && uv run ruff format vspeech/shared_context.py`
（未使用 import が残れば削除して再実行）

Run: `uv run pytest tests/test_event_chains.py tests/test_worker_input.py -v`
Expected: **全 pass**。特に `test_worker_input.py::test_from_command_sound_data_invalid` と `::test_from_command_reload_invalid` が**今回から pass に転じる**（validator が正しく発火）。

**もし** WorkerInput 構築時に `EventAddress`（stdlib dataclass）の再検証で失敗する場合のみ、`WorkerInput` クラス本体先頭に次を追加して既構築インスタンスをそのまま受け入れる:
```python
    model_config = ConfigDict(arbitrary_types_allowed=True)
```
（追加後に Step 6 のテストを再実行して緑を確認。）

- [ ] **Step 7: Commit**

```bash
git add vspeech/shared_context.py
git commit -m "refactor: shared_context.py を pydantic v2 へ移行 (validator 発火修正含む)"
```

---

## Task 4: vspeech/lib/audio.py と vspeech/lib/ami.py を v2 へ

**Files:** `vspeech/lib/audio.py`, `vspeech/lib/ami.py`

- [ ] **Step 1: audio.py の Config を v2 化**

import に `from pydantic import ConfigDict` を追加。
`HostAPIInfo` と `DeviceInfo` のそれぞれの
```python
    class Config:
        alias_generator = camelize
```
を
```python
    model_config = ConfigDict(alias_generator=camelize)
```
に置換。

- [ ] **Step 2: audio.py の parse_obj を model_validate に**

`vspeech/lib/audio.py` 内の全 `parse_obj(` → `model_validate(`（8箇所: `DeviceInfo.parse_obj(...)` / `HostAPIInfo.parse_obj(...)`）。各 PyAudio が返す dict は camelCase キーで、alias_generator により検証される。

- [ ] **Step 3: ami.py の parse_obj を model_validate に**

`vspeech/lib/ami.py` の `AmiResponse.parse_obj(res_json)` → `AmiResponse.model_validate(res_json)`。

- [ ] **Step 4: 検証**

Run: `uv run python -c "import vspeech.lib.ami"`
Expected: import 成功（httpx は base 依存）。

Run: `uv run ruff check vspeech/lib/audio.py vspeech/lib/ami.py && uv run ruff format vspeech/lib/audio.py vspeech/lib/ami.py`
Run: `grep -nE 'parse_obj|class Config:|orm_mode|alias_generator = ' vspeech/lib/audio.py vspeech/lib/ami.py`
Expected: `parse_obj` ゼロ、`class Config:`（旧式）ゼロ。
（注: `vspeech.lib.audio` は pyaudio 未インストールのため import 検証不可。grep と ami の import で代替。）

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/audio.py vspeech/lib/ami.py
git commit -m "refactor: lib/audio.py・lib/ami.py を pydantic v2 へ移行"
```

---

## Task 5: vspeech/worker/tts.py と gui/gui.py を v2 へ

**Files:** `vspeech/worker/tts.py`, `gui/gui.py`

- [ ] **Step 1: tts.py の dict を model_dump に**

`vspeech/worker/tts.py` の
```python
                **vvox_config.params.dict(exclude={"speed_scale", "pitch_scale"}),
```
を
```python
                **vvox_config.params.model_dump(exclude={"speed_scale", "pitch_scale"}),
```
に置換。

- [ ] **Step 2: gui.py の json を model_dump_json に**

`gui/gui.py` の3箇所の `self.config.json()` → `self.config.model_dump_json()`
（`file.write(bytes(self.config.model_dump_json(), encoding="utf-8"))` / `temp_config_file.write(self.config.model_dump_json())` ×2）。

- [ ] **Step 3: 検証**

Run: `uv run pytest tests/test_tts_worker.py -q`
Expected: **6 pass**（tts ワーカーのモックテスト。tts.py の model_dump 変換が効く）。

Run: `uv run ruff check vspeech/worker/tts.py gui/gui.py && uv run ruff format vspeech/worker/tts.py gui/gui.py`
Run: `grep -nE '\.dict\(|\.json\(\)' vspeech/worker/tts.py gui/gui.py`
Expected: `.dict(` / `.json()`（pydantic 由来）ゼロ。
（注: `gui.gui` は ttkbootstrap 未インストールのため import 検証不可。grep で代替。）

- [ ] **Step 4: Commit**

```bash
git add vspeech/worker/tts.py gui/gui.py
git commit -m "refactor: worker/tts.py・gui.py を pydantic v2 へ移行"
```

---

## Task 6: テスト更新 + secret round-trip + フルスイート緑

**Files:** `tests/test_voicevox_config.py`, `tests/test_config_secret.py`

- [ ] **Step 1: test_voicevox_config.py の parse_obj を model_validate に**

`VoicevoxConfig.parse_obj(` → `VoicevoxConfig.model_validate(`。

- [ ] **Step 2: secret round-trip テストを追加**

Create `tests/test_config_secret.py`:
```python
import json

from vspeech.config import Config


def test_secret_roundtrip_via_model_dump_json():
    cfg = Config.model_validate({"ami": {"appkey": "topsecret"}})
    assert cfg.ami.appkey.get_secret_value() == "topsecret"
    dumped = cfg.model_dump_json()
    # json_encoders により秘密値は平文で出力される（GUI→本体の受け渡しに必須）
    assert "topsecret" in dumped
    reloaded = Config.model_validate(json.loads(dumped))
    assert reloaded.ami.appkey.get_secret_value() == "topsecret"
```

- [ ] **Step 3: フルスイート緑を確認**

Run: `uv run pytest -q`
Expected: 全 pass（`voicevox_e2e` は deselected）。`test_worker_input.py` の2件を含め緑。失敗が残る場合は該当ファイルの変換漏れを修正（必要なら Task 3 Step 6 の `arbitrary_types_allowed` フォールバックを適用）。

- [ ] **Step 4: Commit**

```bash
git add tests/test_voicevox_config.py tests/test_config_secret.py
git commit -m "test: テストを pydantic v2 API へ更新し secret round-trip を追加"
```

---

## Task 7: CLAUDE.md を v2 前提へ更新しコミット

**Files:** `CLAUDE.md`（従来未追跡 → 今回からリポジトリ追跡）

- [ ] **Step 1: Pydantic 規約の記述を v2 に書き換え**

`CLAUDE.md` の Conventions 節、Pydantic に関する行
```markdown
- **Pydantic v1** (`pydantic>=1.10.7,<2`). Code uses v1 APIs: `BaseSettings`, `root_validator`, `parse_obj`, `.dict()`, `allow_population_by_field_name`, `orm_mode`, `SecretStr`. Do **not** introduce Pydantic v2 syntax.
```
を次に置換:
```markdown
- **Pydantic v2** (`pydantic>=2,<3`). Code uses v2 APIs: `model_config = ConfigDict(...)` / `SettingsConfigDict(...)`, `model_validate`, `model_dump`, `model_dump_json`, `model_validator(mode="after")`, `populate_by_name`, `from_attributes`, `AliasChoices`, `SecretStr`. `BaseSettings` is imported from **`pydantic-settings`**. Do **not** reintroduce v1 APIs (`parse_obj`/`.dict()`/`.json()`/`root_validator`/`orm_mode`/`Field(env=)`).
```
（`Config loading` 行で `parse_obj` に言及があれば `model_validate` に更新。）

- [ ] **Step 2: Commit（CLAUDE.md を追跡開始）**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md を Pydantic v2 前提に更新"
```

---

## Task 8: フル検証

**Files:** なし（検証のみ）

- [ ] **Step 1: フォーマット & lint（変更ファイル）**

Run: `uv run ruff format --check vspeech/ gui/ tests/ && uv run ruff check vspeech/ gui/ tests/`
Expected: 変更ファイルにエラー無し（既存の未変更ファイルの既存エラーは Task 範囲外。`ruff check .` の総数が base main と同等以下であること）。

- [ ] **Step 2: 警告ゼロ確認（pydantic v2 の protected namespace 等）**

Run: `uv run python -W error::UserWarning -c "import vspeech.config; import vspeech.shared_context; import vspeech.worker.tts"`
Expected: 例外なく終了（pydantic の UserWarning が出ない＝protected_namespaces 等の対応漏れが無い）。

- [ ] **Step 3: フルスイート**

Run: `uv run pytest -q`
Expected: 全 pass、`voicevox_e2e` deselected。`test_worker_input.py` の2件も pass。

- [ ] **Step 4: 残存 v1 API の最終 grep**

Run: `grep -rnE 'parse_obj|\.dict\(\)|\.json\(\)|root_validator|orm_mode|from_orm|allow_population_by_field_name|BaseSettings\.Config|Field\([^)]*env=' vspeech/ gui/ tests/`
Expected: pydantic 由来のヒットはゼロ。**例外**: `vspeech/worker/transcription.py` の `r.json()`（httpx レスポンス）は pydantic とは無関係の正当なヒットなので除外してよい。それ以外（`parse_obj`/`.dict()`/`root_validator`/`orm_mode`/`from_orm`/`allow_population_by_field_name`/`BaseSettings.Config`/`Field(env=)`）はゼロであること。

- [ ] **Step 5: （ユーザ環境）実機 E2E**

Run（資産あり環境）: `uv run pytest -m voicevox_e2e -q`
Expected: pass（v2 で WorkerInput/SoundInput 経路が健全）。実装担当の環境では実行不可、ユーザが確認。

---

## 完了条件

- `uv.lock` が pydantic v2 + pydantic-settings で解決。
- `uv run pytest`（通常実行）が全緑（既存2件は pass へ転じる）。
- `python -W error::UserWarning` でモジュール import が警告ゼロ。
- 残存 v1 pydantic API ゼロ。
- CLAUDE.md が v2 前提に更新され追跡されている。
