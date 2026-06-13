# Pydantic v1 → v2 移行 設計書

- 日付: 2026-06-14
- ステータス: 承認済み（実装計画へ）
- ブランチ: `pydantic-v2-migration`
- 前提: voicevox-core 0.16.4 化（PR #1, main マージ済み）でブロッカー解消。`uv.lock` 上で pydantic を要求するのは本プロジェクトのみ → `pydantic>=2` へ解決可能。

## 1. 背景と目的

`vspeech` / `gui` は Pydantic v1 (`>=1.10.7,<2`) に依存し、v1 専用 API（`BaseSettings`、`root_validator`、`parse_obj`、`.dict()`、`.json()`、`orm_mode`/`from_orm`、`allow_population_by_field_name`、`Field(env=)`、`json_encoders`）を使用している。これを **Pydantic v2 ネイティブ API** へ全面移行する（互換シム `pydantic.v1` は使わない）。

CLAUDE.md は現在「Pydantic v1 を使え／v2 構文を導入するな」と明記しているが、本タスクはユーザの明示指示によりこれを上書きする。**移行完了後に CLAUDE.md を v2 前提へ更新し、コミットする**（CLAUDE.md は従来未追跡だったが、今回リポジトリに追跡させる）。

## 2. 依存変更（pyproject.toml / uv.lock）

- `pydantic>=1.10.7,<2` → **`pydantic>=2,<3`**
- **`pydantic-settings>=2,<3` を追加**（v2 で `BaseSettings` は別パッケージへ分離）
- `uv lock` 再生成、`requirements-pod.txt` を `make` で再生成。

## 3. API 置換マップ（機械的変換）

| v1 | v2 | 箇所 |
|---|---|---|
| `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` / `SettingsConfigDict` | config.py |
| ネスト `class Config(BaseSettings.Config)`（`env_prefix`/`env_nested_delimiter`/`json_encoders`） | `model_config = SettingsConfigDict(env_prefix=..., env_nested_delimiter=..., json_encoders=...)` | config.py `Config` |
| `Field(default=8080, env="PORT")` | `Field(default=8080, validation_alias="PORT")` | config.py `listen_port` |
| `Field(..., cli=("-c","--channels"))` 等の未使用拡張 kwarg | 削除（アプリ内で未参照） | RecordingConfig |
| `Model.parse_obj(x)` | `Model.model_validate(x)` | config.py(`read_config_from_file`) / lib/ami.py / lib/audio.py(×8) / tests/test_voicevox_config.py |
| `self.dict(...)` | `self.model_dump(...)` | config.py(`export_to_toml`) / worker/tts.py(`params.dict(exclude=...)`) |
| `self.config.json()` | `self.config.model_dump_json()` | gui/gui.py(×3) |
| `@root_validator(pre=False, skip_on_failure=True)` | `@model_validator(mode="after")` | shared_context.py `WorkerInput` |
| `class Config: allow_population_by_field_name=True` | `model_config = ConfigDict(populate_by_name=True)` | shared_context.py `Params` |
| `class Config: orm_mode=True` + `Model.from_orm(x)` | `model_config = ConfigDict(from_attributes=True)` + `Model.model_validate(x)` | shared_context.py `SoundInput` / `from_orm`(×2) |
| `class Config: alias_generator=camelize` | `model_config = ConfigDict(alias_generator=camelize)` | lib/audio.py `HostAPIInfo` / `DeviceInfo` |

import 整理: `from pydantic import BaseModel, Field, ConfigDict, SecretStr, model_validator` を一行ずつ（ruff `force-single-line`）。`ValidationError` の import（transcription.py / tests）は v2 でも `from pydantic import ValidationError` で有効。

## 4. v2 固有の要対応（v1 にはなかった点）

### 4.1 protected namespace 衝突
v2 は **`model_` 接頭辞**（末尾アンダースコア含む）のフィールド名を予約しており、衝突すると警告。該当:
- `RvcConfig.model_file`
- `VoicevoxConfig.model_dir`

→ この2モデルに `model_config = ConfigDict(protected_namespaces=())` を付与して警告を無効化（フィールド名は変えない＝config 互換維持）。
注: `WhisperConfig.model` は `"model"` ちょうど（`model_` で始まらない）ので**対象外**。`RvcConfig.hubert_model_file` / `crepe_model_file` も接頭辞が異なり対象外。実装時に `uv run python -c "import vspeech.config"` 等で警告ゼロを確認する。

### 4.2 `filters: Iterable[str]`（WorkerInput）
v2 は `Iterable` 型を遅延バリデータ（使い切りイテレータ）でラップし得るため、複数回反復で問題が出る可能性。**`list[str]` に変更**して安全側に倒す（呼び出し側は `[]` や protobuf repeated を渡しており list 化に支障なし。実装時に filters 消費箇所を確認）。

### 4.3 BaseModel 内の stdlib dataclass フィールド（最重要リスク）
`WorkerInput` は `current_event: EventAddress` と `following_events: list[list[EventAddress]]` を持つ。`EventAddress` は stdlib `@dataclass`（独自 `__eq__`/`__hash__` を持ち、内部に alias 付き pydantic モデル `Params` を含む）。v2 は stdlib dataclass を再検証・再構築するため、v1 と挙動が変わり得る。

対応方針: まず素直に v2 変換し、`test_event_chains.py` / `test_worker_input.py` / `test_tts_worker.py` / `test_voicevox_e2e.py` を防護壁に挙動を確認。再検証で問題が出る場合は `WorkerInput` に `model_config = ConfigDict(arbitrary_types_allowed=True)`（および必要なら `revalidate_instances` 調整）を付与して、既に構築済みの `EventAddress` インスタンスをそのまま受け入れる。`Params`（純粋な pydantic モデル）は通常どおり検証。

### 4.4 `Optional[...]` の必須化
v2 では明示デフォルトの無い `Optional` は必須化されるが、本コードの Optional フィールドは全て `= None` / `Field(default=None)` 済み → 影響なし（確認のみ）。

## 5. 挙動変更（承認済み）

現状 `WorkerInput.root_validator` は `@classmethod` を `@root_validator` の上に重ねた誤りで**発火していない**（このため `test_worker_input.py::test_from_command_sound_data_invalid` と `::test_from_command_reload_invalid` が main でも失敗中）。`@model_validator(mode="after")`（インスタンスメソッド、`self.sound`/`self.current_event` を参照し、不正時に `ValueError` を送出）に直すと**正しく発火**する。

結果として:
- 上記2件のテストが **pass に転じる**（期待値とする）。
- ライブパイプラインで、sound イベントに対する不正 sound 入力／file_path 無し reload が**実行時に拒否される**（正しい挙動。ユーザ承認済み）。

`mode="after"` のバリデータ本体ロジックは現行と等価（`is_sound_event(current_event)` かつ sound 不正なら ValueError、reload かつ file_path 無しなら ValueError）。

## 6. SecretStr のシリアライズ（採用: json_encoders 維持）

GUI が `config.model_dump_json()` で秘密値を**平文** JSON に書き出し → 本体プロセスが `read_config_from_file`（`model_validate`）で読み戻す経路を維持する必要がある。

- 採用: `model_config = SettingsConfigDict(..., json_encoders={SecretStr: lambda v: v.get_secret_value()})` を維持（v2 でも機能。非推奨マークはあるが動作する）。変更最小で round-trip を保証。
- `export_to_toml` は `self.model_dump()`（python モードで SecretStr を保持）＋ ami.appkey / gcp.service_account_info を `get_secret_value()` で手動置換、という現行構造を維持。`CustomTomlEncoder` は Path/Enum を処理。

## 7. テスト戦略

- 既存テスト全実行（特に `test_event_chains.py` / `test_worker_input.py`）を v2 で緑にする。
  - `test_worker_input.py` の2件は §5 により **pass へ転じる**ことを期待値とする。
- `tests/test_voicevox_config.py` の `parse_obj` を `model_validate` に更新。
- 既存のモック/ワーカーテスト（`test_tts_worker.py` 等）が v2 でも通ることを確認。
- secret round-trip の小テストを追加（`Config().model_dump_json()` が appkey を平文で出力し、`model_validate(json)` で読み戻せること）。
- `voicevox_e2e` は資産ゲートのまま（v2 で WorkerInput 構築が壊れていないことを実機で再確認可能）。

## 8. ドキュメント

- **CLAUDE.md** を v2 前提に更新（「Pydantic v2 (`>=2,<3`)。`model_config`/`ConfigDict`/`SettingsConfigDict`/`model_validate`/`model_dump`/`model_dump_json`/`model_validator` を使う。`BaseSettings` は `pydantic-settings` から」等）。**移行完了後にコミット**（今回からリポジトリに追跡させる）。
- `config.toml.example` は設定キー不変のため変更不要。

## 9. 検証範囲

担当（実装）が検証できる:
- `uv lock` が pydantic v2 + pydantic-settings で解決。
- `uv run pytest`（通常実行）が緑（既存2件は pass へ転じる、`voicevox_e2e` は deselected）。
- 変更ファイルが `ruff check`/`ruff format` クリーン、`ty` 新規実害なし。

ユーザ環境でのみ検証可能:
- `voicevox_e2e` 実機合成（v2 で WorkerInput/SoundInput 経路が健全なこと）。

## 10. 想定リスク

- §4.3 の dataclass 再検証が最大リスク。テストで早期検出し `arbitrary_types_allowed` 等で対応。
- `json_encoders` は v2 で非推奨。将来 `field_serializer` への置換余地あり（今回は範囲外）。
- pydantic-settings v2 の env 読み込み（`env_prefix`/`env_nested_delimiter`）は v1 と同等機能だが、ネスト区切りの挙動を env ベース構築のテストで確認する余地あり（現状 env ベースの自動テストは無い）。
