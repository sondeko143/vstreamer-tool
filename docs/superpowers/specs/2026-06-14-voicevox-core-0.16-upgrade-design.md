# voicevox-core 0.14.3 → 0.16.4 アップグレード 設計書

- 日付: 2026-06-14
- ステータス: 承認済み（実装計画へ）
- スコープ: voicevox-core のアップグレードのみ。pydantic は v1 のまま据え置く（pydantic v2 化は別ブランチ）。

## 1. 背景と目的

`vstreamer-tool` を pydantic v2 へ移行する際の唯一のハードブロッカーが
`voicevox-core==0.14.3`（`pydantic>=1.9.2,<2` を要求、win32 のみ）である。
voicevox-core **0.16.4** は Python API から pydantic 依存を完全に削除しているため、
0.16.4 へ上げることでこのブロッカーが解消し、後続の pydantic v2 化が可能になる。

本ブランチではこの「ブロッカー除去」までを完結させる。pydantic 本体は v1 のままとし、
段階的に検証しやすくする。

## 2. 0.14.3 → 0.16.4 の API デルタ

| 項目 | 0.14.3（現状） | 0.16.4（移行先） |
|---|---|---|
| pydantic 依存 | `>=1.9.2,<2`（ブロッカー） | なし |
| wheel | `0.14.3+cuda-cp38-abi3-win_amd64.whl`（資産同梱） | `0.16.4-cp310-abi3-win_amd64.whl`（abi3 cp310 ＝ 3.11 可、`+cuda` 廃止） |
| 中核クラス | `VoicevoxCore` モノリス | `Onnxruntime` + `OpenJtalk` + `Synthesizer` + `VoiceModelFile`（`voicevox_core.blocking`） |
| ONNX Runtime | wheel に CUDA 同梱 | 専用ビルド `voicevox_onnxruntime`（v1.17.3）を別途取得し `Onnxruntime.load_once(filename=...)` |
| 辞書 | wheel 同梱 | 別途取得した `open_jtalk_dic_utf_8-1.11` を `OpenJtalk(dict_dir)` |
| モデル | speaker_id で同梱モデルを `load_model` | `.vvm` ファイルをパス指定で `VoiceModelFile.open` → `load_voice_model` |
| 合成 | `core.audio_query(text, speaker_id=)` / `core.synthesis(aq, speaker_id=)` | `synth.create_audio_query(text, style_id)` / `synth.synthesis(aq, style_id)` |
| AudioQuery | pydantic モデル（属性可変） | プレーンな型（属性可変、フィールド同一） |

### 0.16.4 blocking API の使用形（example/python/talk.py より）

```python
from voicevox_core import AccelerationMode
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

onnxruntime = Onnxruntime.load_once(filename=ort_path)  # voicevox_onnxruntime の実パス
synthesizer = Synthesizer(
    onnxruntime,
    OpenJtalk(dict_dir),
    acceleration_mode=AccelerationMode.AUTO,
    cpu_num_threads=...,
)
with VoiceModelFile.open(vvm_path) as model:
    synthesizer.load_voice_model(model)
audio_query = synthesizer.create_audio_query(text, style_id)  # style_id: int 可
# audio_query の各 *_scale / *_phoneme_length を上書き
wav = synthesizer.synthesis(audio_query, style_id)  # WAV bytes（既定 24000Hz/mono/INT16）
```

AudioQuery の可変フィールド: `speed_scale`, `pitch_scale`, `intonation_scale`,
`volume_scale`, `pre_phoneme_length`, `post_phoneme_length`
（既存 `VoicevoxParam` のフィールドと一致）。

## 3. 重要な技術的制約

### 3.1 voicevox_onnxruntime と onnxruntime-gpu の併存

- 本プロジェクトは既に `onnxruntime-gpu`（whisper/rvc 用）に依存している。
- voicevox 0.16 は**専用ビルドの `voicevox_onnxruntime` を要求**し、pip の
  `onnxruntime`/`onnxruntime-gpu` では代替できない（別ライブラリ）。両者は別 DLL として共存する。
- 既知の不具合として、voicevox が誤って別の onnxruntime DLL を掴むケースがある。
  本プロジェクトは onnxruntime-gpu の DLL が PATH 上に存在するため、
  **`Onnxruntime.load_once(filename=<voicevox_onnxruntime の実パス>)` の明示指定を既定運用**とする。

### 3.2 実行時資産が wheel 非同梱

0.16 では onnxruntime ライブラリ・辞書・`.vvm` モデルが wheel に同梱されない。
公式ダウンローダ（`download-windows-x64.exe`、GPU は CUDA ビルド）で取得し、config で指す。

## 4. 変更内容

### 4.1 依存・パッケージング（pyproject.toml / uv.lock）

- `[tool.uv.sources]` の `voicevox-core` URL を
  `https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-win_amd64.whl`
  に変更。
- `[project.optional-dependencies]` の `voicevox = ["voicevox-core ; sys_platform == 'win32'"]`（win32 マーカー）は維持。
- `uv lock` を再生成し、**voicevox-core が pydantic<2 を要求しなくなる**ことを確認（ブロッカー解消の実証）。
  pydantic は引き続き 1.10.x が解決されること、他に新たな競合がないことを確認。
- `make`（`uv export --extra voicevox`）で `requirements-pod.txt` を再生成。
  Linux pod は win32 マーカーにより voicevox 非搭載のまま（影響なし）。

### 4.2 Config スキーマ（vspeech/config.py: VoicevoxConfig）

維持:
- `speaker_id: int = 1` — 意味的には style_id だが、config キー名は互換のため据え置き。
- `params: VoicevoxParam`
- `openjtalk_dir: Path` — 辞書ディレクトリ（既定値はダウンローダ配置に合わせて見直す）。

追加:
- `model_dir: Path` — `.vvm` 群を置くディレクトリ。
- `onnxruntime_path: Path | None = None` — `voicevox_onnxruntime` の実パス。
  None の場合は `load_once` の既定ファイル名解決にフォールバック（誤 DLL リスクありと docs に明記）。

`VoicevoxParam` は変更なし。`acceleration_mode` は AUTO 固定（YAGNI、config 化しない）。

### 4.3 ライブラリ書き換え（vspeech/lib/voicevox.py）

`Voicevox` クラスを 0.16 blocking API でラップ:

- `__init__(self, open_jtalk_dict_dir, model_dir, onnxruntime_path=None)`:
  - `Onnxruntime.load_once(filename=...)`（onnxruntime_path 指定時は明示、未指定は既定）。
  - `Synthesizer(ort, OpenJtalk(dict_dir), acceleration_mode=AccelerationMode.AUTO, cpu_num_threads=...)`。
  - `model_dir/*.vvm` を走査し、各 `VoiceModelFile.open(p).metas` から
    `style_id → vvm パス` の索引 `self._style_index: dict[int, Path]` を構築。
  - `self._loaded: set[int]`（ロード済み style_id 管理）。
- `load_model(self, style_id)`: 未ロードなら索引から vvm を引き、
  `with VoiceModelFile.open(path) as m: self.synthesizer.load_voice_model(m)`、ロード済みに記録。
  未知 style_id は `ValueError`（現状の例外挙動を踏襲）。
- `is_model_loaded(self, style_id) -> bool`: 内部集合で判定。
- `voicevox_tts(self, text, speaker_id, params) -> bytes`:
  `create_audio_query(text, speaker_id)` → `for key, value in params: setattr(aq, key, value)`
  → `synthesis(aq, speaker_id)` で WAV bytes を返す。

import: `from voicevox_core import AccelerationMode` および
`from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile`。

### 4.4 ワーカー（vspeech/worker/tts.py: voicevox_worker）

- `Voicevox(vvox_config.openjtalk_dir, vvox_config.model_dir, vvox_config.onnxruntime_path)` に変更。
- 遅延ロードの guard（`loaded_models`）、パラメータ合成（route の `spd`/`pit` で上書き、
  `params.dict(exclude={"speed_scale","pitch_scale"})` のマージ）、`speech[44:]`（WAV ヘッダ除去）、
  `rate=24000 / INT16 / mono` はそのまま維持。

### 4.5 GUI（gui/gui.py: draw_voicevox_tab）

`voicevox.model_dir` と `voicevox.onnxruntime_path` の入力欄を追加（既存 `openjtalk_dir` と同様の `draw_tb`）。

### 4.6 ドキュメント

- `config.toml.example`: `[voicevox]` セクションに `model_dir` / `onnxruntime_path` を追記し、
  ダウンローダでの資産取得手順と onnxruntime_path 明示の必要性を記載。
- `CLAUDE.md`: プラットフォーム制約の voicevox 行を 0.16.4 に更新し、
  「実行時資産は wheel 非同梱」「voicevox_onnxruntime は onnxruntime-gpu とは別物・併存」を追記。

## 5. テスト戦略（2層）

### 5.1 ワーカー単体テスト（完全自動・CI 可）— tests/test_tts_worker.py

既存 `tests/test_transcription_worker.py` を踏襲。`voicevox_worker` は非同期ジェネレータなので
`anext(...)` + `wait_for` で 1 件処理させる。ネイティブ依存 `Voicevox` を monkeypatch でフェイクに差し替え
（`voicevox_worker` 内の遅延 import を利用）、実行時資産・GPU 不要。検証項目:

- speaker_id 解決: route param `i`（speaker_id）が config の `speaker_id` を上書きする。
- パラメータ合成: route の `spd`/`pit` が config params を上書き、その他は config を踏襲する。
- 遅延ロード: 同一 speaker_id で `load_model` が一度だけ呼ばれる。
- 出力: `WorkerOutput.sound` が rate=24000 / INT16 / mono、`data == fake_wav[44:]`、
  `text == demojize(入力)`。
- 例外処理: `ValueError` / `UnicodeEncodeError` を握りつぶしクラッシュしない。

### 5.2 E2E テスト（専用マーカー・資産ゲート付き）— tests/test_voicevox_e2e.py

- `pyproject.toml` `[tool.pytest.ini_options]` に
  `markers = ["voicevox_e2e: 実資産が必要な VOICEVOX 実合成テスト"]` を登録。
- `addopts = "-m 'not voicevox_e2e'"` で**通常実行では自動除外**。
- 資産パスは環境変数（`VSPEECH_VVOX_ONNXRUNTIME` / `VSPEECH_VVOX_DICT` / `VSPEECH_VVOX_MODEL_DIR`）
  または既定 `tests/assets/voicevox/` から解決。実 `VoicevoxConfig` を組み、実 `Voicevox` で
  `voicevox_worker` を E2E 実行し、実 WAV（RIFF ヘッダ＋非空）を検証。
- 資産不在時は `skipif` で明示スキップ（CI でも事故らない）。
- 資産取得は `make voicevox-assets`（公式ダウンローダを `tests/assets/voicevox/` に展開）で用意。
  実行は「資産配置 → `uv run pytest -m voicevox_e2e`」。

## 6. 検証範囲

担当（私）が検証できる:
- `uv lock` / `uv sync --extra voicevox` が 0.16.4 で解決し、pydantic<2 ピンが消えること。
- `uv run ruff check` / `uv run ty check` がグリーン。
- `uv run pytest`（マーカー除外通常実行）がグリーン（5.1 を含む）。

ユーザ環境（Windows + GPU + 資産）でのみ検証可能:
- 5.2 の実合成 E2E（`uv run pytest -m voicevox_e2e`）。

## 7. 想定リスク

- `metas` の構造（`CharacterMeta.styles[].id`）は実装時に導入 wheel の型スタブで最終確認する。
- `synthesis` の WAV ヘッダが常に 44 バイトである前提（`speech[44:]`）は現状踏襲。0.16 でも既定
  24000Hz/mono/INT16 のため維持するが、E2E で実バイト列を確認する。
- onnxruntime 誤 DLL ロード（3.1）。`onnxruntime_path` 明示で回避。
