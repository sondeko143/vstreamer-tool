# GUI 全面書き直し: マルチ pipeline 管理 (フェーズ1) 設計

- 日付: 2026-07-12
- ブランチ: `feat/gui-multi-pipeline`
- 状態: 設計承認済み → 実装計画へ

## 1. 概要とゴール

現行 GUI (`gui/gui.py`, 1460 行の単一巨大クラス) は「単一 `Config` を全 worker タブで編集し、**1 つ**の vspeech サブプロセスだけを起動/停止する」構造になっている。これを全面的に書き直し、次を満たす:

1. **GUI から起動する場合は、ユーザーデフォルトのプロファイルから常に設定を読む** — 起動時に固定のユーザー設定場所(platformdirs)を常に読む。config ファイル引数は不要にする。
2. **複数の pipeline を起動/停止できる** — 各 pipeline は独立した vspeech サブプロセス。個別に起動/停止。
3. **各 pipeline の設定を管理できる** — pipeline ごとに専用 `Config` を GUI で編集。
4. **config の migration 機構** — 仕様変更 (フィールド追加/リネーム) と壊れたファイルの load に備える。
5. **とりあえず起動すれば何か動かせることを重視** — プリセットレシピで、新規 pipeline が最初から配線済み・実行可能。

### 非ゴール (フェーズ1)

- subtitle / translation worker のフォーム対応 (生 TOML で編集可能なまま)。
- 5 worker (recording/playback/vc/transcription/tts) 以外のバックエンド詳細をフォーム化すること (生 TOML)。
- pause/resume/reload の実行中コントロール (フェーズ2)。
- リモート (マルチマシン) 配線の GUI 補助。

## 2. 概念

- **プロファイル (Profile)** — platformdirs のユーザー設定 dir 配下の作業空間。GUI が起動時に常に読む単一の場所。「default 雛形 Config + pipeline 群 + マニフェスト」を含む。
- **pipeline** — 独立した vspeech サブプロセス。専用 `Config` と専用 `listen_port` を持ち、個別に起動/停止できる。GUI で各自の完全な `Config` を独立編集する。
- **レシピ (Recipe)** — 新規 pipeline の雛形。worker の `enable` と route 配線を一括で与える純粋関数 `apply(base: Config) -> Config`。

現アーキテクチャでは「1 vspeech プロセス = 1 config = 1 listen_port = 有効化された worker 群」であり、pipeline はこの単位に一致する。

## 3. データモデルと永続化

### ファイル配置

```
<user_config_dir>/vstreamer/        (Win: %APPDATA%\vstreamer\ / Linux: ~/.config/vstreamer/)
├─ default.toml          ユーザーデフォルトプロファイル (雛形 Config)。無ければ Config() 既定で自動生成
├─ pipelines.toml        マニフェスト (GUI 専用メタデータ)。順序保持
└─ pipelines/<id>.toml   各 pipeline の完全な vspeech Config (純粋 Config 形状)
```

- `<id>` は衝突しない短い識別子 (例: uuid4 の hex 先頭 8 桁)。
- config ファイル (`default.toml`, `pipelines/<id>.toml`) は **純粋な `Config` 形状**を保つ。runtime が `--config` で直接読むため、`Config` 以外のキーを入れてはならない (§9 の実測理由)。

### マニフェスト (`pipelines.toml`) スキーマ

```toml
profile_version = 1
default_config_version = 1

[[pipelines]]
id = "ab12cd34"
name = "文字起こし+読み上げ"
port = 8080
recipe = "mic_transcribe_tts"
config_version = 1
```

GUI 専用モデル:

- `PipelineEntry`: `id: str`, `name: str`, `port: int`, `recipe: str`, `config_version: int`
- `Profile`: `profile_version: int`, `default_config_version: int`, `pipelines: list[PipelineEntry]`

**バージョンは config ファイルではなくマニフェストに集約する** (§9 参照)。

### ポート採番 (空きポート自動割当)

base = 8080 から昇順に走査し、**予約済みでも OS で使用中でもない最初のポート**を割り当てる:

- **予約済み判定**: マニフェスト上の他 pipeline が持つ `port` の集合を避ける。
- **OS 空き判定** (`is_port_free`): `SO_REUSEADDR` なしのソケットを候補ポートに `bind` して成否を見る (成功で即 close)。他プロセス (外部アプリや別 pipeline の稼働中プロセス) が握るポートも避けられる。
- `allocate_free_port(claimed: set[int], base=8080) -> int` — 予約済み + OS 使用中をスキップし最初の空きを返す。上限まで空きが無ければ例外。
- フォームで手動編集可。
- **起動直前に再確認**: `Config.listen_port = entry.port` を注入する直前に `is_port_free(port)` を再チェックし、既に塞がっていれば警告して起動を止める (割当てから起動までの間に他プロセスが握った競合を検出)。

`is_port_free` / `allocate_free_port` は `gui/ports.py` に置く (OS ソケット判定は monkeypatch でテスト可能にする)。

### 「常にユーザーデフォルトプロファイルから読む」の解釈

- GUI 本体は起動時に常にこの固定場所の**プロファイル作業空間を読む** (config ファイルパス引数を取らない)。
- 新規 pipeline は `default.toml` (ユーザーデフォルトプロファイル) を雛形にコピー → レシピ適用 → 保存する。
- 既存 pipeline は各自の完全 `Config` を読む。以後は独立編集 (「各 pipeline の設定を管理」)。

## 4. モジュール構成 (`gui/` を全面書き直し)

| ファイル | 責務 |
|---|---|
| `gui/__main__.py` | エントリ。`app.main()` 呼び出し |
| `gui/app.py` | click エントリ `main()` + メインウィンドウ (左=pipeline 一覧 / 右=選択 pipeline エディタ) の配線 |
| `gui/paths.py` | platformdirs でプロファイル dir 解決 + ファイル配置。純粋・テスト可 |
| `gui/profile.py` | `PipelineEntry`/`Profile` モデル + 読み書き (マニフェスト・各 config・雛形)。安全ロード。純粋・テスト可 |
| `gui/recipes.py` | レシピ登録簿 (名前 → `apply(base: Config) -> Config`)。純粋・テスト可 |
| `gui/migration.py` | config / マニフェストの versioning + migration チェーン + 退避 + 安全ローダ。純粋・テスト可 |
| `gui/ports.py` | `is_port_free` / `allocate_free_port` (空きポート自動割当)。OS ソケット判定は monkeypatch でテスト可 |
| `gui/process.py` | `PipelineRunner`: サブプロセス生死・ログ読取スレッド・gRPC 送信 (send_text/ping) |
| `gui/pipeline_editor.py` | 右パネル (フォーム/生 TOML タブ + 起動/停止/状態 + テキスト送信 + ログ) 1 pipeline 分 |
| `gui/form.py` | 最小フォーム (必須フィールド宣言 + `Config` 双方向バインド) |
| `gui/rawedit.py` | 生 TOML エディタ (`Config` ⇄ TOML ラウンドトリップ、apply 時に検証) |
| `gui/widgets.py` | 小物ウィジェット (Checkbutton/Textbox/Spinbox ラッパ, ログ用 `TextHandler`) を旧 `gui.py` から回収 |
| `gui/autocomplete_combobox.py` | **維持** (デバイス選択に再利用) |

- 旧 `gui/gui.py` (単一巨大クラス) は**削除**。
- `gui/dummy_param.py` は VR2 全パラメータ用で最小フォームでは不使用 → **削除** (デッドコード化するため)。

依存方向: `gui/` → `vspeech/` の一方向のみ (現行同様、逆参照しない)。

## 5. レシピ

新規 pipeline 作成時にレシピを選ぶと、`enable` と route 配線が自動で揃う。route の伝播はコードで確認済み: `recording.routes_list` の各チェーンが下流全体を担い、各 worker が `WorkerOutput.from_input` で `following_events` を次段へ伝播する。テキスト始点は `text_send_operations`。

| レシピ key | 表示名 | 有効 worker | 配線 |
|---|---|---|---|
| `mic_loopback` | マイク→再生 (モニター) | recording, playback | `recording.routes_list = [["playback"]]` |
| `mic_transcribe_tts` | マイク→文字起こし→読み上げ→再生 | recording, transcription, tts, playback | `recording.routes_list = [["transcription","tts","playback"]]` |
| `mic_vc` | マイク→ボイチェン→再生 | recording, vc, playback | `recording.routes_list = [["vc","playback"]]` |
| `text_tts` | テキスト→読み上げ→再生 | tts, playback | `text_send_operations = [["tts","playback"]]` (recording 無効、GUI のテキスト送信で駆動) |
| `blank` | 空 (default のまま) | (雛形のまま) | 変更なし |

- `mic_loopback` は ML 非依存で、audio extra とデバイスさえあれば確実に動くスモークテスト。
- レシピ関数は純粋: `apply(base: Config) -> Config`。`base` は `default.toml` のディープコピー。該当 worker の `enable=True` と routes を設定し、それ以外は base を尊重する。

## 6. 最小フォームの必須フィールド

worker ごとに `enable` + 起動に効く項目のみをフォーム化。バックエンド系は `worker_type` 選択に応じて条件表示。残り全項目は生 TOML エディタで編集する。

| worker | フィールド (常時) | バックエンド条件表示 |
|---|---|---|
| recording | `recording.enable`, `recording.input_device_index` (コンボ), `recording.rate` (既定 16000), `recording.silence_threshold` | — |
| playback | `playback.enable`, `playback.output_device_index` (コンボ), `playback.volume` | — |
| transcription | `transcription.enable`, `transcription.worker_type` (コンボ) | WHISPER → `whisper.model`, `whisper.gpu_id` / GCP → `gcp.service_account_file_path` / ACP → `ami.appkey`, `ami.engine_uri`, `ami.engine_name`, `ami.service_id` |
| tts | `tts.enable`, `tts.worker_type` (コンボ) | VOICEVOX → `voicevox.openjtalk_dir`, `voicevox.model_dir`, `voicevox.onnxruntime_path`, `voicevox.speaker_id` / VR2 → `vr2.voice_name` |
| vc | `vc.enable` | `rvc.model_file`, `rvc.hubert_model_file`, `rvc.rmvpe_model_file`, `rvc.f0_up_key`, `rvc.gpu_id` |

- フィールド仕様は宣言的テーブルとして定義 (config パス, ウィジェット種別, 数値範囲/増分)。旧 `gui.py` の `draw_sb`/`draw_cb`/`draw_tb`/`draw_checkbutton` 相当を `widgets.py` から回収して使う。
- デバイス列挙は `vspeech.lib.audio.list_all_devices(input=/output=)`。audio extra 未導入時は空リストで劣化し、手動 index 入力を許す (`import` を try/except で守る)。
- SecretStr フィールド (`ami.appkey`) は文字列入力を `SecretStr(...)` に包んで set (旧 `set_config` 同様)。

## 7. フォーム ⇄ 生 TOML の同期

- フォーム編集 → メモリ上の `Config` を更新 (dirty マーク)。
- 生 TOML タブへ切替時: 現 `Config` を `export_to_toml()` で TOML 化して表示。
- 生 TOML の apply: `toml.loads` → migration チェーン → `Config.model_validate`。成功でメモリ `Config` を差し替え、フォームへ反映。失敗はエラーバナー表示 (保存はブロック)。
- 保存: `Config` を `pipelines/<id>.toml` に書き、マニフェストの該当 `config_version` を現行へ更新。

## 8. プロセス / 実行中コントロール

`PipelineRunner` が 1 pipeline 分のサブプロセスを管理:

- **起動**: 現 `Config` を保存 (`listen_port` 注入) → `Popen([sys.executable, "-m", "vspeech", "--config", <path>])`。`nosec`: 固定 argv・shell なし・自作パス。
- **ログ**: プロセスごとに stdout/stderr を読む reader スレッド → `widget.after(0, ...)` で UI スレッドへ渡し、ログペインへ追記 (旧 `TextHandler` の after パターンを踏襲)。
- **テキスト送信**: `localhost:<port>` へ gRPC `Command`(chains=`text_send_operations`, `operand.text`=入力行)。tts→再生 をマイク無しで即確認できる。
- **停止**: `terminate()` + `wait()`。プロセス死活は `after` ポーリングで状態ランプ (●running / ■stopped) を更新。
- gRPC 送信は localhost・短時間なのでインライン実行で可 (旧 GUI 同様)。ログ読取のみ別スレッド。

## 9. Config migration と壊れたファイル耐性 (GUI 専用)

### バージョンの持ち方 (実測により確定)

`Config` は `extra="forbid"` (pydantic-settings BaseSettings 既定)。実測で確認:

```
Config.model_validate({..., 'config_version': 3}) → ValidationError: extra_forbidden
```

したがって config ファイル内に version キーを埋め込むと **runtime の `--config` 起動が壊れる**。バージョンは config ファイルには入れず、GUI 専用マニフェスト (`pipelines.toml`) に集約する。config ファイルは純粋な `Config` 形状を保つ。

### `gui/migration.py`

- `CURRENT_CONFIG_VERSION = 1`, `CURRENT_PROFILE_VERSION = 1` (現行基準)。
- `Migration(to_version: int, describe: str, apply: Callable[[dict], dict])` の**順序付きチェーン** (`CONFIG_MIGRATIONS`, `PROFILE_MIGRATIONS`)。
- `migrate_dict(data, from_version, migrations, current) -> tuple[dict, int]` — `from_version` を起点に `to_version > from_version` のステップだけ順に適用。
- **各 migration は shape 検出で冪等**に書く (旧形状があれば変換、無ければ no-op)。マニフェスト消失・手編集でバージョンが信用できない場合でも「version 0 (未記録) から全チェーン再適用」を安全にできる。
- version 欠落 (未記録/インポート) → **0** として全チェーンを走らせる。
- フェーズ1は実マイグレーション 0 件 (基準版 = 1)。**チェーン実行器 + 退避 + テスト**で機構を確立し、将来のフィールド追加/リネームは `to_version=2` 以降のステップを足すだけ。

### 安全ロードと退避

- `quarantine(path) -> Path` — 原本を `<path>.bak-<n>` に複製 (番号を増やして非破壊、原本は消さない)。上書き前に必ず退避。
- `LoadResult` — `ok: bool`, `value: T | None`, `error: str | None`, `raw_text: str | None`, `migrated: bool`, `quarantined_path: Path | None`。
- 安全ローダの流れ: テキスト読取 → parse (toml) → migration → validate。
- **移行が起きたら** (`migrated=True`): 原本を退避してから現行版へ**書き戻し**、マニフェストの version を更新 (ファイルを現行へ収束)。

### 壊れた時の挙動 (呼び側で方針を分ける)

| 対象 | 壊れた時 |
|---|---|
| `pipelines/<id>.toml` (pipeline) | 退避 → **生 TOML に壊れた中身を表示 + エラーバナー + 退避先パス**。フォーム/起動は無効化。手修正して apply→検証成功で復帰。 |
| `default.toml` (雛形) | 退避 → `Config()` 既定へ fallback + 警告ログ (GUI は必ず起動できる)。 |
| `pipelines.toml` (マニフェスト) | 退避 → 空プロファイルへ fallback + 警告ログ (GUI は必ず起動できる)。 |

## 10. メインウィンドウのレイアウト

```
┌─────────────────────────────────────────────┐
│ File (メニュー: プロファイル dir を開く 等)   │
├──────────────┬──────────────────────────────┤
│ Pipelines    │  [選択 pipeline エディタ]      │
│ ─────────    │  name:[   ] port:[   ]         │
│ ●A transc :80│  recipe:[▼ マイク→…→再生]     │
│ ■B vc     :81│  ┌ Form ┬ Raw TOML ┐          │
│ ●C loopbk :82│  │ ☑ recording  入力:[▼Mic]  │ │
│ [+ new][del] │  │ ☑ transcription [▼WHISPER]│ │
│              │  │ ☑ tts [▼VOICEVOX] ...     │ │
│              │  ├───────────────────────────┤ │
│              │  │ [Start][Stop] ●running     │ │
│              │  │ 送信:[こんにちは__][send]  │ │
│              │  │ log: ....................  │ │
│              │  └───────────────────────────┘ │
└──────────────┴──────────────────────────────┘
```

- 左: pipeline 一覧 (状態ランプ + ポート)。`[+ new]` はレシピ選択ダイアログ、`[del]` は選択削除 (config ファイルも退避)。
- 右: 選択 pipeline の Form / Raw TOML タブ + 実行中コントロール (Start/Stop/状態) + テキスト送信 + ログペイン。

## 11. 依存

- `gui` extra に **`platformdirs`** を追加 (純 Python・cp314 wheel あり)。
- 既存の `ttkbootstrap`, `pillow` は維持。
- gRPC (`grpc`, `vstreamer-protos`) は base 依存で既存利用。

## 12. ドキュメント更新

- `CLAUDE.md` の「Run the GUI control panel」行 (`uv run python -m gui -c config.toml`) を新起動方法へ更新: 引数不要 (`uv run python -m gui`)、テスト/上書き用に `--profile-dir PATH` と `--theme` のみ。
- README に GUI 記述があれば併せて更新。
- 機械固有パス/ホストはドキュメントに直書きせず placeholder (`<USER>` 等) を使う (gitleaks ゲート方針)。プロファイル dir は platformdirs 抽象で記述。

## 13. テスト (Tk 非依存の純粋ロジックのみ、リポジトリ慣習通り)

- `tests/gui/test_recipes.py` — 各レシピが期待どおりの `enable` + routes を生む (**最重要**: route グラフの正しさ)。`mic_transcribe_tts` は `routes_list == [["transcription","tts","playback"]]` かつ 4 worker が enable、等。
- `tests/gui/test_profile.py` — 読み書きラウンドトリップ / 雛形自動生成 / マニフェスト追加・削除・並替 / 固定場所からの読込 / マニフェスト migration / 破損時 fallback (default → `Config()`, manifest → 空)。
- `tests/gui/test_paths.py` — プロファイル dir 解決 (platformdirs を monkeypatch / `--profile-dir` 上書き)。
- `tests/gui/test_migration.py` — チェーン順序 (注入した擬似 migration で N→N+1) / version 欠落→0→全適用 / 冪等性 / `quarantine` の非破壊連番 / 安全ローダ (正常・破損=raw 保持+退避・移行=書き戻し+原本退避)。
- `tests/gui/test_ports.py` — `allocate_free_port` が予約済み + OS 使用中 (availability 述語を monkeypatch) をスキップし最初の空きを返す / 全滅時に例外。
- `tests/gui/test_process.py` — argv 構築 (`python -m vspeech --config <path>`) / 起動前のポート注入・再確認 / gRPC `Command` 構築 (スタブ mock、実プロセス無し)。

Tk ウィジェットのテストは行わない (純粋ロジックのみ)。テストは `asyncio_mode = "auto"` の pytest 下。

## 14. 実装順序 (概略)

1. `gui/paths.py` + `gui/ports.py` + `gui/migration.py` (純粋基盤) + テスト。
2. `gui/profile.py` + `gui/recipes.py` (安全ロード/レシピ) + テスト。
3. `gui/process.py` (`PipelineRunner`) + テスト (argv/gRPC)。
4. `gui/widgets.py` (回収) → `gui/form.py` / `gui/rawedit.py`。
5. `gui/pipeline_editor.py` → `gui/app.py` → `gui/__main__.py`。
6. 旧 `gui/gui.py` / `gui/dummy_param.py` 削除、`pyproject.toml` に platformdirs、ドキュメント更新。
7. `uv run poe check` グリーン確認。手動 smoke: `mic_loopback` でマイク→再生、`text_tts` でテキスト→読み上げ。

## 15. フェーズ2 以降 (スコープ外メモ)

- pause/resume/reload の pipeline 単位コントロール。
- subtitle/translation を含む全 worker のフォーム化。
- リモート配線 (マルチマシン) の GUI 補助。
- pipeline 一括起動/停止、起動プリセット。
