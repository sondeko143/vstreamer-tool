# per-request 構造化テレメトリ（JSONL）設計書

- 日付: 2026-06-23
- ステータス: 承認済み（実装計画へ）
- ブランチ: `feat/per-request-jsonl-telemetry`
- 前提: [性能テレメトリ設計](2026-06-23-perf-telemetry-design.md)（実装済み・main マージ済み）の拡張。終了時サマリ＋真E2E は既に動作。

## 1. 背景と目的

既存テレメトリは各段の処理時間を**メモリ集計し終了時にサマリ出力**するだけで、1発話ごとの粒度はメモリ常駐（人間向け行ログは vc/playback のみ）。本拡張で **1発話ごとの各段所要時間を機械可読（JSONL）で即時に永続化**し、後から `trace_id` で全段・全マシンを join して分析できるようにする。

段は別プロセス/別マシンに分散しているため、**1レコード＝1段（trace_id で後結合）**とする（自完結の1行/リクエストは protos 再変更を要するため非採用）。

## 2. 非目標（YAGNI / スコープ外）

- 1リクエストを全段まとめた単一行（wire でのstage→duration伝搬・protos再変更が必要なため不採用）。
- payload サイズ・キュー待ち時間など追加メトリクスの記録。
- ローテーション/圧縮/外部送信。出力は追記のみ。
- 既存の終了時サマリ・人間向け行ログの変更（**非破壊**で並行追加）。

## 3. 設計

### 3.1 レコード形式

`telemetry.record(...)` / `record_e2e(...)` 発生時に、設定された JSONL ファイルへ1行追記：

```json
{"ts": 1718900000.123, "trace_id": "ab12cd…", "stage": "vc", "dur_s": 1.234, "pid": 12345}
```

- `ts`: `time.time()`（epoch秒, 壁時計。`origin_ts` と同一基準で相関可能）
- `trace_id`: 発話の相関ID（無い場合は `""`）
- `stage`: 段名（`"transcription"`/`"translation"`/`"tts"`/`"vc"`/`"playback"`）または `"e2e"`
- `dur_s`: 所要秒（E2E は発話→再生の総遅延）
- `pid`: `os.getpid()`（同一ファイルに複数プロセスが書いても識別可能）

### 3.2 trace_id の受け渡し

現状 `record`/`timer` は段名と時間のみ。以下に**任意引数 `trace_id: str = ""` を追加**（後方互換・既存呼び出しは `""`）：

- `Telemetry.record(stage, seconds, trace_id="")`
- `Telemetry.record_e2e(seconds, trace_id="")`
- `Telemetry.timer(stage, trace_id="")`（`__exit__` で `record(stage, dur, trace_id)`）

各ワーカーは処理中の `WorkerInput.trace_id` を渡す（[shared_context.py](../../../vspeech/shared_context.py) の `WorkerInput.trace_id`）:

- transcription/translation/tts: `with telemetry.timer("<stage>", trace_id=<item>.trace_id):`
- vc: `record_vc_elapsed(seconds, trace_id=speech.trace_id)`（[vc.py](../../../vspeech/worker/vc.py)）
- playback: `with telemetry.timer("playback", trace_id=speech.trace_id):` ＋ `record_playback_e2e` 内で `telemetry.record_e2e(e2e, trace_id=speech.trace_id)`（[playback.py](../../../vspeech/worker/playback.py)）

### 3.3 sink（出力先）と既定

`TelemetryConfig` に `jsonl_path: str = ""` を追加。

- **空＝無効（既定オフ）**。非空＝そのパスへ追記。`log_file` と同じ `%%`→`%` 変換後に strftime 適用（例 `./telemetry_%%Y_%%m_%%d.jsonl`）。**ローカル/UNCネットワークパス両対応**（§3.6）。
- `Telemetry.configure(enabled, max_samples, jsonl_path="")` がパスを受け、非空かつ `enabled` 時に、親ディレクトリを `mkdir(parents=True, exist_ok=True)` してから append モードでファイルを開いて保持。
- 書込: `json.dumps(record) + "\n"` を write→flush（発話レートで低頻度なので1行ごとflush＝クラッシュ耐性）。
- 失敗時（mkdir/オープン/書込）は `logger.warning` で1回通知し、以後 JSONL を無効化（パイプラインは止めない）。
- 各プロセスが自分の `jsonl_path` へ書く（`log_file` と同じ運用。プロセスごとに別パス推奨、`pid` でも識別可）。

### 3.4 実装の集中点

JSONL 書込は `Telemetry` 内の `_emit_jsonl(stage, seconds, trace_id)` に集約し、`record`/`record_e2e` から呼ぶ。全段が自動で対象になり、ワーカー側改修は trace_id を渡すだけ。メモリ集計（`summary`/`log_summary`）は不変。

### 3.5 起動時配線

`main.py cmd` の `telemetry.configure(...)` に `jsonl_path=config.telemetry.jsonl_path` を追加。strftime 適用は **`configure` 内**で `datetime.now().strftime(jsonl_path.replace("%%", "%"))` を行ってからファイルを開く（呼び出し側は生パターンを渡すだけ。`%%`→`%` は `log_file` と同じ規約）。`config.toml.example` に `jsonl_path` を記載。

### 3.6 ネットワーク/UNC パス対応（`jsonl_path` と `log_file` 両方）

`\\<NAS_HOST>\d\vs\...` のような UNC パスを `jsonl_path` と既存の `log_file` の両方で使えるようにする。

- **パス処理は本質的に UNC 対応済み**（検証済み）: `strftime` はバックスラッシュを保持し、`Path(unc).parent` は共有配下を正しく返し、`mkdir(parents=True, exist_ok=True)` の `parents` は共有ルート `\\host\share\` で停止する（ホスト/共有自体を作ろうとしない）。よって**文字列処理の追加変換は不要**。
- **要対応＝耐障害性（`log_file` の修正点）**: 現在の [logger.py `configure_logger`](../../../vspeech/logger.py) は `Path(filename).parent.mkdir(...)` と `TaskFileHandler(...)` を **例外保護していない**ため、起動時に共有が到達不能だと `configure_logger` が例外送出→**プロセス起動がクラッシュ**する。ネットワークパス運用ではこれが実害になるので、**mkdir とファイルハンドラ生成を try/except で保護**し、失敗時は stderr/stdout へ警告して**ファイル出力なしで起動継続**（stdout ハンドラと処理は生きる）。`jsonl_path` も同じ耐障害方針（§3.3）。
- **TOML での書き方（ドキュメント）**: Windows/UNC パスは **TOML のリテラル文字列（シングルクォート）**で書く。基本文字列（ダブルクォート）だと `\d`/`\v` 等がエスケープ解釈され壊れる。例:
  - `jsonl_path = '\\<NAS_HOST>\d\vs\tel_%%Y%%m%%d.jsonl'`
  - `log_file = '\\<NAS_HOST>\d\vs\voice_%%Y_%%m_%%d.log'`
  `config.toml.example` に上記の注記を追加する。

## 4. データフロー（1発話）

各プロセスが**自分の段の完了時**に1行ずつ自分の JSONL へ追記。例（trace_id=ab12）:

```text
whisper機:  {"ts":…, "trace_id":"ab12", "stage":"transcription", "dur_s":0.51, "pid":4001}
vc機:       {"ts":…, "trace_id":"ab12", "stage":"vc",            "dur_s":1.14, "pid":4002}
再生機:     {"ts":…, "trace_id":"ab12", "stage":"playback",      "dur_s":0.03, "pid":5001}
再生機:     {"ts":…, "trace_id":"ab12", "stage":"e2e",           "dur_s":1.71, "pid":5001}
```

分析時に `trace_id` で join → 1発話の全段内訳。

## 5. テスト方針（`tests/`）

1. **書込**: `configure(enabled=True, jsonl_path=tmp)` 後 `record("vc", 1.2, trace_id="t")` → ファイルに1行、parse して `{stage:"vc", dur_s:1.2, trace_id:"t", pid:int, ts:float}` を検証。
2. **無効**: `jsonl_path=""` で `record` → 書込ゼロ（ファイル未生成）。`enabled=False` でも書込ゼロ。
3. **timer 経由**: `with timer("tts", trace_id="t2")` → JSONL に `stage:"tts", trace_id:"t2"`。
4. **E2E**: `record_e2e(1.7, trace_id="t3")` → `stage:"e2e"` の行。
5. **valid JSON**: 全行が `json.loads` 可能。
6. **耐障害（jsonl）**: 作成不能なパス（例 存在しないドライブ `Z:\nope\x.jsonl`）を `jsonl_path` に与えても `configure`/`record` が例外を投げず、警告のうえ JSONL 無効化。
7. **耐障害（log_file）**: `configure_logger` に作成不能な `log_file` を与えてもクラッシュせず、stdout ハンドラのみで継続（mkdir/ハンドラ生成の try/except を検証）。
8. 既存 `test_telemetry.py`（summary 等）が green 維持（trace_id 追加は後方互換）。

## 6. リスクと緩和

| リスク | 緩和 |
| --- | --- |
| イベントループ上の同期ファイル書込で遅延 | 発話レート（数/秒）＋buffered append。`jsonl_path` 空で no-op。気になれば将来キュー化。 |
| 同一ファイルへ複数プロセス追記で行交錯 | 1 write=1行（小サイズはほぼ原子的）＋`pid` で識別。プロセス別パス推奨。 |
| 書込失敗でパイプライン停止 | try/except で warning＋以後無効化、停止しない。 |
| trace_id 引数追加で既存呼び出し破壊 | 既定 `""` の任意引数＝後方互換（テスト6で担保）。 |

## 7. 影響範囲

- 変更: [vspeech/lib/telemetry.py](../../../vspeech/lib/telemetry.py)（`record`/`record_e2e`/`timer` に trace_id、`_emit_jsonl`、`configure` に jsonl_path＋mkdir＋耐障害）、[vspeech/config.py](../../../vspeech/config.py)（`TelemetryConfig.jsonl_path`）、[vspeech/main.py](../../../vspeech/main.py)（configure 配線）、[vspeech/logger.py](../../../vspeech/logger.py)（`configure_logger` の mkdir/ファイルハンドラ生成を try/except 化＝UNC/ネットワーク耐障害, §3.6）、各ワーカー（transcription/translation/tts/vc/playback: trace_id を渡す）、`config.toml.example`（jsonl_path＋UNCパスの TOML リテラル文字列注記）。
- 追加: `tests/test_telemetry_jsonl.py`、`tests/test_logger_resilience.py`。
- 無改修: protos（wire 変更なし）、shared_context、sender/receiver。
