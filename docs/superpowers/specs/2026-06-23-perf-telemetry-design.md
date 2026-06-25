# 性能テレメトリ（終了時サマリ＋真E2E）設計書

- 日付: 2026-06-23
- ステータス: 承認済み（実装計画へ）
- ブランチ: `feat/perf-telemetry`（vstreamer-tool）／ protos 側は別repoで別ブランチ
- 関連: 改善案 F（per-destination transport, PR マージ済み）と同じ「性能改善」系。先行調査で **転送は総遅延の2〜3%**、支配項は RVC（F0/推論）とコールドスタートと判明済み。本タスクはその PDCA を回すための**計測基盤**を入れる。

## 1. 背景と目的

現状のログは人間向けライフサイクルログで、性能テレメトリではない（相関ID無し・各段の処理時間が不統一・キュー待ち未計測・`asctime` ベースでマシン間クロックskewに弱い）。本タスクで以下を満たす計測基盤を導入する。

- **消費モデル**: プロセス終了時に**自動サマリ集計**（段ごと count/p50/p95/max）。スクリプト不要、外部スタック不要。
- **計測項目**: ①各段の処理時間 ②真E2E遅延（発話停止→再生機で音が鳴るまで）。
- **真E2E**: プロセス／マシンを跨ぐため、トレースID＋原点時刻を **wire で伝搬**し、終点（playback）で `now - origin_ts` を算出。**マシン間はNTP同期前提**。
- リアルタイム経路のため計測は軽量（`perf_counter` 数十ns＋メモリappend）。`enable=false` で完全 no-op。

## 2. 非目標（スコープ外 / YAGNI）

- リアルタイムダッシュボード・Prometheus/OpenTelemetry 連携。
- キュー待ち時間・送信時間/サイズの計測（今回の選択外。将来 `record()` 追加で拡張可）。
- 近似E2E（合算推定）。真E2Eを採用。
- NTP 同期自体の自動化（運用前提として文書化。コードは skew 検知の警告のみ）。

## 3. 担い手（carrier）の決定

トレース伝搬は **vstreamer-protos に専用フィールドを追加（A2）**。理由: Operand レベルの一級フィールドで wire 的に正しく、`OperationRoute.queries`/`Params`/`EventAddress` 等価判定に一切混ざらない（ルーティング不変）。`encode_trace`/`decode_trace` の薄いシームに載せ降ろしを隔離する。

## 4. 前提作業（vstreamer-protos, 別repo）

`protos/vstreamer_protos/commander/commander.proto` の `Operand` に2フィールド追加（proto3後方互換: 旧コードは未知フィールド無視、欠落時は既定値 `""`/`0.0`）:

```proto
message Operand {
  Sound sound = 1;
  string text = 2;
  string file_path = 3;
  repeated string filters = 4;
  string trace_id = 5;    // 追加: 1発話の相関ID
  double origin_ts = 6;   // 追加: 発話原点のNTP同期壁時計（epoch秒）
}
```

手順:

1. `.proto` 編集。
2. `make -C python`（`grpc_tools.protoc`）で `commander_pb2.py` / `_pb2.pyi` / `_grpc.py` 再生成・commit。
3. push to `main` → GitHub Actions CI が wheel をビルドしリリース（prerelease, tag `main-<sha>`）。
4. vstreamer-tool の `pyproject.toml:89` の pin URL を新リリースへ更新 → `uv lock`。

> 実装は protos のローカル clone（`c:\Users\<USER>\vstreamer\vstreamer-protos`）で行う。`.pyi` も再生成されるので ty は新フィールドを認識する。

## 5. vstreamer-tool 側 設計

### 5.1 テレメトリレジストリ `vspeech/lib/telemetry.py`（新規）

プロセス内シングルトン `telemetry`。

- 状態: `enabled: bool`, `max_samples: int`, `_durations: dict[str, list[float]]`（段名→秒）, `_e2e: list[float]`。
- `record(stage: str, seconds: float)`: `enabled` 偽なら即 return。`_durations[stage]` に append、`len > max_samples` なら先頭を捨てる（直近 `max_samples` を保持）。
- `record_e2e(seconds: float)`: 同様に `_e2e` へ。
- `timer(stage: str)`: コンテキストマネージャ。`t = perf_counter()`; `yield`; `finally: record(stage, perf_counter() - t)`。
- `summary() -> dict[str, dict]`: 段ごと `{count, p50, p95, max, mean}`（`numpy.percentile`、既存依存）。`e2e` も同形で含める。
- `log_summary()`: 整形して `logger.info`（段が空ならスキップ）。
- `configure(config: TelemetryConfig)`: `enabled`/`max_samples` を設定。`reset()`: テスト用。

### 5.2 各段の処理時間計測

各ワーカーの**処理本体**を `with telemetry.timer(<stage>):` で囲む。段名は `EventType` 名（`"transcription"`/`"translation"`/`"tts"`/`"vc"`/`"playback"`）。

- transcription（whisper/gcp/ami の各 `transcribe` 呼び出し）。
- translation（翻訳呼び出し）。
- tts（vr2/voicevox の生成）。
- vc（`change_voice`）。既存の `vc_end-vc_start` ログ（[vc.py:170](../../../vspeech/worker/vc.py#L170)）は `telemetry.timer("vc")` に統一（生ログは残してよい）。
- playback（`stream.write`）。

### 5.3 トレース伝搬と真E2E

- `WorkerInput` / `WorkerOutput`（[shared_context.py](../../../vspeech/shared_context.py)）に `trace_id: str = ""`、`origin_ts: float = 0.0` を追加。
- **原点**: recording が1発話送出時に `trace_id = uuid4().hex`、`origin_ts = time.time()` を採番して `WorkerOutput` に設定。
- **伝搬（in-process）**: `WorkerOutput.from_input(worker_input)` が `trace_id`/`origin_ts` をコピー。
- **伝搬（wire）**: シーム関数を `shared_context.py`（呼び出し元 `to_pb`/`from_command` と同居）に置く。
  - `encode_trace(operand: Operand, trace_id: str, origin_ts: float)`: `operand.trace_id`/`operand.origin_ts` に書込。`WorkerOutput.to_pb` から呼ぶ。
  - `decode_trace(operand: Operand) -> tuple[str, float]`: 読戻し。`WorkerInput.from_command` から呼び、生成する `WorkerInput` に設定（`input_id` は従来通り再採番だが trace_id でプロセス跨ぎ相関が可能）。
- **終点**: playback がE2E算出。`origin_ts > 0` のとき `e2e = time.time() - origin_ts`。
  - 正常: `telemetry.record_e2e(e2e)`、`log_raw_e2e` 真なら `logger.info("e2e trace=%s %.3fs", trace_id, e2e)`。
  - **skew検知**: `e2e < 0` または `e2e > skew_warn_threshold`（既定 120s）なら `record_e2e` せず `logger.warning("clock skew suspected: e2e=%.3fs trace=%s (NTP同期を確認)", e2e, trace_id)`。

### 5.4 終了時サマリ

`vspeech_coro`（[main.py:33-70](../../../vspeech/main.py#L33-L70)）の TaskGroup を `try/finally` で包み、`finally: telemetry.log_summary()`。WorkerShutdown / KeyboardInterrupt いずれの終了でも各プロセスが自分の担当段サマリを出力、playback プロセスは E2E も出力。

### 5.5 設定 `TelemetryConfig`（config.py）

```toml
[telemetry]
enable = true
max_samples = 5000
log_raw_e2e = true
skew_warn_threshold = 120.0
```

`Config` に `telemetry: TelemetryConfig` を追加。`configure_logger` の近傍（`cmd`）で `telemetry.configure(config.telemetry)` を呼ぶ。`enable=false` で全 `record`/`timer` が no-op。

## 6. データフロー（真E2E, 1発話）

```text
recording(GPU機)         : trace_id=uuid, origin_ts=time.time()  → Operand へ encode
  └gRPC→ whisper(GPU機)  : decode→WorkerInput, timer("transcription"), from_input でコピー→encode
    └gRPC→ vc(GPU機)     : decode, timer("vc"), コピー→encode
      └gRPC→ playback(再生機): decode, timer("playback"), e2e=time.time()-origin_ts → record_e2e
```

各プロセス終了時に自段サマリ、playback は E2E サマリ。マシン間はNTP同期で `time.time()` が比較可能。

## 7. テスト方針（`tests/`, pytest `asyncio_mode="auto"`）

1. **レジストリ**: `record` 後 `summary` の count/p50/p95/max が既知入力に一致；`max_samples` 超過で直近のみ保持；`enabled=false` で no-op（summary 空）。
2. **timer**: `with timer("vc")` が `"vc"` に1サンプル記録。
3. **trace ラウンドトリップ**: `WorkerOutput.to_pb` → `WorkerInput.from_command` で `trace_id`/`origin_ts` が保存される。
4. **ルーティング不変**: 既存 `tests/test_event_chains.py` / `test_worker_input.py` が green（trace は Operand 直下、queries/Params/EventAddress 等価に非干渉）。
5. **E2E算出**: 既知 `origin_ts` から playback が `record_e2e` する；未来 `origin_ts`（負のe2e）で `record_e2e` せず警告。
6. **設定**: `enable=false` で計測コールがゼロ動作。

> protos 変更後は `uv run pytest`・`uv run ruff check .`・`uv run ty check` を全 green に保つ。新フィールドは `.pyi` 再生成で ty 認識。

## 8. リスクと緩和

| リスク | 緩和 |
| --- | --- |
| protos の wheel リリース前は tool 側が新フィールドを import/テスト不可 | 実装順序を「protos→wheel→tool」に固定（計画の Phase 0）。それまで tool 側は registry 等 carrier 非依存部のみ着手可。 |
| マシン間クロックskewでE2Eが無意味/負値 | NTP同期を前提として文書化。コードは skew 警告で誤設定を検知（5.3）。 |
| 計測がリアルタイム経路に負荷 | `perf_counter`＋list append のみ。`enable=false` で no-op。`max_samples` でメモリ上限。 |
| trace フィールドがルーティングに影響 | Operand 直下に置き、queries/Params/EventAddress 等価に一切混ぜない（テスト4で担保）。 |
| サマリ未出力（異常終了） | `vspeech_coro` の `finally` で出力（5.4）。 |

## 9. 影響範囲

- protos（別repo）: `commander.proto` ＋再生成スタブ、CIリリース。
- 追加: `vspeech/lib/telemetry.py`、テスト `tests/test_telemetry.py`。
- 変更: `vspeech/config.py`（`TelemetryConfig`）、`vspeech/shared_context.py`（`WorkerInput`/`WorkerOutput` の trace フィールド＋`to_pb`/`from_command`＋`encode_trace`/`decode_trace`）、`vspeech/main.py`（summary／configure）、各ワーカー（`timer` 挿入）、`vspeech/worker/recording.py`（原点採番）、`vspeech/worker/playback.py`（E2E算出）、`pyproject.toml`/`uv.lock`（pin更新）、`config.toml.example`（telemetry節＋NTP注記）。
