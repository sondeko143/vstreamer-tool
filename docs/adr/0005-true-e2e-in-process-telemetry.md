# 0005. 真 E2E を wire 伝搬で測り消費はプロセス内自己集計に限定する

- Status: Accepted
- Date: 2026-06-23
- Related: spec [2026-06-23-perf-telemetry-design](../superpowers/specs/2026-06-23-perf-telemetry-design.md) / [ADR-0006](0006-clock-skew-threshold-warning.md) / [ADR-0007](0007-per-request-jsonl-telemetry.md)

## Context

現状のログは人間向けライフサイクルログで、相関 ID が無く各段の処理時間が不統一、キュー待ちも未計測で、`asctime` ベースのためマシン間クロック skew に弱い。性能改善の PDCA を回すには、各段の処理時間と、発話停止から再生機で音が鳴るまでの真 E2E 遅延を軽量に計測する基盤が要る。E2E はプロセス／マシンを跨ぐため、各段合算の近似では計れない。

## Decision

E2E は各段合算の近似ではなく、録音原点で `origin_ts=time.time()` と `trace_id` を採番して protobuf `Operand` の専用フィールド（`trace_id` field5:string, `origin_ts` field6:double, proto3 後方互換）で wire 伝搬し、終点で `now-origin_ts` を実測する。伝搬は薄い `encode_trace`/`decode_trace` のシームに隔離し、ルーティング等価判定を汚さない。消費モデルはプロセス内シングルトンに `perf_counter`＋list append で貯め、終了時に count/p50/p95/max を出す自己集計に限定し、`enable=false` で全 no-op とする。

## Alternatives rejected

- **各段合算の近似 E2E** — キュー待ち・転送・再生音長を取りこぼす。
- **trace をルート文字列 queries/Params に相乗り** — 経路の等価判定に trace が混ざり、ルーティングが不変でなくなる。
- **Prometheus / OpenTelemetry / リアルタイムダッシュボード** — 小規模の自己運用にはオーバースペック。

## Consequences

別 repo `vstreamer-protos` の `.proto` 変更・スタブ再生成・wheel リリース・pin 更新が前提作業として発生する。マシン間で `time.time()` を比較するため NTP 同期が前提化する（→ [ADR-0006](0006-clock-skew-threshold-warning.md)）。外部依存ゼロで回るが、履歴の永続化とライブ可視化は将来拡張として残る。
