# 0006. クロック skew は自動同期せず閾値警告で検知する

- Status: Accepted
- Date: 2026-06-23
- Related: spec [2026-06-23-perf-telemetry-design](../superpowers/specs/2026-06-23-perf-telemetry-design.md) / [ADR-0005](0005-true-e2e-in-process-telemetry.md)

## Context

[ADR-0005](0005-true-e2e-in-process-telemetry.md) の真 E2E は原点と終点で `time.time()` を比較するため、マシン間のクロック skew があると E2E が無意味な値や負値になる。時刻同期をコードで抱え込むと OS 運用領域まで複雑化する。

## Decision

マシン間の `time.time()` 比較におけるクロック skew は、コードで自動同期せず、NTP 同期を運用前提として文書化する。`e2e<0` または `e2e>skew_warn_threshold`（既定 120s）のサンプルは集計から除外して警告する。保持サンプル数には上限を設け、メモリの無制限増加を防ぐ。

## Alternatives rejected

- **コード側で時刻同期を自動化** — OS 運用領域まで抱え込み複雑化する。

## Consequences

誤設定は警告で顕在化する。一方、閾値内の緩やかな skew は検知できない（120s は緩め）。後にこの閾値と skew による e2e 汚染が運用課題として顕在化しうる。
