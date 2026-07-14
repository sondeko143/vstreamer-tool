# 0033. GUI の version/migration を専用マニフェストに隔離し config は純粋 Config 形状を保つ

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-gui-multi-pipeline-rewrite-design.md](../superpowers/specs/2026-07-12-gui-multi-pipeline-rewrite-design.md); [ADR-0032](0032-gui-multi-pipeline-rewrite.md)

## Context

実測で `Config` は `extra="forbid"`。config ファイルに version キーを埋めると runtime の config 起動が `ValidationError(extra_forbidden)` で壊れる。将来のフィールド追加 / リネームと、マニフェスト消失・手編集でバージョンが信用できないケースにも備えたい。

## Decision

config ファイル（`default.toml` / `pipelines/<id>.toml`）は純粋な `Config` 形状を保ち、version は GUI 専用の `pipelines.toml` マニフェストに集約する。migration は実 migration 0 件でも、順序付きチェーン実行器＋退避＋テストを基準版 = 1 で先に確立する。各 migration は shape 検出で冪等とし、version 欠落は 0 として全チェーン再適用を安全にする。

## Alternatives rejected

- **各 config ファイルに `config_version` を埋め込む** — `extra_forbidden` で runtime 起動が壊れる（実測確認済み）。
- **migration を必要になってから導入する** — 壊れたファイル耐性と「version 0 から安全再適用」が後付けできない。

## Consequences

runtime は config を無改修で直接読める。将来は `to_version>=2` を足すだけで拡張できる。冪等性が全 migration の必須制約になる。
