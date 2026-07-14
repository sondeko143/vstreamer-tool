# 0034. 壊れた GUI 入力に対し対象別復旧＋非破壊退避で必ず起動する

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-gui-multi-pipeline-rewrite-design.md](../superpowers/specs/2026-07-12-gui-multi-pipeline-rewrite-design.md); [ADR-0032](0032-gui-multi-pipeline-rewrite.md)

## Context

雛形・マニフェスト・pipeline config で望ましい復旧の仕方が異なり、かつ壊れたファイルでも GUI は必ず起動させたい。

## Decision

壊れた入力への挙動を対象ごとに分岐する——pipeline config は退避して生 TOML ＋エラーバナー表示（手修正の apply 成功で復帰）、雛形は退避して `Config()` 既定へ、マニフェストは退避して空プロファイルへ fallback する。上書き前には必ず非破壊の連番退避を行う。

## Alternatives rejected

- **一律 fallback する** — 破損した pipeline 設定を黙って捨て、ユーザーが手修正で復帰できなくなる。

## Consequences

GUI は必ず起動でき、失われる編集内容は退避に残る。
