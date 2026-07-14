# 0030. pyworld を遅延 import 化し既定 f0 抽出器を rmvpe にして rvc extra から撤去する

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

pyworld は cp314 wheel 不在・`Requires-Python` null のため、3.14 では sdist ビルドに落ちる（roadmap が「3.14 の唯一の wheel ギャップ」と断定していた依存）。一方、実機の f0 抽出は既に rmvpe を使っている。[ADR-0025](0025-target-python-314-phased.md) のフェーズ③で対処する。

## Decision

フェーズ③で pyworld の import を dio / harvest の関数内へ遅延化し、既定の `f0_extractor_type` を harvest → **rmvpe** へ変更、rvc extra から pyworld を撤去する（dio / harvest は手動導入時のみ利用可）。

## Alternatives rejected

- **pyworld を必須依存のまま残す** — 3.14 で rvc extra が解決しない。
- **3.14 で sdist ビルドする** — 失敗する。

## Consequences

既定挙動が rmvpe になる（実機は既に rmvpe なので実害は無いが、既定変更を伴う決定である）。
