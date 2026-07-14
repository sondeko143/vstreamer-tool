# 0029. audioop を audioop-lts へ置換する

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

PEP 594 が Python 3.13+ で stdlib の audioop を削除する。audioop は録音の dBFS 算出と再生音量スケールという core 経路で使用している。[ADR-0025](0025-target-python-314-phased.md) のフェーズ③（3.14 化）で import が壊れる。

## Decision

フェーズ③で base 依存に `audioop-lts ; python_version >= '3.13'` を追加して `import audioop` を復元する。audioop を使う 3 箇所のコードは無変更とする。

## Alternatives rejected

- **numpy の astype で置換する** — 飽和ではなく wrap し、クリックアーティファクトを生む。

## Consequences

C ポートで飽和セマンティクスを保存できる。audioop-lts の `Requires-Python` が `>=3.13` のため、依存にはマーカー指定が必須となる。
