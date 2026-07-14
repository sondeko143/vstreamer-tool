# 0012. 型チェッカに ty を採用し pyright を置換する

- Status: Accepted
- Date: 2026-06-24
- Related: [ADR-0010](0010-python-health-on-demand-skill.md) / spec [2026-06-24-python-health-skill-design](../superpowers/specs/2026-06-24-python-health-skill-design.md)

## Context

健全性 skill（[ADR-0010](0010-python-health-on-demand-skill.md)）の型チェックゲートに使う型チェッカを選ぶ必要がある。このプロジェクトのツールチェーンは既に Astral 製で揃っている（lint/format は ruff、パッケージ管理は uv）。型チェッカだけ別系統だとツールチェーンの一貫性を欠く。

## Decision

型チェッカに Astral の ty を採用し pyright を置換する。リポジトリ全体の型チェックは ty で行い、ruff/uv/ty で Astral ツールチェーンに揃える。

## Alternatives rejected

- **mypy / pyright** — 一般的で成熟しているが、Astral 系に揃わずツールチェーンの一貫性を欠く。

## Consequences

ruff/uv/ty で一貫し、設定・運用感覚が揃う。一方 ty はまだ新しく、診断の癖に追随が必要になる。
