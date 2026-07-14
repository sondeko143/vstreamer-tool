# 0009. code-metrics は advisory 専用としゲートしない

- Status: Accepted
- Date: 2026-06-24
- Related: [ADR-0010](0010-python-health-on-demand-skill.md)（ゲート群の担当） / spec [2026-06-24-code-metrics-insight-skill-design](../superpowers/specs/2026-06-24-code-metrics-insight-skill-design.md)

## Context

code-metrics スキルは「どこをリファクタすれば効くか」を答えるランキング材料である。同じスキルにコミット/CI をブロックするゲート機能を持たせると、advisory ランキングとしての価値が薄れ、循環的複雑度のゲーティングという別役割と重複する。循環的複雑度のゲートは ruff の `C901` + `[tool.ruff.lint.mccabe] max-complexity` の一行設定で足り、それは python-health のゲート群（[ADR-0010](0010-python-health-on-demand-skill.md)）の `ruff lint` ゲートが担う。

## Decision

code-metrics スキルは advisory 専用にする。非ゼロの「失敗」ステータスを返さず、コミット/CI をブロックせず、コードも編集しない（report-only）。循環的複雑度のゲーティングは持たず python-health（[ADR-0010](0010-python-health-on-demand-skill.md)）へ委ねる。閾値バンド（`ccn_warn=10` / `ccn_high=20` / `cog_warn=15`）はゲートではなくランキングの色分け（バケット振り分け）にのみ効き、CLI で上書き可能とする。

## Alternatives rejected

- **ここでもゲートする** — python-health と役割が重複し、advisory ランキングとしての価値を薄める。循環的複雑度のゲートは ruff `C901` の一行で足りる。

## Consequences

判断は常に人間に返る（ランキングは判定でなく材料）。閾値はバケットの色分けに使うだけなので、閾値変更が「合否」を動かすことはない。
