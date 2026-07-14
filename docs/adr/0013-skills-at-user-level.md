# 0013. 再利用スキルをユーザーレベル ~/.claude/skills に配置する

- Status: Superseded by [ADR-0014](0014-relocate-skills-to-project.md)
- Date: 2026-06-24
- Related: spec [2026-06-24-python-health-skill-design](../superpowers/specs/2026-06-24-python-health-skill-design.md)

## Context

python-health（および同種の再利用スキル）は複数の Python リポジトリで使い回すことを狙う。スキルを特定リポジトリ内に閉じると、他プロジェクトから使えず再利用性が下がる。

## Decision

再利用性を優先し、python-health（および同種スキル）を `~/.claude/skills/` にユーザーレベル配置する。本リポジトリを最初の適用先・検証台とする。

## Alternatives rejected

- **プロジェクト内配置** — spec 時点では再利用性を優先したため不採用。

## Consequences

複数リポジトリでの再利用性は上がる。一方、スキルが repo と分離し、リポジトリとスキルのバージョン管理が別々になる。
