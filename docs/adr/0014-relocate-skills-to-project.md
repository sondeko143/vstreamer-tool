# 0014. スキル資産をプロジェクトレベル（.claude/skills + scripts）へ移設する

- Status: Accepted
- Date: 2026-06-25
- Related: supersedes [ADR-0013](0013-skills-at-user-level.md)

## Context

[ADR-0013](0013-skills-at-user-level.md) で再利用性を優先してスキルを `~/.claude/skills/` にユーザーレベル配置したが、実運用でこれが裏目に出た。スキルが repo 外に散逸してバージョン管理から外れ、プロジェクト固有ゲート（poe check/fix）と乖離することが判明した。スキルの実行コードとそれが依存するプロジェクトゲートが別の場所に置かれると、片方だけ更新されて食い違う。

## Decision

python-health・code-metrics・startup-profile のスキル資産をプロジェクトレベルへ移設する。実行コードは `scripts/`、スキル定義は `.claude/skills/<name>/` に置き、poe タスク経由で走らせる。ユーザーレベルおよび外部コピーは削除し、単一ソース化する。

## Alternatives rejected

- **ユーザーレベル配置を維持（[ADR-0013](0013-skills-at-user-level.md)）** — repo 外への散逸・版管理外・プロジェクトゲートとの乖離が実際に起きたため却下。

## Consequences

スキルが単一ソースでプロジェクトに同梱され、リポジトリと同じ版管理下に入る。ゲート（poe check/fix）と常に同期する。一方で他プロジェクトへは持ち出せなくなり、プロジェクト固有ゲート前提に振り切る。
