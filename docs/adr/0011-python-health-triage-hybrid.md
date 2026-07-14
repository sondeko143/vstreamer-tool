# 0011. 健全性の修正はトリアージ型ハイブリッド（ruff の2種のみ自動）にする

- Status: Accepted
- Date: 2026-06-24
- Related: [ADR-0009](0009-code-metrics-advisory-only.md) / [ADR-0010](0010-python-health-on-demand-skill.md) / spec [2026-06-24-python-health-skill-design](../superpowers/specs/2026-06-24-python-health-skill-design.md)

## Context

健全性 skill（[ADR-0010](0010-python-health-on-demand-skill.md)）の修正方針を決める必要がある。全自動で緑になるまで直すと、型やテストを通すためにロジックを自動書き換えし、本質的な問題を隠蔽しうる。逆に一切変更しないレポート専用だと Claude の修正能力を活かせず往復が増える。修正には「決定的で可逆・ロジック非改変のもの」と「本質判断が要るもの」が混在する。

## Decision

「機械的なものは直す、本質的なものは（直さず）上げる」トリアージ型ハイブリッドを採る。決定的・可逆・ロジック非改変の修正、すなわち `ruff format` と `ruff check --fix` の safe fix のみ（`--unsafe-fixes` は不使用）を自動適用する。型エラー・テスト失敗・脆弱性（pip-audit）・dead code は自動修正せず、原因を切り分けて修正案を提示し承認を仰ぐ。各ゲートは pass/fail/fixed/skipped/error のいずれかに分類する。カバレッジはハードゲートにせず、初回値をベースラインとして記録し以降はそれを下回ったら警告するに留める（`--cov-fail-under=N` は将来オプション）。

## Alternatives rejected

- **レポート専用 doctor** — 一切変更せず pass/fail のみ。最も安全だが Claude の修正能力を活かせず往復が増える。
- **全自動リメディエーションループ** — 緑になるまで自動修正を反復。型/テストを通すためのロジック自動書換で本質的問題を隠蔽しうる。
- **初手から `--cov-fail-under=N` ハードゲート** — 導入初日から赤で機能しない。段階導入が現実的。

## Consequences

自動修正の境界が「決定的・可逆・ロジック非改変」に明確化される。適用後は変更サマリ（diff 件数/対象ファイル）を提示し revert 可能性を明示する。本質的な失敗は承認フローを通すため、そのぶんの往復は残る。
