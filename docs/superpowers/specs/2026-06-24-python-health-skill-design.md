# 設計: `python-health` skill — Python プロジェクト健全性担保

## 問題

品質ツール（ruff/ty/pytest+cov 等）はローカルに揃っているが、自動で強制する仕組みが無い。CI はほぼ空で、PR ゲートも pre-commit も存在しない。GitHub ホスト runner はこのプロジェクトの Windows 専用 wheel（release URL pin）と GPU 前提の E2E に合わず、依存も GPU も揃っているのは手元だけである。

## ゴール

依存も GPU も揃った手元で on-demand に回せる、再利用可能な健全性チェック skill。安い→高い順にゲート群を走破し、機械的な問題は自動修正し、本質的な問題は報告してトリアージ（原因切り分け＋修正案提示）する。

## 非ゴール

- GitHub Actions ワークフローの追加・更新。
- pre-commit / tox / nox の導入。
- カバレッジの数値ハードゲート（将来オプション）。
- 自動依存アップグレード（報告のみ）。

## 受入基準

- 依存が揃った手元で明示起動でき、対象パッケージ名を自動導出する（特定名をハードコードしない）。
- 静的（format/lint/型）→ テスト+カバレッジ → 依存/security の順にゲートを走破する。
- 機械的な修正（フォーマット・安全な lint 自動修正）だけを自動適用し、変更サマリを提示する。
- 型エラー・テスト失敗・脆弱性・dead code は自動修正せず報告する。
- 各ゲートが pass / fail / fixed / skipped / error に分類される。
- extra 未同期やゲート単体の失敗でも全体が死なず、残りを走破する。
- 型チェックが ty で行われる。
- カバレッジはハードゲートではなく、ベースライン比較の警告に留まる。
- 別の uv プロジェクトでも同様に動く。

---

- 決定根拠: [ADR-0010](../../adr/0010-python-health-on-demand-skill.md) , [ADR-0011](../../adr/0011-python-health-triage-hybrid.md) , [ADR-0012](../../adr/0012-adopt-ty-type-checker.md) , [ADR-0013](../../adr/0013-skills-at-user-level.md)
- 実装計画: [plan](../plans/2026-06-24-python-health-skill.md)
