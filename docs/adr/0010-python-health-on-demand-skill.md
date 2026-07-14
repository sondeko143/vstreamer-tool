# 0010. 健全性ゲートを GitHub Actions でなく手元 on-demand skill にする

- Status: Accepted
- Date: 2026-06-24
- Related: spec [2026-06-24-python-health-skill-design](../superpowers/specs/2026-06-24-python-health-skill-design.md)

## Context

品質ツール（ruff/ty/pytest+cov 等）はローカルには揃っているが、自動で強制する仕組みが無い。CI はほぼ空で、PR ゲートも pre-commit も存在しない。一方このプロジェクトは GitHub ホスト runner と相性が悪い固有制約を持つ。`torch`(cu128) / `voicevox-core` / `pyvcroid2` / `fairseq` などが release URL 固定で pin された Windows 専用 wheel であり、VOICEVOX/RVC の E2E（`voicevox_e2e` マーカー）は GPU 前提だが GitHub ホスト runner に GPU が無い。依存も GPU も実際に揃っているのは「この開発環境（手元）」である。

## Decision

健全性ゲート群（ruff format/lint, ty, pytest+cov, `uv lock --check`, uv audit, bandit, vulture）を、GitHub ホスト runner ではなく、依存も GPU も揃った手元で明示起動する再利用可能な on-demand skill として実装する。対象パッケージ名は `pyproject.toml`（build-backend `module-name` → `[project.scripts]` → packages → 正規化 `[project].name`）から自動導出し、`vspeech` などの特定名をハードコードしない。bandit/vulture は AST を解析するため `uvx` ではなく `uv run --with` でプロジェクトインタプリタ上で走らせる。

## Alternatives rejected

- **GitHub Actions CI** — Windows 専用 wheel が release URL pin で入らず、GPU が無く E2E が走らない。手元で回すほうが実用的。
- **bandit/vulture を `uvx` 実行** — プロジェクト構文（3.11 `except*` 等）を解釈できない Python が選ばれうる。対照的に code-metrics の lizard/complexipy は自前パーサなので `uvx` 可。

## Consequences

「押せば回る」CI ではなく、開発者が明示起動する運用になる。プロジェクト venv を前提にするぶん環境依存は増えるが、依存も GPU も揃った場所で解析が確実に走る。extra 未同期は「skip（理由付き）」として skill 全体は死なせない。
