# 設計: `python-health` skill — Python プロジェクト健全性担保

- 日付: 2026-06-24
- 対象: `vstreamer-tool`(`vspeech` パッケージ)を最初の適用先・検証台とする、再利用可能な健全性チェック skill
- ステータス: 設計合意済み(実装プラン待ち)

## 1. 背景と目的

### 1.1 ゴール(合意済み)
「Python プロジェクトの健全性担保」の手法を **調査 → このrepoに適用 → skill化** まで一気通貫で行う。
最終成果物は **GitHub Actions ではなく、この開発環境(Claude Code / 依存もGPUも揃った手元)で on-demand に回せる健全性チェック skill**。

GitHub ホスト runner を使わない理由はこのプロジェクト固有の制約に合致する:
- **Windows 専用 wheel**(`torch` cu128, `voicevox-core`, `pyvcroid2`, `fairseq` …)が release URL 固定で pin されている。
- **GPU 前提の E2E**(VOICEVOX/RVC、`voicevox_e2e` マーカー)があり、GitHub ホスト runner には GPU が無い。
- 実際に依存もGPUも揃っているのは「この環境」。よって skill として手元で回す方が実用的。

### 1.2 現状(調査結果)
- ローカルツールは `pyproject.toml` に設定済み: `ruff`(lint `I`/`UP` + isort `force-single-line`)、`ty`(型)、`pytest`(`asyncio_mode=auto`、`addopts = "-m 'not voicevox_e2e'"`)、`pytest-cov` / `pytest-grpc` / `pytest-asyncio` / `pytest-httpx`。
- テストは `tests/` に 23 ファイル。
- CI はほぼ空: 唯一の workflow が `.github/workflows/codeql.yml`(しかも `checkout@v3` / `codeql-action@v2` と旧版 pin)。
- **ruff / ty / pytest を自動実行するゲートが存在しない**。PR ゲートも pre-commit も無い。カバレッジツールはあるが閾値/レポート運用なし。
- Dependabot アラートは稼働(commit `4404123`)だが `dependabot.yml` は in-repo に無い。

→ 要するに「品質ツールはローカルに揃っているが、自動で強制する仕組みが無い」のが本質的ギャップ。本 skill はそのギャップを「手元で on-demand に回せるゲート」で埋める。

## 2. 採用アプローチ: 案A トリアージ型ハイブリッド

「**機械的なものは直す、本質的なものは(直さず)上げる**」。

1回の起動でゲートを **安い→高い順**(静的 → テスト/カバレッジ → 依存/security)に実行し、原則として全ゲートを走破(collect-all)。
- **決定的に安全な修正は自動適用**: `ruff format`、`ruff check --fix`(safe fix のみ)。
- **本質的な失敗は適用せず報告 + トリアージ**: `ty` 型エラー、テスト失敗、`pip-audit` 脆弱性、dead code。Claude が原因を切り分け、修正案を提示して承認を仰ぐ。
- 最後に **構造化サマリ**(各ゲートの pass/fail・所要・指摘件数・自動修正の有無)を出す。

### 却下した代替案
- **案B レポート専用 doctor**: 一切変更せず pass/fail のみ。最も安全だが Claude の修正能力を活かせず往復が増える。
- **案C 全自動リメディエーションループ**: 緑になるまで自動修正反復。型/テストを通すためにロジックを自動書き換えすると**本質的な問題を隠蔽**しうるため非推奨。

## 3. skill の形・配置

- **ユーザレベル** `~/.claude/skills/python-health/` に配置(複数の Python repo で再利用するため)。このrepoが最初の適用先・検証台。
- 構成(writing-skills 準拠):
  - `SKILL.md` — frontmatter(`name`, `description`)+ Claude が従う手順・トリアージ方針・チェックリスト。
  - `scripts/health.py` — ゲートを順に実行し **構造化レポート(JSON + 人間可読)** を出すオーケストレータ。`uv run python` で起動。Python にする理由: Windows(PowerShell)と Bash の両方で動き、Python プロジェクトと相性が良く、コマンド合成/出力パースをスクリプトに閉じ込めて決定的にできる。
  - `references/gate-catalog.md` — 各ゲートの定義・自動修正可否・除外チューニング(bandit/vulture の false positive 対策)を分離。

## 4. ゲート・パイプライン

順序は安い→高い。既定は全ゲート走破。`--fail-fast` 指定時は最初の失敗で停止。

| 順 | ゲート | コマンド(uv 前提) | 自動修正 |
|---|---|---|---|
| 0 | 環境検出 | `pyproject.toml` + `uv.lock` 確認、設定済みツール(ruff/ty/pytest addopts)を読取り | — |
| 1a | format | `uv run ruff format --check .` | ✅ 失敗時 `ruff format .` 適用 |
| 1b | lint | `uv run ruff check .` | ✅ `--fix`(safe のみ、`--unsafe-fixes` は使わない) |
| 1c | 型 | `uv run ty check` | ❌ 報告のみ |
| 2 | test+cov | `uv run pytest --cov=<pkg> --cov-report=term-missing`(addopts の e2e 除外を尊重) | ❌ 報告のみ |
| 3a | lock 整合 | `uv lock --check` | ❌ 報告のみ |
| 3b | 脆弱性 | `uvx pip-audit`(`uv export` 経由、依存を汚さない) | ❌ 報告のみ |
| 3c | 古い依存 | `uv pip list --outdated`(best-effort、advisory) | ❌ 報告のみ |
| 4a | security lint | `uvx bandit -r <pkgs>`(test 除外チューニング) | ❌ 報告のみ |
| 4b | dead code | `uvx vulture <pkg>`(低信頼は advisory) | ❌ 報告のみ |

## 5. 自動修正ポリシー(案A の肝)

- **自動適用してよい** = 決定的・可逆・ロジック非改変:
  - `ruff format`、`ruff check --fix`(safe のみ)。
  - 適用後は **変更サマリ(diff 件数 / 対象ファイル)を提示**し、いつでも revert 可能であることを明示。
- **絶対に自動適用しない** = 本質判断が要る:
  - `ty` 型エラー、テスト失敗、`pip-audit` 脆弱性、依存アップグレード、bandit/vulture 由来の削除。
  - → Claude が原因を切り分け、**修正案を提示して承認を仰ぐ**(ハイブリッドの境界)。

## 6. カバレッジの扱い

- いきなりハード失敗にはしない。
- 初回に **現状値をベースラインとして記録**(`references/coverage-baseline.json` 等)し、以降は **ベースラインを下回ったら警告**。
- `--cov-fail-under=<N>` は将来オプション(数値ゲートは後から段階導入)。

## 7. エラーハンドリング / 環境耐性

- **extra 未同期は「skip(理由付き)」**: `audio` / `whisper` / `rvc` 等が無くても skill 全体は死なない。pytest は元々 e2e 除外、import 依存は丁寧に skip 表示。
- **ゲート間は独立**: 1 つが失敗/ネットワークエラー(pip-audit 等)でも残りは走り、その項目は「error / inconclusive」と記録。
- **OS 差**: Windows 専用依存は非 Windows で graceful degrade。
- 総合ステータスは集約するが **走破優先**(`--fail-fast` 時のみ早期終了)。
- 各ゲートは「pass / fail / fixed / skipped / error」のいずれかに分類して報告。

## 8. 汎用化(再利用)

- **generic-first**: `<pkg>` 等は `pyproject.toml`(`[tool.uv.build-backend] module-name` / `[project.scripts]` / packages)から自動導出。`vspeech` をハードコードしない。
- 任意の uv プロジェクトで動く。`vstreamer-protos/python` を 2 例目の検証に使う。

## 9. 検証(skill 自体のテスト)

1. **このrepoで実走**: addopts 尊重・全ゲート実行・有害な自動修正ゼロ・レポート生成を確認。
2. **意図的に壊して分類確認**: scratch に format 崩れ / lint 違反 / 型エラー / 失敗テストを仕込み、skill が正しく検出し「自動修正 vs 要承認」に振り分けるか確認。
3. **2 例目で汎用化確認**: `vstreamer-protos/python` で `<pkg>` 自動導出と全ゲートが動くか確認。

## 10. 実例(調査: 実際の Python エコシステムでの相当物)

各ゲートは「業界で確立した健全性プラクティス」を手元の skill に写したもの:

| 本 skill のゲート | 実世界の相当プラクティス |
|---|---|
| ruff format/lint 自動修正 | Astral(ruff/uv)自身の運用、pre-commit.ci の「mechanical は自動修正、ロジックはゲートして人に返す」思想 |
| ty 型チェック | mypy/pyright を CI ゲートに置く一般的構成(本 skill は ty を採用) |
| pytest + coverage ベースライン | `pytest --cov` + Codecov/`--cov-fail-under` の段階導入、scientific-python エコシステムの coverage 運用 |
| uv lock --check | lockfile が pyproject と整合しているかを CI で検査する標準パターン |
| pip-audit | PyCQA `pip-audit` による既知 CVE スキャン(Dependabot のローカル版的位置づけ) |
| bandit / vulture | PyCQA `bandit`(security lint)、`vulture`(dead code 検出)を補助ゲートに置く構成 |

全体としては「pre-commit / tox / nox で束ねるローカルゲート」を、**Claude Code skill として実行+推論で直せる形に再構成**したもの。

## 11. スコープ外(YAGNI)

- GitHub Actions ワークフローの追加・更新(CodeQL の action 版上げ含む)。今回は手元 skill が目的。
- pre-commit / tox / nox の導入(skill がその役割を担う)。
- カバレッジの数値ハードゲート(将来オプション)。
- 自動依存アップグレード(報告のみ)。

## 12. 実装の次ステップ

成果物が skill であるため、実装は **writing-skills** スキルの規約に沿って行う(SKILL.md frontmatter、scripts、references の構成)。詳細手順は writing-plans で実装プランに落とす。
