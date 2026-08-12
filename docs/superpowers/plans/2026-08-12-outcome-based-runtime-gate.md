# ランタイムの軽さを成果で守る Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> または superpowers:executing-plans。実行の規律は implementer-led-execution に従う。

ADR: [0085](../../adr/0085-gate-runtime-weight-on-outcome.md) (Proposed / 既定)、
[0086](../../adr/0086-banned-dependencies-as-declared-data.md) (Proposed / 既定)

Spec: [2026-08-12-outcome-based-runtime-gate-design.md](../specs/2026-08-12-outcome-based-runtime-gate-design.md)

**Goal:** ランタイムの重さを、パッケージ名の列挙ではなく観測可能な成果（モジュール集合と
常駐メモリ）で守り、成果で捉えられない禁止だけを理由と失効条件つきの宣言データとして残す。

**Architecture:** 既存の `tests/test_forbidden_imports.py` は 3 つの異なる役目を 1 つの名前
リストで担っている — 重さの防止、バージョン下限の保護、監査対象面積の抑制。重さの分を子
プロセス測定へ移し、残りを `pyproject.toml` の宣言データへ移す。[ADR-0084](../../adr/0084-dependency-table-torch-gate.md)
の依存表ガードは、uv に解決時点で止める手段が無い以上そのまま必要なので維持する。

**Tech Stack:** Python 3.14 / uv / pytest / subprocess

## Global Constraints

- Python は **3.14 のみ**。依存操作は uv で行う。
- `uv sync --extra rvc` を単独で実行しない（他の extra が外れる）。**`uv sync --all-extras`**、
  ad-hoc 実行は `uv run --all-extras`。
- 新規の `#` コメントと docstring は**英語**。ユーザーが読む文字列（ログ・例外メッセージ・
  `config.py` の `description=`・argparse/click の help）は**日本語**。散文ドキュメント
  （ADR / spec / plan / README.md）は**日本語**。`CLAUDE.md` は全編英語なのでそのまま英語。
- import は 1 行 1 つ（ruff `force-single-line = true`）。
- pydantic v2 API のみ。
- GPU 対応 onnxruntime セッションは `vspeech/lib/onnx_session.py` の `create_session` から
  のみ開く。`vspeech/lib/vad.py` だけが意図的な例外。**ファクトリを二重化しない。**
- `vspeech/` は `fairseq` / `transformers` / `pydantic_settings` / `torch` / `torchaudio` を
  import しない。
- 検証コマンドの成否は**パイプを通さずに終了コードで判定する**。出力が要るならファイルへ
  リダイレクトする。pytest は完全な node ID で指定する。
- 性能・メモリの主張には **N を明示する**。1 回の実行は測定ではない。
- **起動時間を合否判定に使わない**（[ADR-0085](../../adr/0085-gate-runtime-weight-on-outcome.md)）。
  同一コードでテストスイートが 30.45s / 113.70s / 35.40s と振れた実測がある。
- 繰り越す指摘は、**該当コードの位置に理由つきコメントとして**残す。レポートだけに書かない。
- gitleaks ゲートが環境 PII（LAN IP、`C:\Users\<name>` パス、appkey）を弾く。

## Implementer Authority

この plan が拘束するのは 3 つだけ: **公開契約**（各 task の「契約」欄の名前・型・方向）、
**Global Constraints の逐語値**、**各 task の受入基準**。

それ以外 — 内部設計、関数・ファイルの分割、命名、アルゴリズム、テストの設計と粒度、
エラー処理の形、依存の使い方 — はすべて実装者が決める。plan に書かれていない実装を
選んだことは逸脱ではない。

この plan が参照する ADR のうち、実装者を拘束するのは **効力: 制約** のものだけ。**既定** と
**便宜** は「なぜ今こうなっているか」の記録であって、守る義務はない。より良い方法があれば
変えてよい。変えたら report の逸脱1行に書く。

plan の記述より良い方法を見つけたら、良い方を採る。plan は使い捨てなので書き換えない。
その選択が adr-writing の基準に当たるならトリガ2 で ADR を起票する。

停止して人間に確認するのは、上の 3 つのいずれかを**変える必要がある**と判断したときだけ。

---

### Task 1: 成果ゲートを敷く（ADR-0085）

**目的:** ランタイムの起動が実際に持ち込むモジュール集合と常駐メモリを、通常のテスト実行で
検査できるようにする。

**範囲:** `tests/`、必要なら `scripts/`。`vspeech/` は変更しない。

**契約**
- Produces: ランタイムのエントリポイントを**子プロセスで** import し、モジュール集合と常駐
  メモリを観測する手段。GPU・モデル資産・設定ファイルを必要としない。
- Produces: モジュール集合の基準データと常駐メモリの閾値。どちらも根拠（実測値と N）が
  同じ場所に併記される。
- Produces: 検査が失敗したとき、**増えたモジュールを名指しする**メッセージ。
- Consumes: 既存の `tests/test_forbidden_imports.py` の子プロセス方式（`sys.modules` を
  pristine な子で確認する形）。同じ理由が当てはまるので踏襲してよい。

**受入基準**
- [ ] 通常の `pytest` 実行でモジュール集合と常駐メモリが検査される。GPU も実行時資産も
      要求しない。
- [ ] 名前を列挙していない依存がランタイムの起動経路に入った場合にも検査が失敗する。
      これを**実際に混入させて確かめた**記録がある（混入は元に戻すこと）。
- [ ] 検査が失敗したとき、メッセージが増えたモジュールを名指しする。実際の失敗出力が
      記録されている。
- [ ] 常駐メモリの閾値に根拠がある。何回測った何の値に対してどれだけの余裕を取ったかが
      同じ場所に書かれている。
- [ ] 起動時間が合否判定に使われていない。
- [ ] 検査が連続実行で安定する。**同一コードで 5 回以上**走らせて揺れないことが記録されて
      いる（このゲートは既存スイートに常駐するので、暴れると無視されるようになる）。

**検証**
- `uv run --all-extras pytest -q --no-cov`（終了コードをファイル経由で直接判定）。
- 新しい検査だけを完全な node ID で指定した実行を 5 回以上。
- `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` の終了コードが 0。

**コミット単位:** 成果ゲートの追加で 1 コミット。

---

### Task 2: 禁止依存を宣言データへ移し、棚卸しする（ADR-0086）

**目的:** 成果で捉えられない禁止だけを、理由と失効条件つきの宣言データとして残す。

**範囲:** `pyproject.toml`、`tests/test_forbidden_imports.py`。

**契約**
- Consumes: Task 1 の成果ゲート（重さ由来の禁止がそちらで守られていること）。
- Produces: `pyproject.toml` の禁止依存テーブル。各エントリは対象・**理由**・**失効条件**を持つ。
- Produces: `tests/test_forbidden_imports.py` がその表を読む。対象名と理由がテスト側に
  重複して書かれていない。
- Produces: [ADR-0084](../../adr/0084-dependency-table-torch-gate.md) の依存表ガード
  （`torch` / `torchaudio` / `faiss-cpu` が pyproject と uv.lock の解決済み集合に現れない）が
  引き続き機能する。

**受入基準**
- [ ] 禁止エントリが理由と失効条件を伴う宣言データとして 1 箇所にあり、テストはそれを読む。
      同じ内容が 2 箇所に書かれていない。
- [ ] 現在の各エントリについて、その理由が**今も成立するかを実測で確かめ**、残したか外したかと
      判断の根拠が記録されている。少なくとも次の 2 件は実測で扱う:
      `transformers`（`uv audit` が現在どう応答するか）、`fairseq`（3.14 環境で解決できるか、
      解決すると何を道連れにするか）。
- [ ] 理由が成立しないエントリが残っていない。
- [ ] `torch` / `torchaudio` / `faiss-cpu` が依存表と解決済み集合の双方から排除されている
      ことが、引き続きテストで守られている。**この保護を弱めていないことを、実際に混入させて
      確かめた記録がある**（混入は元に戻すこと）。
- [ ] 宣言データの形式が壊れている場合（理由や失効条件の欠落など）、テストが fail loud する。

**検証**
- `uv run --all-extras pytest -q --no-cov`（終了コードをファイル経由で直接判定）。
- `uv lock --check` の終了コードが 0。
- `uv run ruff check .` / `uv run ty check` の終了コードが 0。

**コミット単位:** 宣言データへの移行で 1 コミット、棚卸しの結果で 1 コミット。

---

### Task 3: 実測の記録と ADR の昇格

**目的:** spec の受入基準すべてに実測の裏づけを与え、決定層を確定させる。

**範囲:** `docs/adr/`、`docs/adr/README.md`。

**契約**
- Consumes: Task 1 / Task 2 の記録。
- Produces: ADR-0085 / 0086 の Status が、実装の結果に応じて `Accepted` へ昇格するか、
  覆った場合は supersede される。

**受入基準**
- [ ] ランタイムのモジュール集合の大きさと常駐メモリの実測値が、**N とともに**記録されている。
- [ ] spec の受入基準 8 項目すべてについて、満たしたことの根拠（実測値または検証コマンドの
      終了コード）が対応づけて記録されている。満たせなかった項目があれば、その項目と理由が
      明示されている。
- [ ] ADR-0085 / 0086 の Status が `Proposed` のまま残っていない。
- [ ] 昇格前に、各 ADR の本文に後付けの訂正が積まれたまま残っていない（`Accepted` になると
      本文は不変になるため）。
- [ ] `docs/adr/README.md` の索引の Status 列が各 ADR と一致している。

**検証**
- `uv run --all-extras pytest -q --no-cov` の終了コードが 0。
- `uv run poe check` の終了コードが 0（既知の受容済み指摘を除く。除いたものは列挙する）。
- 記録した実測値と ADR の記述を突き合わせ、乖離がないことを確認する。

**コミット単位:** ADR の Status 昇格と索引更新で 1 コミット。
