# 0032. GUI を複数 pipeline マネージャへ全面書き直しする（1 pipeline=1 サブプロセス=1 config=1 port）

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-gui-multi-pipeline-rewrite-design.md](../superpowers/specs/2026-07-12-gui-multi-pipeline-rewrite-design.md); [ADR-0025](0025-target-python-314-phased.md), [ADR-0033](0033-gui-manifest-versioning.md), [ADR-0034](0034-gui-corrupt-file-resilience.md)

## Context

現行 GUI は 1460 行の単一巨大クラスで、「単一 `Config` を全 worker タブで編集し、単一 vspeech サブプロセスだけを起動 / 停止する」構造になっている。複数 pipeline を並行管理できず、起動のたびに config パスの指定が要る。

## Decision

GUI を複数 pipeline マネージャへ全面書き直しし、**1 pipeline = 1 独立 vspeech サブプロセス = 1 専用 `Config` = 1 `listen_port`** を単位とする（固定 argv・shell なしの Popen）。純粋ロジック（paths / ports / migration / recipes / profile / process）と Tk UI 層に分割し、依存方向は gui → vspeech の一方向のみとする。旧 `gui.py` と未使用の `dummy_param.py` は削除する。空きポートはマニフェスト予約集合＋`SO_REUSEADDR` なしの OS bind 成否で判定し、起動直前に競合を再確認する。プリセットレシピにより新規 pipeline を最初から配線済み・実行可能にする。起動時は固定のユーザー設定場所（platformdirs）を常に読み、config パス引数を取らない。

## Alternatives rejected

- **単一クラスを漸進リファクタする** — 単一 `Config` / 単一プロセス前提が骨格に食い込んでおり、割に合わない。
- **マニフェスト予約ポートのみ回避する** — 外部プロセスが握るポートを検出できない。
- **pipeline を 1 サブプロセス以外の単位にする** — なし（1 プロセス = 1 config = 1 `listen_port` という既存アーキテクチャの単位に一致する唯一の現実解）。

## Consequences

pipeline 間が完全に分離し、純粋ロジックが Tk 非依存で TDD 可能になる。UI 層は手動 smoke 検証にとどまる。3.14 着地（[ADR-0025](0025-target-python-314-phased.md)）を前提とする（platformdirs の cp314・PEP 695）。
