# 0008. 複雑度を lizard+complexipy の2レンズで測り radon/MI/wily/SaaS を却下する

- Status: Accepted
- Date: 2026-06-24
- Related: [ADR-0009](0009-code-metrics-advisory-only.md) / spec [2026-06-24-code-metrics-insight-skill-design](../superpowers/specs/2026-06-24-code-metrics-insight-skill-design.md)

## Context

「どこをリファクタすれば効くか」を答えるには、複雑度を単一指標で測るのが不十分である。循環的複雑度（CCN）は分岐が多いが平坦な dispatcher を過剰に警告し、認知的複雑度はネストや壊れた制御フローの「本質的な絡まり」を捉える。両者は別物を測るため、どちらか一方では「分岐は多いが読める」コードと「本当に絡まった」コードを取り違える。加えて計測ツール選定にはこのリポジトリ固有の制約がある。`pyproject.toml` の `[tool.uv.sources]` にある pin URL は `%2B` を含み、`configparser` ベースの設定読取りを行うツールはこれで即クラッシュする。ツールは `main.py` の 3.11 `except*` を解釈できる必要もある。

## Decision

複雑度を2つの相補的レンズで測る。lizard で循環的複雑度 CCN（＝独立経路数、テスト容易性の代理）と行数・トークン・引数、complexipy で認知的複雑度（＝可読性の代理）を得る。両者を（正規化パス, 単純名）キーで結合し、`Class::method` の接頭辞は剥がす。行番号を持つ lizard を結合の主にする。両ツールとも自前トークナイザを持ちプロジェクト import も `pyproject` 読取りも行わないため `uvx` で走らせる。complexipy は絵文字 stdout が cp932/cp1252 コンソールでクラッシュするため、常に `-q --output-format json --output <file>` で JSON をファイル出力し（さらに `PYTHONIOENCODING=utf-8`）、stdout を解析しない。結果は総合グレードを出さず、両レンズの高低から both-high／high-CCN-only／high-cognitive-only のバケットラベルで示す。（後日 beniget ベースの DepDegree＝def-use 結合を第3レンズ `dep` として追加したが、多レンズ方針は不変。）

## Alternatives rejected

- **radon** — `pyproject` の pin URL（`%2B`）で `configparser` が invalid interpolation でクラッシュし、`uvx` でも非プロジェクトインタプリタが選ばれ `except*` を invalid syntax 扱いする。かつメンテナンスモード（最終 6.0.1, 2023）。lizard が CCN+LOC を代替でき脆くないため全面的に不採用。
- **単一レンズ** — 高CCN低cognitive の dispatcher と、高cognitive の本質的な絡まりを取り違える。
- **Maintainability Index** — 恣意的・未較正の定数、LOC 支配、平均化でホットスポットを隠す（radon 自身が experimental と明記）。
- **wily（git 履歴トレンド）** — stale（2023）・クリーンツリー要求・履歴が重く、小規模単独リポジトリに過剰。
- **SaaS（SonarQube/Qlty/Code Climate）** — 公開しない小規模単独リポジトリには過剰。
- **complexipy の stdout を直接パース** — Windows 日本語コンソールで絵文字により `UnicodeEncodeError` クラッシュ。

## Consequences

2ツールの出力を結合する必要と、Windows encoding 回避（JSON ファイル読取り＋`PYTHONIOENCODING`）が常に要る。総合グレードは出さずバケットラベルで「どれだけ悪いか」を示すため、判定でなく材料として読ませる運用になる。complexipy には行番号が無く関数を `Class::method` で名付けるため、行番号は lizard 側から補う。
