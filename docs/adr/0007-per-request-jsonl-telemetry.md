# 0007. 発話ごとテレメトリを 1レコード=1段・opt-in・耐障害で JSONL 追記する

- Status: Accepted
- Date: 2026-06-23
- Related: spec [2026-06-23-per-request-jsonl-telemetry-design](../superpowers/specs/2026-06-23-per-request-jsonl-telemetry-design.md) / [ADR-0005](0005-true-e2e-in-process-telemetry.md)

## Context

[ADR-0005](0005-true-e2e-in-process-telemetry.md) のテレメトリは各段の処理時間をメモリ集計して終了時にサマリ出力するだけで、1発話ごとの粒度はメモリ常駐にとどまる。発話ごとの各段所要時間を機械可読で永続化し、後から `trace_id` で全段・全マシンを join して分析できるようにしたい。段は別プロセス／別マシンに分散している。

## Decision

1レコード＝1段（各プロセスが自段完了時に `ts`/`trace_id`/`stage`/`dur_s`/`pid` を1行 JSONL 追記し、分析時に `trace_id` で join）を採る。既定オフ（`jsonl_path` 空＝無効）のオプトインとし、書込は発話レート前提でイベントループ上の同期 write→flush（クラッシュ耐性優先、キュー化しない）で行う。出力先の mkdir/open は耐障害化し、既存 `log_file` も同様に保護して、到達不能な UNC 共有でも起動・処理を継続する。

## Alternatives rejected

- **1リクエスト＝全段まとめた単一行** — 段→所要時間の wire 伝搬と protos の再変更が必要になる。
- **既定オン** — 不要な同期 I/O を全実行に課す。
- **非同期キュー化** — 発話レートが低く YAGNI。
- **mkdir/open を無保護のまま** — ネットワーク不通時に起動がクラッシュする。
- **ファイルロック／プロセス別パス強制** — 小サイズ追記はほぼ原子的で pid で識別可能、過剰。

## Consequences

protos 無改修で分散に強い。使う環境だけが同期 I/O コストを払う。ネットワーク不通でも stdout と処理は生存し、ファイル/JSONL 出力のみが失われうる。
