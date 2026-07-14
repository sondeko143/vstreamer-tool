# 0015. 起動プロファイラに py-spy を spawn+--subprocesses+--idle で使う

- Status: Accepted
- Date: 2026-06-25
- Related: [ADR-0014](0014-relocate-skills-to-project.md) / spec [2026-06-25-startup-profile-skill-design](../superpowers/specs/2026-06-25-startup-profile-skill-design.md)

## Context

`python -m vspeech` の起動が最初のワーカーがログを出すまで数秒スタールすることがあり、原因はワーカー起動時の同期的なブロッキング待ち（DNS/socket/TLS、SMB、subprocess）に散在していた（例: `google_auth_default()` が `gcloud` サブプロセスへ ~2s 費やす）。一つずつ潰すのではなく、起動時間の内訳を再現可能に捕捉・解析し、ブロッキング待ち・import・モデルロードを区別したい。かつ観測はアプリを一切計装しない方針とする（code-metrics と同じ「source を触らない」哲学）。環境固有の障害として、uv の `.venv\Scripts\python.exe` は CPython コピーではなく trampoline のため py-spy が「Failed to find python version」でバージョン判定できず直接アタッチできない。また Python はブロッキング I/O 中に GIL を解放するため、既定の on-CPU サンプリングでは狙いの待ちを取り逃す。

## Decision

起動レイテンシ調査に、アプリを一切計装しない外部サンプリング profiler py-spy を採用し speedscope JSON を出力する。trampoline 問題を避けるため spawn モード（`py-spy record … -- <python> -m vspeech …`）で走らせ、`--subprocesses`（trampoline が起動する真の基底インタプリタに追従）と `--idle`（GIL 解放中のブロッキング待ち＝DNS/socket/subprocess を捕捉）を必須とする。当初計画の `--pid` アタッチ＋明示ツリーkill は廃止し（py-spy が子ツリーを reap する）、アナライザ側でヘッドライン集計から idle（asyncio イベントループ待ち・parked thread、grpc poller 等）を除外する。

## Alternatives rejected

- **viztracer + アプリ内 exit フック** — アプリコードに計装が要り「source を触らない」方針に反する。
- **cProfile** — async/ブロッキング待ちのフレームグラフ表現が弱い。
- **pyinstrument** — in-process で動くが、アナライザが解釈できない `evented` speedscope 変種を吐く。
- **`--pid` アタッチ＋明示 tree-kill** — trampoline でバージョン判定不能。プロセス管理も余分（py-spy が子ツリーを reap する）。

## Consequences

ソース無改変で可視化（speedscope.app）と機械解析を両立できる。一方 `--idle` が拾う idle ノイズ（イベントループ待ち・parked thread）をアナライザで除去する必要が生じる。実測ではこのホストの起動ブロッカーが `gcloud` 認証サブプロセス（`subprocess.communicate` ~1.9s）であり、メタデータ HTTP プローブではないと判明した。
