# 0051. マシン間ストリーミングトランスポートを差し替え層とし段階昇格する

- Status: Proposed
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0050](0050-streaming-vc-separate-subsystem.md), [0004](0004-per-destination-sender-transport.md), [0035](0035-bound-sender-reconnect-backoff.md)

## Context

録音+VC マシンと再生マシンを分け、変換音声のみをマシン間に流す。許容遅延は対象 GPU の per-block RTF 実測後に確定する方針(現時点で未定)なので、トランスポートを先に一つへ決め打つと、実測結果次第で作り直しになる。候補は、既存 unary gRPC の流用、生ソケット、双方向ストリーミング RPC(wire 契約 `vstreamer-protos` の変更を要す)の 3 つで、遅延・実装規模・別リポ跨ぎのトレードオフが異なる。

## Decision

producer/consumer をトランスポート interface(`send`/`recv` と `StreamPacket{session_id, seq, pts, pcm, sample_rate, flags}`)の背後に置き、実装を段階昇格できるようにする。

- **T1 = 既存 unary gRPC 流用**(proto 変更なし)。まず動かして実測する用。
- **T3 = 生 UDP/TCP 音声チャネル**(in-repo、低オーバーヘッド)。低遅延の本命候補。
- **T2 = 双方向ストリーミング RPC**(`vstreamer-protos` の proto 変更、最良の順序/遅延)。

per-block RTF の実測結果に応じ、VC・再生・発話系の他ロジックを変えずにトランスポートだけ差し替える。

## Alternatives rejected

- **双方向ストリーミング RPC を最初から採用** — 別リポ `vstreamer-protos` の wire 契約変更(python/rust/go 各再生成)を実測前に払うことになり、遅延要件を満たすか未検証のまま重い投資になる。
- **unary gRPC に固定** — ブロック毎 unary のオーバーヘッドと `_send` の例外握り潰し(無音穴)を抱えたまま低遅延を追えない。後から差し替えられる構造でないと、実測が悪かったときに詰む。

## Consequences

「実測してから決める」に素直で、最小トランスポートで go/no-go を測り、必要時のみ昇格できる。`StreamPacket` が seq/pts を持つため、consumer 側で欠落検出・穴埋め・整列ができる(ジッタバッファ設計へ接続)。反面、interface と各実装の保守コストがかかり、T2 まで昇格する場合は別リポ跨ぎの変更が残る。現状は tier-0(同一プロセス内直結)のみ実装しており、「VC/再生/発話系の他ロジックを変えずにトランスポートだけ差し替えられる」という本 ADR の主張は 2 つ目の実装が出るまで未実証なので、Status は `Proposed` のままとする。
