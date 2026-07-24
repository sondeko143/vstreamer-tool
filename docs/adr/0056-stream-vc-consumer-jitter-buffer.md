# 0056. consumer 側の並べ替え/穴埋め/遅延上限を transport 非依存の jitter buffer に集約する

- Status: Accepted
- Date: 2026-07-24
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0051](0051-stream-transport-swappable-tiered.md), [0050](0050-streaming-vc-separate-subsystem.md), [0055](0055-stream-vc-producer-consumer-role-split.md), [0006](0006-clock-skew-threshold-warning.md)

## Context

M2 の consumer は `InProcessTransport` から `recv` して `drain_to_latest(keep=1)` で到着済みバックログを最新へ畳むだけで足りていた(同一プロセス内なので並べ替え・欠落は起きない, [0051](0051-stream-transport-swappable-tiered.md))。M3 の UDP transport([0051](0051-stream-transport-swappable-tiered.md) T3)では、並べ替え・欠落(パケット/IP フラグメント loss)・到着ジッタが実際に起きる。

spec の受入基準は「欠落・並べ替えが起きても再生側が検出して穴埋めし、無音の穴を黙って作らない」「変換が実時間に追いつかなくても遅延が単調増加しない」「網の jitter/遅延を実測し jitter buffer を調整する」を要求する。この責務をどこに置き、何をどう測るかを決める必要がある。

特に注意すべき制約:この 2 マシン(`.149`/`.150`)は W32Time 由来のクロック skew が既知で秒単位ずれることがある([[pipeline-latency-topology]] / [0006](0006-clock-skew-threshold-warning.md))。したがって cross-machine の**片道遅延の絶対値は skew に汚染され**、ジッタバッファの調整根拠に使えない。

## Decision

- 並べ替え・穴埋め・深さ・遅延上限を transport 実装から切り出し、**transport 非依存の `JitterBuffer` コンポーネント**(専用モジュール)に集約する。UDP transport の `recv()` は届いた datagram を `StreamPacket` にして返すだけにし、順序保証・concealment・深さ管理は `JitterBuffer` が持つ。CPU 単体テスト可能にする(合成 reorder/loss 列で検証、GPU/網 不要)。
- 深さは `jitter_buffer_ms`(既定は浅く、有線 LAN 想定で 1〜2 ブロック相当)。**深さ = 付加遅延**なので、実測ジッタから最小に詰める。起動時から適応制御はしない。
- 穴埋め = 期待 `seq` が出力クロックの締切までに来なければ、**直前ブロックを無音へフェードした concealment ブロック**を出す(ハードなゼロブロックではなく seam クリックを避ける)。期待 seq は進め、遅れて届いた分は drop する。捨てた/欠落は telemetry と seq(gap)簿記で必ず観測する(無音の穴を黙って作らない)。
- 遅延上限 = バッファが超過したら古い順に drop する(M2 の `drain_to_latest` の役割を `JitterBuffer` 内へ取り込む)。
- 計測は **skew 免疫の量のみ**:consumer の**到着間隔ジッタ**(単一クロック)と、**seq gap による loss/reorder**。片道遅延の絶対値は skew 汚染ゆえ主張しない。latency が要るなら clock 同期か RTT echo を別途用意する。
- M2 の `role = local` パスは `JitterBuffer` を通さず現状維持(byte-identity を保つ, [0050](0050-streaming-vc-separate-subsystem.md))。

## Alternatives rejected

- **ジッタ方針を各 transport 実装内に持つ** — transport ごとに reorder/conceal を再実装することになり、「他ロジックを変えず transport だけ差し替える」([0051](0051-stream-transport-swappable-tiered.md))を崩す。単一の transport 非依存コンポーネントの方が再利用でき、網/GPU 無しで単体テストできる。
- **起動時から適応ジッタバッファ**(到着統計で深さを自動調整) — 実測前に制御を入れると調整対象がブラックボックス化し、何が効いているか切り分けられない。まず固定の浅いバッファで測る([0051](0051-stream-transport-swappable-tiered.md) の measure-first に沿う)。適応化は必要が実測で示されてから。
- **cross-machine の片道遅延を主指標に測る** — この 2 機は W32Time skew が既知で秒単位ずれる([[pipeline-latency-topology]])。片道遅延は skew と交絡し、ジッタバッファ深さの調整根拠にならない。過去に skew が e2e テレメトリを汚染した実例がある([0006](0006-clock-skew-threshold-warning.md))。
- **穴埋めをゼロ(無音)ブロックにする** — 境界でハードな不連続になりクリックが乗る。直前ブロックのフェードで seam を隠す([0053](0053-streaming-vc-fixed-block-crossfade.md) の crossfade と同じ思想)。

## Consequences

- transport を UDP から将来 TCP/bidi に替えても、並べ替え/穴埋め/遅延方針は 1 箇所のまま不変。
- concealment・深さ・loss 検出を CPU で単体テストでき、実機/GPU 無しで回帰を張れる。
- 深さは付加遅延なので、既定は浅く保ち、実測ジッタから最小に詰める運用ステップ(M3 の実測)が要る。
- 片道遅延の数値は出さない(estimate 扱い)。将来 latency 数値が必要なら M1 の RTF budget + 実測 buffer 深さから積むか、clock 同期/RTT echo を足す。
- Status は `Accepted`(実装 + 実機実測が裏づけ)。final review が既定 `jitter_buffer_ms=0` での永久無音化バグ(空バッファでの conceal が cursor を進め in-flight packet を追い越す)を摘出→「実 loss(新 seq が在る)だけ advance、starvation は advance しない」に修正済み。
