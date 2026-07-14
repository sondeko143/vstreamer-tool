# 0004. sender を宛先ごと並行・永続チャネル・有界キューに再構成する

- Status: Accepted
- Date: 2026-06-14
- Related: spec [2026-06-14-sender-per-destination-transport-design](../superpowers/specs/2026-06-14-sender-per-destination-transport-design.md)

## Context

現在の sender は `for remote in worker_output.remotes: await send_command(...)` で宛先を完全直列に送信するため、同一発話内で playback-host（字幕）への送信が localhost（vc）への送信をブロックし、音声経路の開始を遅らせる。さらに `send_command` は送信ごとに gRPC チャネルを張り直し、生成コストと再接続失敗（冷却時 ~2.26s）を毎回払う。転送自体は総遅延の約2〜3%だが、tail 遅延と音声経路の遅延に寄与している。

## Decision

sender をディスパッチャ化し、宛先ごとに専用 FIFO キュー＋単一消費タスクを持たせて宛先間を並行化する（同一宛先は投入順を厳守）。gRPC チャネルは宛先ごとに1本張りっぱなしで再利用する。各宛先キューは有界（既定 `REMOTE_QUEUE_MAXSIZE=16`）とし、満杯時は最古の Command を破棄して鮮度を優先する。送信は fire-and-forget とし、`CancelledError` 以外を広く捕捉してログ＆継続し、1件の失敗でプロセスを巻き込まない。対象は改善案 F のうち「宛先ごと並列化」と「チャネル永続再利用」の2点に限定する。

## Alternatives rejected

- **直列送信の維持** — 字幕送信が音声経路をブロックする現象が残る。
- **全宛先を単一キュー** — 宛先間を切り離せない。
- **送信ごとにチャネルを張り直す** — 再接続コスト（冷却時 ~2.26s）を毎回払う。
- **無制限キュー / drop-newest** — 陳腐な音声を溜め込む、または鮮度優先の方針に反する。
- **既知例外のみ捕捉** — `ExceptionGroup` で sender 全体が停止し main へ生伝播する。
- **音声トリムの一般化・remote 無制限増加対策も同時実施** — スコープ外／トポロジ固定で YAGNI。

## Consequences

宛先間ブロックが消え、同一宛先の順序保証と再接続コスト排除が得られる。詰まった宛先では古い発話が失われる。宛先が動的に無数化した場合のリソースリークは将来課題として残る。
