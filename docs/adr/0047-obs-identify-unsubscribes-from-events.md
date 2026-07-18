# 0047. Identify で eventSubscriptions=0 を明示し、読まないイベントを OBS に送らせない（0043 を refine）

- Status: Accepted
- Date: 2026-07-18
- Related: [ADR-0043](0043-obs-websocket-client-in-house.md)（自前クライアント）, [ADR-0042](0042-subtitle-obs-failure-tiers.md)（失敗の階層化）

## Context

localhost の OBS に繋いでいるのに、subtitle worker が不定期に

```
subtitle worker [obs] cannot reach ws://127.0.0.1:4455
  (sent 1011 (internal error) keepalive ping timeout; no close frame received)
```

で再接続していた。ループバック相手に「ping が timeout した」というログは、
ネットワークの問題としては読めない。実際これはネットワークではなく、こちら側が
自分で接続を切っていた（`sent 1011` = 送ったのは自分）。

[ADR-0043](0043-obs-websocket-client-in-house.md) はこのクライアントを
「Hello/Identify/Identified と Request/RequestResponse だけ。イベント購読は
使わない」と決めたが、その決定は **OBS に伝えられていなかった**。
obs-websocket は `eventSubscriptions` が省略されると
`EventSubscription::All` を既定にする（プロトコル文書の Identify:
`"eventSubscriptions": number(optional) = (EventSubscription::All)`）。
つまり「使わない」と決めたイベントを、こちらは購読し続けていた。

そこから先は websockets のバックプレッシャで説明が付く:

1. OBS が op 5 のイベントを push する。
2. こちらが socket を読むのは `ObsWsClient.request()` が応答を待っている間だけ。
   それ以外の時間、`_run_session` は自分の `in_queue` で止まっていて読まない。
3. 未読フレームが 16（websockets の `max_queue` 既定値）を超えると Assembler が
   `transport.pause_reading()` を呼ぶ。
4. 読み取りが止まると **あらゆる** フレームの解析が止まる — Pong も含めて。
5. `keepalive()` は `ping_timeout` 内に pong を受け取れず、自ら
   1011 "keepalive ping timeout" で接続を切る。

実測ログはこの説明とすべて整合する: 字幕 push のたびに `request()` が
バックログを読み捨てるので、発話が続いている間（間隔 44 秒・2 分 10 秒）は健全
なまま、長い沈黙（6 分以上）を挟んだ次の push でだけ死んでいた。

fake OBS サーバでの対照実験でも再現・切り分けを確認した:
イベントあり + `max_queue=16` は production と同一のエラー文言で死に、
`max_queue=None`（バックプレッシャ除去）でも `eventSubscriptions=0`
（イベント除去）でも生き残る。

## Decision

`identify()` が送る Identify に `eventSubscriptions: 0`
（`EventSubscription::None`）を常に載せる。ADR-0043 が決めた「イベントは
使わない」を、こちら側の都合ではなく **ワイヤ上の契約** として OBS に伝える。

## Alternatives rejected

- **`max_queue=None` で接続を開く（バックプレッシャを外す）** —
  対照実験では確かに生き延びるが、生き延びる理由が「未読イベントを無制限に
  バッファし続けるから」で、誰も読まない以上プロセスの生存時間に比例して
  メモリが増える。切断をメモリリークに置き換えただけ。
- **バックグラウンドで socket を drain し続けるタスクを足す** —
  websockets は `recv()` の並行呼び出しを禁じており、`request()` 自身が
  `recv()` するため実際に `ConcurrencyError: cannot call recv while another
  coroutine is already running recv` で落ちる（実測）。回避するには
  「1 本の読み取りループ + requestId によるディスパッチ」へクライアントを
  作り替える必要があり、使わないイベントのために ADR-0043 が意図的に避けた
  複雑さを丸ごと持ち込むことになる。
- **`ping_interval`/`ping_timeout` を伸ばす、または keepalive を切る** —
  症状（timeout の検出）を消すだけで原因（読まれないフレームの滞留）は残る。
  keepalive を切れば、今度は本当に死んだ接続を検出できなくなる。

## Consequences

- 起動直後から OBS はこのクライアントにイベントを一切送らなくなり、
  `request()` の応答待ちループが読み捨てる無関係フレームも無くなる。
  沈黙の長さに関係なく接続が維持される。
- 将来このクライアントがイベントを使いたくなったら、購読の追加と
  「誰がいつ socket を読むか」の設計変更は不可分になる。`eventSubscriptions`
  だけ広げると、本 ADR が消したはずの滞留がそのまま戻る。
- `max_queue` は既定（16）のまま据え置く。バックプレッシャ自体は正しい機構で、
  外すべきは滞留の原因であって上限ではない。
- 回帰は `tests/test_obs_ws.py::test_identify_unsubscribes_from_all_events` が
  Identify の payload そのものを固定して守る。
