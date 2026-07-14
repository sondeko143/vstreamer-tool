# 0035. sender の永続チャネルの再接続バックオフを有界化する

- Status: Accepted
- Date: 2026-07-15
- Related: [ADR-0004](0004-per-destination-sender-transport.md)（refine）

## Context

[ADR-0004](0004-per-destination-sender-transport.md) で sender は宛先ごとに gRPC チャネルを
1 本張りっぱなしで再利用する構成にした（`RemoteSender`）。チャネル生成時にオプションを一切
渡していないため、再接続バックオフは gRPC の既定
（`min_reconnect_backoff_ms=20s` / `max_reconnect_backoff_ms=120s`）のままだった。

これにより「receiver より先に sender を起動して初回接続に失敗する」冷起動で、致命的に遅い
復帰が起きる:

- 接続失敗でサブチャネルが `TRANSIENT_FAILURE` に入り、指数バックオフ（試行間 1s→最大 120s）で
  次の接続を待つ。`min_reconnect_backoff_ms` は 1 回の接続試行のデッドライン下限でもあり、
  SYN が黙殺される LAN 越し（`WSA 10060 Connection timed out`）だと各試行が最大 ~20s 張り付く。
- バックオフ待ちの間に来た RPC は `wait_for_ready=False`（gRPC 既定・fire-and-forget 方針）なので
  即失敗し、**キャッシュ済みの前回接続エラー文字列をそのまま返す**。よって receiver が起動済みでも
  同じ 10060 エラーが数十秒〜最大 2 分ログに出続ける。永続チャネルなので新規チャネル生成で
  バックオフがリセットされることもない。

実測（`wait_for_ready=False` の実 RPC を一定間隔で投げる sender 忠実再現、localhost）: 既定では
サーバ起動後さらに 6〜8s（試行間バックオフのみ）UNAVAILABLE が続いた。バックオフ上限を絞ると
0.5s で復帰した。実 LAN の 10060 では各試行の ~20s 張り付きも加わるため、既定の実待ちはさらに長い。

## Decision

`get_channel` がチャネルを開くとき、`insecure_channel` と `secure_channel`
（`async_secure_authorized_channel` 経由）の両方に有界な再接続バックオフを渡す。値は
`vspeech/worker/sender.py` の定数 `RECONNECT_CHANNEL_OPTIONS`:

- `grpc.initial_reconnect_backoff_ms = 500`
- `grpc.min_reconnect_backoff_ms = 1000`（1 試行のデッドライン下限。10060 の張り付きを 20s→1s に）
- `grpc.max_reconnect_backoff_ms = 5000`（試行間バックオフ上限を 120s→5s に）

永続チャネル再利用（ADR-0004）は維持する。冷起動からの最悪復帰は「数十秒〜最大 2 分」から
「最大 ~6s」に落ちる。`tests/test_sender.py` が両経路でバックオフが既定より十分小さく渡ることを
構造的に固定する。

## Alternatives rejected

- **`wait_for_ready=True` を RPC に付ける** — RPC が接続確立までブロックし、`RemoteSender` の単一
  消費ループと有界 drop-oldest キュー（ADR-0004）を塞ぐ。鮮度優先・fire-and-forget の方針に反する。
- **失敗ごとにチャネルを破棄・再生成してバックオフをリセットする** — 永続チャネル（ADR-0004、
  「再接続コスト排除」）に逆行し、障害中はコマンド毎に接続をやり直すスラッシングを生む。
  バックオフ有界化だけで復帰は十分速く、設計を壊さない。
- **バックオフ値を config に露出する** — 環境ごとの調整余地はあるが、現トポロジでは固定の妥当値で
  足り、config スキーマ・GUI 引き渡し・テストに波及する。YAGNI。必要になれば後続 ADR で足す。
- **既定のまま放置** — receiver 後起動という日常運用で数十秒〜2 分エラーが出続け、体感で「壊れている」。

## Consequences

冷起動（sender 先行）からの復帰が秒オーダーになり、receiver 起動後に古い接続エラーが出続ける
現象が実質消える。トレードオフとして、本当に長時間ダウンしている宛先へは最大 5s 間隔で接続を
試み続ける（既定 120s 間隔より頻繁）。宛先は固定・少数で、失敗は握り潰してログするだけ（ADR-0004）
なので実害は無視できる。値はコード定数で、環境依存の調整が要るようになれば config 露出を別途検討する。
