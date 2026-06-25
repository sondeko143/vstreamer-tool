# sender 宛先ごと並列化＋チャネル永続再利用 設計書

- 日付: 2026-06-14
- ステータス: 承認済み（実装計画へ）
- ブランチ: `perf/sender-per-destination-transport`
- 前提: 別セッションで A（モデルウォームアップ）/ C（whisper のイベントループ閉塞解消）/ D（マルチGPU整合）等は実装済み。本タスクは改善案 **F（transport 最適化）** のうち、ユーザ選択により **「宛先ごと並列化（順序保証つき）」** と **「チャネル永続再利用」** の2点のみを対象とする。**音声トリムの一般化は対象外**（既存の SUBTITLE 限定トリムは現状維持）。

## 1. 背景と目的

`vspeech` は録音→whisper→rvc→再生をプロセス分割し、各ホップを gRPC `Command` で接続する。実測（`voice_2026_06_14.log` ほか）では各ホップの送受信は 7〜30ms で、**転送自体は総遅延の約2〜3%**にすぎない。ただし現在の sender には以下の構造的無駄があり、tail 遅延と音声経路の遅延に寄与する:

1. **送信が完全直列**: [sender.py](../../../vspeech/worker/sender.py) の `sender` ループは `for remote in worker_output.remotes: await send_command(...)` で1宛先ずつ逐次送信。同一発話内で **playback-host（字幕）への送信が localhost（vc）への送信をブロック**する。字幕送信が遅いと、その分だけ音声経路（vc→playback）の開始が遅れる。
2. **チャネルを毎回張り直し**: `send_command` が `async with get_channel(...)` で送信ごとにチャネルを生成・破棄。生成コストと初回接続ハンドシェイク、さらに相手が一瞬落ちている時の再接続失敗（冷却時 ~2.26s, `voice_2026_06_14.log:7-13`）を毎回払う。

目的: 宛先間の送信を**並行化**して相互ブロックを解消しつつ、同一宛先の**FIFO順序を厳守**（音声の順序逆転を防止）。あわせて宛先ごとに gRPC チャネルを**1本張りっぱなし**にして再接続コストを排除する。

## 2. 非目標（YAGNI / スコープ外）

- 音声トリムの一般化（テキストのみ消費するチェーン宛 Command からの生音声除去）。**既存の SUBTITLE 限定トリム（[sender.py:80-81](../../../vspeech/worker/sender.py#L80-L81)）はそのまま維持**する。
- silence-gate 短縮・ストリーミング認識（改善案 G）。
- whisper / rvc / GPU 周りの変更（別セッション済み）。
- 送信のバッチ化・圧縮・チャンク分割。

## 3. 設計

### 3.1 構成

sender を「ディスパッチャ」と「宛先ごとの送信タスク」に分離する。

- **`RemoteSender`（新規・小クラス, sender.py 内）**
  - フィールド: `remote: str`, `credentials: GcpIDTokenCredentials | None`, `queue: asyncio.Queue[Command]`（`maxsize=REMOTE_QUEUE_MAXSIZE`, 既定 16）, `channel: grpc.aio.Channel | None = None`。`maxsize` はコンストラクタ引数（テストで小さく差し替え可能）。
  - `async def run(self)`: `try: while True: command = await self.queue.get(); await self._send(command)` / `finally: if self.channel is not None: await self.channel.close()`
  - `async def _send(self, command)`: `self.channel` が `None` なら `get_channel(self.remote, self.credentials)` で生成して保持。`stub = CommanderStub(self.channel)` を都度生成（軽量）。既存と同じログ（`send: s(...), t(...), to ...` / `success response: ...`）と **SUBTITLE トリム**を踏襲。例外は現状と同じ集合（`RefreshError` / `MutualTLSChannelError` / `AioRpcError`）を捕捉してログ＆**継続**（チャネルは保持＝gRPC が次回 RPC で自動再接続）。
- **`sender`（改修）**
  - `credentials = get_id_token_credentials(...)` は従来どおり1回取得。
  - `async with asyncio.TaskGroup() as send_tg:` の中で `senders: dict[str, RemoteSender] = {}` を保持。
  - `while True: worker_output = await in_queue.get()` で取り出し、`for remote in worker_output.remotes:`
    - `remote` が空 → 従来どおりローカル即時ディスパッチ（`WorkerInput.from_output` → `process_command`、`EventDestinationNotFoundError` 捕捉も維持）。
    - 非空 → `rs = senders.get(remote)`、無ければ生成して `send_tg.create_task(rs.run(), name=f"sender:{remote}")` し `senders[remote]=rs`。続けて `rs.enqueue(worker_output.to_pb(remote=remote))`（満杯時は最古を破棄、3.5）。
  - `except CancelledError as e: raise shutdown_worker(e)` は維持。

### 3.2 順序保証（最重要）

- ディスパッチャは `sender_queue` から **到着順**に `WorkerOutput` を処理し、各宛先キューへ `put_nowait` する。各宛先キューは **単一の `RemoteSender` が単独消費**するため、同一宛先への `Command` は投入順＝送信順（FIFO）。→ 音声の順序逆転は発生しない。
- 異なる宛先は別タスク・別キューで**並行**。ある宛先の `await stub.process_command` での停止は、その宛先タスクのみをブロックし、他宛先の送信は進む。

### 3.3 ライフサイクル / シャットダウン（フェイルファスト対策込み）

- 全 `RemoteSender` は sender 内のネスト `TaskGroup` 配下。main の TaskGroup が sender をキャンセルすると、ディスパッチャの `await in_queue.get()` が `CancelledError` を送出 → ネスト `TaskGroup.__aexit__` が全宛先タスクをキャンセル → 各 `run` の `finally` で `channel.close()` → `CancelledError` が再送出され `shutdown_worker` でラップ。
- **フェイルファストの巻き込み回避（リスク A）**: `TaskGroup` は子1つの未捕捉例外で全子＋ディスパッチャをキャンセルし、`ExceptionGroup`（≠`CancelledError`）で終了するため、放置すると sender 全体が道連れになり `except CancelledError` を素通りして main へ生伝播する。これを防ぐため、`_send` は既知例外（`RefreshError`/`MutualTLSChannelError`/`AioRpcError`）に加えて**広く `Exception` を捕捉してログ＆継続**し、宛先タスクを決して死なせない（`CancelledError` は再送出してキャンセルは正しく伝える）。fire-and-forget の転送として、1件の Command 失敗でプロセス全体を巻き込まない。
- **チャネルクローズのガード（リスク D）**: 各 `run` の `finally` での `await self.channel.close(grace=...)` はキャンセル進行中に走るため `try/except`（`Exception` を握りつぶし）で保護する。

### 3.4 既存挙動の保持（不変条件）

- `get_channel` は**無改修**（チャネル工場として再利用）。secure(GCP) パスは audience が remote 由来で宛先ごとに固定のため、宛先単位キャッシュは安全（トークン更新は `AuthMetadataPlugin` が都度実施）。
- SUBTITLE 音声トリム、ローカルディスパッチ、`EventDestinationNotFoundError` ログ、エラー時 fire-and-forget（Command を捨ててプロセスは生存）を維持。
- `worker_output.to_pb(remote)` はディスパッチャ側で実施（現状と同位置）。

### 3.5 宛先キュー方針：上限付き＋満杯で最古を破棄（リスク B）

詰まった宛先（例：再生機 playback-host が一時停止）で陳腐な音声を溜め込まないよう、各宛先キューは **`maxsize` 付き**とし、満杯時は**最古の Command を捨てて最新を残す**。

`RemoteSender.enqueue(command)`:

```python
try:
    self.queue.put_nowait(command)
except QueueFull:
    try:
        self.queue.get_nowait()  # 最古を破棄
        logger.warning("drop oldest command for %s (queue full)", self.remote)
    except QueueEmpty:
        pass
    self.queue.put_nowait(command)
```

- 安全性: 生産者（ディスパッチャ）と消費者（当該 `RemoteSender.run`）は同一イベントループ上の単一コルーチン同士で、`get_nowait`→`put_nowait` 間に `await` を挟まないため**アトミック**。`QueueFull`/`QueueEmpty` 競合は起きない。
- 既定 `REMOTE_QUEUE_MAXSIZE = 16`（実時間パイプラインで宛先あたり数発話分。発話単位の Command なのでフレーム欠落ではなく「古い発話の丸ごと破棄」になり、再生の鮮度を優先）。

## 4. テスト方針（TDD, `tests/`）

`asyncio_mode = "auto"`。フェイクの `CommanderStub` / `get_channel` を `monkeypatch` し、実 gRPC を張らずに検証する。

1. **ディスパッチ正当性**: 2つの remote を持つ `WorkerOutput` を投入 → 各 remote の `RemoteSender.queue` に、その remote 向け chains のみを含む `Command` が積まれる。
2. **順序（FIFO）**: 同一 remote 宛の連続2 `WorkerOutput` → 当該 remote の送信順が投入順と一致。
3. **宛先間の切り離し（非ブロック）**: remote A のフェイク送信を `asyncio.Event` で待たせる → remote B 宛の送信が A の完了を待たずに実行される（B 完了を待ち受けて assert、タイムアウト付き）。
4. **チャネル再利用**: 同一 remote へ複数回送信 → `get_channel` の呼び出しが remote ごと**1回**（mock 呼び出し回数）。
5. **既存保持**:
   - SUBTITLE 始まりの Command で `operand.sound.data` が空化される。
   - 空 remote はローカル `process_command` 経路（put_queue）へ流れる。
   - `_send` が `AioRpcError` を捕捉してタスク継続（後続送信が成功する）。
6. **満杯時 drop-oldest（3.5）**: `maxsize=2` の `RemoteSender` に 3 件 `enqueue` → キューには新しい 2 件のみ残り、最古が破棄される（消費を止めた状態で検証）。
7. **広域例外でタスクが死なない（リスク A）**: `_send` が `ValueError` 等を投げても `run` ループが継続し、後続 `enqueue` が処理される。

既存の `tests/test_event_chains.py`（ルーティング/`to_pb`/`from_output`）はグリーン維持を回帰条件とする。

## 5. リスクと緩和

| ID | リスク | 緩和 |
| --- | --- | --- |
| A | 子タスクの未捕捉例外でネスト `TaskGroup` ごと巻き込み停止（`ExceptionGroup` が `except CancelledError` を素通り） | `_send` で `Exception` を広く捕捉してログ＆継続（`CancelledError` のみ再送出）。テスト7で担保（3.3）。 |
| B | 詰まった宛先のキューが無制限増加＝バックプレッシャ無し／陳腐な音声を再生 | キュー方針を選択（3.5）。 |
| C | remote ごとの恒久タスク/チャネル/キューが無制限増加（remote 文字列が無数に変化する場合のリーク） | トポロジ固定で実害なし（YAGNI）。注記のみ。将来は上限/アイドル退避を検討。 |
| D | シャットダウン時 `finally` の `await channel.close()` が二重キャンセルで例外化 | `try/except` でガードし握りつぶす（3.3）。 |
| E | 並列化で音声が順序逆転 | 同一宛先＝単一消費者キューで FIFO 厳守（3.2）。テスト2で担保。 |
| F | secure チャネルのトークン期限切れ | `AuthMetadataPlugin` が都度更新。宛先単位 audience 固定で安全（3.4）。 |
| G | 効果が小さい（転送は総遅延の2〜3%） | 既知の前提。主目的は tail 遅延と「字幕送信が音声経路をブロックする」現象の解消。定常遅延の短縮は狙いではない。 |

## 6. 影響範囲

- 変更: [vspeech/worker/sender.py](../../../vspeech/worker/sender.py)（`sender` 改修、`RemoteSender` 追加、`send_command` は `RemoteSender._send` へ吸収し削除）。
- 追加: sender 用テスト（新規ファイル、例: `tests/test_sender.py`）。
- 無改修: `get_channel`、`shared_context.py`、`command.py`、receiver、各 worker。
