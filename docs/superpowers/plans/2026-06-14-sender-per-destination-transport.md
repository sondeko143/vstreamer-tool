# sender 宛先ごと並列化＋チャネル永続再利用 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** sender を宛先ごとの送信タスク＋永続チャネルに作り替え、宛先間の送信を並行化（同一宛先は FIFO 厳守）しつつ再接続コストを排除する。

**Architecture:** `sender` をディスパッチャ化し、`sender_queue` から取り出した `WorkerOutput` を宛先ごとの `RemoteSender`（専用 `asyncio.Queue` ＋張りっぱなしの gRPC チャネル）へ振り分ける。各 `RemoteSender.run` はネスト `TaskGroup` 配下で並行動作。宛先キューは上限付きで満杯時に最古を破棄。子タスクは `_send` 内で広く例外を捕捉して死なせない。

**Tech Stack:** Python 3.11（`asyncio.TaskGroup` / `except*`）、grpc.aio、protobuf（vstreamer-protos）、pytest（`asyncio_mode = "auto"`）、uv / ruff / ty。

設計の出典: [docs/superpowers/specs/2026-06-14-sender-per-destination-transport-design.md](../specs/2026-06-14-sender-per-destination-transport-design.md)

---

## File Structure

- Modify: `vspeech/worker/sender.py`
  - 追加: 定数 `REMOTE_QUEUE_MAXSIZE`、クラス `RemoteSender`、関数 `_dispatch_output`。
  - 改修: `sender`（TaskGroup＋ディスパッチャ化）。
  - 削除: `send_command`（責務は `RemoteSender._send` へ吸収）。
  - 無改修: `get_channel`、`async_secure_authorized_channel`。
- Create: `tests/test_sender.py`（`RemoteSender` / `_dispatch_output` / `sender` の単体・結合テスト）。
- 無改修: `vspeech/shared_context.py`、`vspeech/lib/command.py`、receiver、各 worker。

---

## Task 1: RemoteSender スケルトン＋上限付きキュー（drop-oldest）

**Files:**
- Modify: `vspeech/worker/sender.py`
- Test: `tests/test_sender.py`（新規）

- [ ] **Step 1: 失敗するテストを書く（テストファイル新規作成・共通ヘルパー込み）**

`tests/test_sender.py`:

```python
import asyncio
import contextlib
from asyncio import Queue
from uuid import uuid4

from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import VC
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operand
from vstreamer_protos.commander.commander_pb2 import Operation
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2 import OperationRoute
from vstreamer_protos.commander.commander_pb2 import Response
from vstreamer_protos.commander.commander_pb2 import Sound

import vspeech.worker.sender as sender_mod
from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.exceptions import WorkerShutdown
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerOutput
from vspeech.worker.sender import RemoteSender


def make_command(op: Operation = VC, data: bytes = b"abc") -> Command:
    return Command(
        chains=[OperationChain(operations=[OperationRoute(operation=op)])],
        operand=Operand(
            sound=Sound(
                data=data, rate=16000, format=SampleFormat.INT16, channels=1
            ),
            text="t",
        ),
    )


def make_output(followings, sound: bool = True, text: str = "hi") -> WorkerOutput:
    return WorkerOutput(
        input_id=uuid4(),
        followings=followings,
        sound=SoundOutput(
            data=b"abc", rate=16000, format=SampleFormat.INT16, channels=1
        )
        if sound
        else None,
        text=text,
    )


def fake_transport(monkeypatch, process):
    """get_channel / CommanderStub を差し替える。

    process: async fn(channel, command) -> Response。
    返り値 state["channels"] は get_channel 呼び出しで作られた FakeChannel 列、
    state["commands"] は process_command に渡った (remote, command) 列。
    """
    state = {"channels": [], "commands": []}

    class FakeChannel:
        def __init__(self, remote):
            self.remote = remote
            self.closed = False

        async def close(self, grace=None):
            self.closed = True

    def fake_get_channel(remote, credentials):
        ch = FakeChannel(remote)
        state["channels"].append(ch)
        return ch

    class FakeStub:
        def __init__(self, channel):
            self.channel = channel

        async def process_command(self, command):
            state["commands"].append((self.channel.remote, command))
            return await process(self.channel, command)

    monkeypatch.setattr(sender_mod, "get_channel", fake_get_channel)
    monkeypatch.setattr(sender_mod, "CommanderStub", FakeStub)
    return state


def test_enqueue_drops_oldest_when_full():
    rs = RemoteSender(remote="//r", credentials=None, maxsize=2)
    rs.enqueue(make_command(data=b"1"))
    rs.enqueue(make_command(data=b"2"))
    rs.enqueue(make_command(data=b"3"))  # 満杯 → 最古(b"1")を破棄
    assert rs.queue.qsize() == 2
    assert rs.queue.get_nowait().operand.sound.data == b"2"
    assert rs.queue.get_nowait().operand.sound.data == b"3"


def test_enqueue_preserves_fifo_order():
    rs = RemoteSender(remote="//r", credentials=None, maxsize=8)
    rs.enqueue(make_command(data=b"1"))
    rs.enqueue(make_command(data=b"2"))
    assert rs.queue.get_nowait().operand.sound.data == b"1"
    assert rs.queue.get_nowait().operand.sound.data == b"2"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: FAIL（`ImportError: cannot import name 'RemoteSender'`）

- [ ] **Step 3: 最小実装（定数＋クラス＋enqueue）**

`vspeech/worker/sender.py` の import 群に追加（一行ずつ・ruff `force-single-line`）:

```python
from asyncio import QueueEmpty
from asyncio import QueueFull
from grpc.aio import Channel
```

既存 import 群の直後（`get_channel` 定義より前で可）に追加:

```python
REMOTE_QUEUE_MAXSIZE = 16


class RemoteSender:
    def __init__(
        self,
        remote: str,
        credentials: GcpIDTokenCredentials | None,
        maxsize: int = REMOTE_QUEUE_MAXSIZE,
    ):
        self.remote = remote
        self.credentials = credentials
        self.queue: Queue[Command] = Queue(maxsize=maxsize)
        self.channel: Channel | None = None

    def enqueue(self, command: Command):
        try:
            self.queue.put_nowait(command)
        except QueueFull:
            try:
                self.queue.get_nowait()  # 最古を破棄
                logger.warning(
                    "drop oldest command for %s (queue full)", self.remote
                )
            except QueueEmpty:
                pass
            self.queue.put_nowait(command)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/sender.py tests/test_sender.py
git commit -m "feat(sender): add RemoteSender with bounded drop-oldest queue"
```

---

## Task 2: RemoteSender._send（チャネル再利用・SUBTITLEトリム・広域例外）

**Files:**
- Modify: `vspeech/worker/sender.py`
- Test: `tests/test_sender.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_sender.py` に追記:

```python
async def _ok(channel, command):
    return Response(result=True)


async def test_send_reuses_channel(monkeypatch):
    state = fake_transport(monkeypatch, _ok)
    rs = RemoteSender(remote="//r", credentials=None)
    await rs._send(make_command())
    await rs._send(make_command())
    assert len(state["channels"]) == 1  # チャネルは1回だけ生成
    assert len(state["commands"]) == 2  # 送信は2回


async def test_send_trims_sound_for_subtitle(monkeypatch):
    state = fake_transport(monkeypatch, _ok)
    rs = RemoteSender(remote="//r", credentials=None)
    await rs._send(make_command(op=SUBTITLE, data=b"loud"))
    assert state["commands"][0][1].operand.sound.data == b""


async def test_send_swallows_errors_and_continues(monkeypatch):
    async def boom(channel, command):
        raise ValueError("boom")

    state = fake_transport(monkeypatch, boom)
    rs = RemoteSender(remote="//r", credentials=None)
    await rs._send(make_command())  # 例外を外に漏らさない
    await rs._send(make_command())
    assert len(state["commands"]) == 2  # 2回とも試行された
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_sender.py -k "send_reuses or trims or swallows" -q`
Expected: FAIL（`AttributeError: 'RemoteSender' object has no attribute '_send'`）

- [ ] **Step 3: 最小実装（_send を RemoteSender に追加）**

`vspeech/worker/sender.py` の `RemoteSender` に メソッド追加:

```python
    async def _send(self, command: Command):
        try:
            if self.channel is None:
                self.channel = get_channel(self.remote, self.credentials)
            stub = CommanderStub(self.channel)
            logger.info(
                "send: s(%s), t(%s), to %s",
                len(command.operand.sound.data),
                command.operand.text,
                self.remote,
            )
            if command.chains[0].operations[0].operation == SUBTITLE:
                command.operand.sound.data = b""
            logger.debug("send: chains(%s)", command.chains)
            res = cast(Response, await stub.process_command(command))
            logger.info("success response: %s", str(res))
        except (RefreshError, MutualTLSChannelError, AioRpcError) as e:
            logger.warning("%s", e)
        except Exception as e:  # noqa: BLE001 - 宛先タスクを死なせない
            logger.warning("send error to %s: %s", self.remote, e)
```

注: `except Exception` は `CancelledError`（`BaseException` 派生）を捕捉しないため、キャンセルは正しく伝播する。

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/sender.py tests/test_sender.py
git commit -m "feat(sender): RemoteSender._send with channel reuse and resilient errors"
```

---

## Task 3: RemoteSender.run（消費ループ＋宛先間の非ブロック）

**Files:**
- Modify: `vspeech/worker/sender.py`
- Test: `tests/test_sender.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_sender.py` に追記:

```python
async def test_run_processes_commands_in_order(monkeypatch):
    state = fake_transport(monkeypatch, _ok)
    rs = RemoteSender(remote="//r", credentials=None)
    rs.enqueue(make_command(data=b"1"))
    rs.enqueue(make_command(data=b"2"))
    task = asyncio.create_task(rs.run())
    try:
        for _ in range(200):
            if len(state["commands"]) >= 2:
                break
            await asyncio.sleep(0.005)
        assert [c.operand.sound.data for _, c in state["commands"]] == [b"1", b"2"]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_blocked_remote_does_not_block_others(monkeypatch):
    a_started = asyncio.Event()
    a_finished = asyncio.Event()
    b_done = asyncio.Event()
    block = asyncio.Event()

    async def process(channel, command):
        if channel.remote == "A":
            a_started.set()
            await block.wait()  # 永久にブロック
            a_finished.set()
        else:
            b_done.set()
        return Response(result=True)

    fake_transport(monkeypatch, process)
    a = RemoteSender(remote="A", credentials=None)
    b = RemoteSender(remote="B", credentials=None)
    a.enqueue(make_command())
    b.enqueue(make_command())
    ta = asyncio.create_task(a.run())
    tb = asyncio.create_task(b.run())
    try:
        await asyncio.wait_for(b_done.wait(), timeout=2)
        assert a_started.is_set()
        assert not a_finished.is_set()  # A はまだ詰まっている
        assert b_done.is_set()  # それでも B は完了
    finally:
        for t in (ta, tb):
            t.cancel()
        for t in (ta, tb):
            with contextlib.suppress(asyncio.CancelledError):
                await t
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_sender.py -k "run_processes or blocked_remote" -q`
Expected: FAIL（`AttributeError: 'RemoteSender' object has no attribute 'run'`）

- [ ] **Step 3: 最小実装（run を RemoteSender に追加）**

`vspeech/worker/sender.py` の `RemoteSender` にメソッド追加:

```python
    async def run(self):
        try:
            while True:
                command = await self.queue.get()
                await self._send(command)
        finally:
            if self.channel is not None:
                try:
                    await self.channel.close()
                except Exception as e:  # noqa: BLE001 - クローズ失敗は無視
                    logger.debug("channel close error for %s: %s", self.remote, e)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/sender.py tests/test_sender.py
git commit -m "feat(sender): RemoteSender.run consumer loop with guarded channel close"
```

---

## Task 4: _dispatch_output（宛先振り分け＋ローカルディスパッチ）

**Files:**
- Modify: `vspeech/worker/sender.py`
- Test: `tests/test_sender.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_sender.py` の import に追記:

```python
from vspeech.worker.sender import _dispatch_output
```

`tests/test_sender.py` に追記:

```python
def test_dispatch_routes_each_remote_to_own_sender():
    context = SharedContext(config=Config())
    senders: dict[str, RemoteSender] = {}
    spawned: list[RemoteSender] = []
    output = make_output(
        [
            [EventAddress(event=EventType.subtitle, remote="//playback-host:8080")],
            [
                EventAddress(event=EventType.vc, remote="//localhost:8084"),
                EventAddress(event=EventType.playback, remote="//playback-host:8083"),
            ],
        ]
    )
    _dispatch_output(context, senders, None, spawned.append, output)
    assert set(senders) == {"//playback-host:8080", "//localhost:8084"}
    assert len(spawned) == 2
    sub_cmd = senders["//playback-host:8080"].queue.get_nowait()
    vc_cmd = senders["//localhost:8084"].queue.get_nowait()
    assert sub_cmd.chains[0].operations[0].operation == SUBTITLE
    assert vc_cmd.chains[0].operations[0].operation == VC


def test_dispatch_empty_remote_uses_local_process_command():
    context = SharedContext(config=Config())
    worker = context.add_worker(event=EventType.subtitle, configs_depends_on=[])
    senders: dict[str, RemoteSender] = {}
    output = make_output(
        [[EventAddress(event=EventType.subtitle)]], sound=False, text="hello"
    )
    _dispatch_output(context, senders, None, lambda rs: None, output)
    assert senders == {}  # リモート送信は発生しない
    wi = worker.in_queue.get_nowait()
    assert wi.current_event == EventType.subtitle
    assert wi.text == "hello"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_sender.py -k "dispatch" -q`
Expected: FAIL（`ImportError: cannot import name '_dispatch_output'`）

- [ ] **Step 3: 最小実装（_dispatch_output を追加）**

`vspeech/worker/sender.py` の import に追記（一行ずつ）:

```python
from typing import Callable
```

`sender` 関数の直前に追加:

```python
def _dispatch_output(
    context: SharedContext,
    senders: dict[str, RemoteSender],
    credentials: GcpIDTokenCredentials | None,
    spawn: Callable[[RemoteSender], None],
    worker_output: WorkerOutput,
):
    for remote in worker_output.remotes:
        if remote:
            rs = senders.get(remote)
            if rs is None:
                rs = RemoteSender(remote=remote, credentials=credentials)
                spawn(rs)
                senders[remote] = rs
            rs.enqueue(worker_output.to_pb(remote=remote))
        else:
            for worker_input in WorkerInput.from_output(
                output=worker_output, remote=remote
            ):
                process_command(context=context, request=worker_input)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: PASS（9 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/sender.py tests/test_sender.py
git commit -m "feat(sender): _dispatch_output fans out per-remote, preserves local dispatch"
```

---

## Task 5: sender 改修（TaskGroup 化）＋ send_command 削除＋仕上げ

**Files:**
- Modify: `vspeech/worker/sender.py:66-117`（`send_command` 削除、`sender` 改修）
- Test: `tests/test_sender.py`

- [ ] **Step 1: 失敗する結合テストを書く**

`tests/test_sender.py` に追記:

```python
async def test_sender_dispatches_to_remote_end_to_end(monkeypatch):
    done = asyncio.Event()
    received: list[tuple[str, Command]] = []

    async def process(channel, command):
        received.append((channel.remote, command))
        done.set()
        return Response(result=True)

    fake_transport(monkeypatch, process)
    monkeypatch.setattr(sender_mod, "get_id_token_credentials", lambda gcp: None)
    context = SharedContext(config=Config())
    in_queue: Queue = Queue()
    await in_queue.put(
        make_output([[EventAddress(event=EventType.vc, remote="//localhost:8084")]])
    )
    task = asyncio.create_task(sender_mod.sender(context, in_queue))
    try:
        await asyncio.wait_for(done.wait(), timeout=2)
        assert received[0][0] == "//localhost:8084"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, WorkerShutdown):
            await task
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_sender.py -k "end_to_end" -q`
Expected: FAIL（現行 `sender` は逐次 `send_command` 実装で、`get_channel`/`CommanderStub` 差し替えはされるが、本テストは新 `sender` の TaskGroup 経路を前提とするため、まず赤を確認してから置換する）

- [ ] **Step 3: `send_command` を削除し `sender` を置換**

`vspeech/worker/sender.py` の `send_command`（関数定義全体）を削除し、`sender` を以下へ置換:

```python
async def sender(
    context: SharedContext,
    in_queue: Queue[WorkerOutput],
):
    credentials = get_id_token_credentials(context.config.gcp)
    logger.info("sender worker started")
    try:
        async with TaskGroup() as send_tg:
            senders: dict[str, RemoteSender] = {}

            def spawn(rs: RemoteSender):
                send_tg.create_task(rs.run(), name=f"sender:{rs.remote}")

            while True:
                try:
                    worker_output = await in_queue.get()
                    _dispatch_output(
                        context=context,
                        senders=senders,
                        credentials=credentials,
                        spawn=spawn,
                        worker_output=worker_output,
                    )
                except EventDestinationNotFoundError as e:
                    logger.warning("unsupported event: %s", e)
    except CancelledError as e:
        raise shutdown_worker(e)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_sender.py -q`
Expected: PASS（10 passed）

- [ ] **Step 5: 回帰・静的チェック（全テスト＋ruff＋ty）**

Run:
```bash
uv run pytest -q
uv run ruff format .
uv run ruff check .
uv run ty check
```
Expected: 全テスト PASS（既存 `tests/test_event_chains.py` 等も green）、ruff/ty とも報告ゼロ。
注: `cast` / `Response` / `SUBTITLE` / `urlparse` の import は `_send` / `get_channel` で引き続き使用される。未使用 import が出たら ruff 指示に従い除去。

- [ ] **Step 6: コミット**

```bash
git add vspeech/worker/sender.py tests/test_sender.py
git commit -m "feat(sender): per-destination concurrent dispatch via TaskGroup; drop send_command"
```

---

## Self-Review（計画作成者によるチェック結果）

**1. Spec coverage:**
- 3.1 構成（RemoteSender / dispatcher）→ Task 1〜5。
- 3.2 順序保証 → Task 1（FIFO enqueue）／Task 3（単一消費者 run）／テスト `test_run_processes_commands_in_order`。
- 3.3 ライフサイクル・フェイルファスト回避（リスクA）→ Task 2（`except Exception`）／Task 3（finally guard, リスクD）／テスト `swallows_errors` `blocked_remote`。
- 3.4 既存挙動保持 → Task 2（SUBTITLE トリム）／Task 4（ローカルディスパッチ・`EventDestinationNotFoundError`）／Task 5（`sender` の except 維持）。
- 3.5 上限付き drop-oldest（リスクB）→ Task 1／テスト `drops_oldest_when_full`。
- §4 テスト 1〜7 → Task 1〜5 のテストへ全て対応（1=Task4 routing, 2=Task1/3 FIFO, 3=Task3 decoupling, 4=Task2 channel reuse, 5=Task2/4 既存保持, 6=Task1 drop-oldest, 7=Task2 広域例外）。
- §6 影響範囲（sender.py 変更・test_sender.py 追加・他無改修）→ File Structure と一致。

**2. Placeholder scan:** TODO/TBD・曖昧な「適切に処理」等なし。各コードステップは実コードを提示。

**3. Type consistency:** `RemoteSender(remote, credentials, maxsize)` / `.queue` / `.channel` / `.enqueue` / `._send` / `.run` の名称・シグネチャは全タスクで一致。`_dispatch_output(context, senders, credentials, spawn, worker_output)` の引数順はテスト呼び出しと一致。`fake_transport` の `state["channels"]` / `state["commands"]` 参照も整合。
