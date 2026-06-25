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
from vspeech.worker.sender import _dispatch_output


def make_command(op: Operation = VC, data: bytes = b"abc") -> Command:
    return Command(
        chains=[OperationChain(operations=[OperationRoute(operation=op)])],
        operand=Operand(
            sound=Sound(data=data, rate=16000, format=SampleFormat.INT16, channels=1),
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
