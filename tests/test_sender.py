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
