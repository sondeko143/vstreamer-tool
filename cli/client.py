"""The gRPC client that sends a single control Command to a running pipeline.

It only sends control events (ping / pause / resume / reload). Data events (transcription
or tts) are never sent -- this CLI operates the pipeline, it does not feed work into it.

Commands are built through vspeech's own conversion (EventAddress.to_pb). Touching
protobuf constants such as PAUSE directly here would split the EventType <-> Operation
mapping across two places, where only one of them gets updated.
"""

from dataclasses import dataclass
from time import monotonic

import grpc
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operand
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.config import EventType
from vspeech.shared_context import EventAddress

# The operations this CLI can send. ping is a reachability check -- the peer only writes
# one log line, so the RPC returning is itself the proof that it arrived.
OPERATIONS: tuple[EventType, ...] = (
    EventType.ping,
    EventType.pause,
    EventType.resume,
    EventType.reload,
)

DEFAULT_TIMEOUT = 3.0


@dataclass(frozen=True)
class SendResult:
    ok: bool
    elapsed_ms: float
    detail: str = ""


def build_command(event: EventType, config_path: str = "") -> Command:
    """A Command of one chain per operation. It carries no following events."""
    route = EventAddress(event=event).to_pb()
    return Command(
        chains=[OperationChain(operations=[route])],
        operand=Operand(file_path=config_path),
    )


def send(
    address: str,
    event: EventType,
    config_path: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> SendResult:
    """Send one operation to the pipeline at `address` and return the outcome and the
    round-trip time.

    Expected failures -- unreachable peer, the peer raising -- are returned as an
    `ok=False` result rather than raised (this dates from when the caller was the GUI
    thread). Always attach a deadline: without one, a call to a dead host never returns
    and whoever pressed the button simply hangs.
    """
    command = build_command(event, config_path=config_path)
    started = monotonic()
    try:
        with grpc.insecure_channel(address) as channel:
            response = CommanderStub(channel).process_command(command, timeout=timeout)
    except grpc.RpcError as e:
        elapsed = (monotonic() - started) * 1000
        return SendResult(ok=False, elapsed_ms=elapsed, detail=_describe(e))
    elapsed = (monotonic() - started) * 1000
    # result=False is the peer saying "received but not processed". Do not conflate that
    # with the RPC having succeeded; surface it as a failure.
    ok = bool(getattr(response, "result", False))
    return SendResult(ok=ok, elapsed_ms=elapsed, detail="" if ok else "result=False")


def _describe(error: grpc.RpcError) -> str:
    # The synchronous stub's RpcError is also a grpc.Call, so it has code()/details(), but
    # the types do not guarantee that; fall back to str() when they are unavailable.
    code = getattr(error, "code", None)
    details = getattr(error, "details", None)
    if code is None or details is None:
        return str(error)
    return f"{code().name}: {details()}"
