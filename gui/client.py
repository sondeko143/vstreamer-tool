"""走っている pipeline へ制御 Command を 1 本送る gRPC クライアント。

送るのは制御イベント (ping / pause / resume / reload) だけ。データイベント
(transcription や tts) は送らない — このパネルは pipeline を操作するもので、
pipeline に仕事を流し込むものではない。

Command の組み立ては vspeech 側の変換 (EventAddress.to_pb) をそのまま使う。
ここで PAUSE などの protobuf 定数を直に触ると、EventType ↔ Operation の対応が
2 箇所に分かれて片方だけずれる。
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

# このパネルが送れる操作。ping は「疎通確認」— 受け側は log を 1 行出すだけ
# なので、RPC が返ったこと自体が到達の証拠になる。
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
    """1 操作 = 1 チェーンの Command。後続イベントは持たせない。"""
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
    """`address` の pipeline へ 1 操作を送り、成否と往復時間を返す。

    到達できない・相手が例外を投げたといった想定内の失敗は例外にせず
    `ok=False` の結果として返す (呼び元は GUI スレッド)。deadline を必ず付ける
    — 付けないと落ちたホスト宛ての呼び出しが返らず、押した人はただ固まる。
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
    # result=False は受け側が「受けたが処理しなかった」と言っている状態。
    # RPC が成功したことと混ぜず、そのまま失敗として見せる。
    ok = bool(getattr(response, "result", False))
    return SendResult(ok=ok, elapsed_ms=elapsed, detail="" if ok else "result=False")


def _describe(error: grpc.RpcError) -> str:
    # 同期 stub の RpcError は grpc.Call でもあるので code()/details() を持つが、
    # 型の上では保証されないので取れなければ str() に落とす。
    code = getattr(error, "code", None)
    details = getattr(error, "details", None)
    if code is None or details is None:
        return str(error)
    return f"{code().name}: {details()}"
