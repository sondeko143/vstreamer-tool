"""obs-websocket 5.x クライアント (ADR-0043)。

必要なのは Hello(0)/Identify(1)/Identified(2) のハンドシェイクと
Request(6)/RequestResponse(7) の往復だけで、イベント購読・バッチ・msgpack は
使わない。websockets の API 変更 (過去に websockets.client.connect ->
websockets.asyncio.client.connect の移行があった) の影響をこのファイルに
閉じ込めるため、呼び出し側は ObsTransport 越しにしか触らない。
"""

import base64
import hashlib
import json
from asyncio import wait_for
from typing import Any
from typing import Protocol
from uuid import uuid4

RPC_VERSION = 1

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7

STATUS_RESOURCE_NOT_FOUND = 600


class ObsProtocolError(Exception):
    """obs-websocket との対話が想定外の形になった。"""


class ObsIdentifyError(ObsProtocolError):
    """Identify が成立しなかった (認証失敗・RPC バージョン不一致など)。

    リトライしても直らない種類なので、呼び出し側は fail-loud に扱う (ADR-0042)。
    """


class ObsRequestError(ObsProtocolError):
    def __init__(self, request_type: str, code: int, comment: str):
        self.request_type = request_type
        self.code = code
        self.comment = comment
        super().__init__(f"{request_type} failed: code={code} {comment}")


class ObsResourceNotFoundError(ObsRequestError):
    """指定した input などが OBS に存在しない (code 600)。"""


class ObsTransport(Protocol):
    """websockets の ClientConnection が満たす最小の口。"""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


def build_auth_string(password: str, salt: str, challenge: str) -> str:
    """obs-websocket 5.x の認証文字列を作る。

    仕様の手順どおり:
      1. password + salt を sha256 して base64 -> base64 secret
      2. secret + challenge を sha256 して base64
    """
    secret = base64.b64encode(
        hashlib.sha256((password + salt).encode("utf-8")).digest()
    )
    return base64.b64encode(
        hashlib.sha256(secret + challenge.encode("utf-8")).digest()
    ).decode("utf-8")


class ObsWsClient:
    def __init__(self, transport: ObsTransport, timeout: float = 5.0):
        self._transport = transport
        self._timeout = timeout

    async def _send(self, op: int, d: dict[str, Any]) -> None:
        await self._transport.send(json.dumps({"op": op, "d": d}))

    async def _recv(self) -> dict[str, Any]:
        raw = await wait_for(self._transport.recv(), timeout=self._timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            message = json.loads(raw)
        except ValueError as e:
            raise ObsProtocolError(f"OBS から不正な JSON: {e}") from e
        if not isinstance(message, dict) or "op" not in message:
            raise ObsProtocolError(f"OBS から不正なメッセージ: {message!r}")
        return message

    async def identify(self, password: str) -> None:
        message = await self._recv()
        if message["op"] != OP_HELLO:
            raise ObsIdentifyError(f"Hello を期待したが op={message['op']} が来た")
        auth = message["d"].get("authentication")
        d: dict[str, Any] = {"rpcVersion": RPC_VERSION}
        if auth:
            if not password:
                raise ObsIdentifyError(
                    "OBS が認証を要求していますが subtitle.obs.password が空です"
                )
            d["authentication"] = build_auth_string(
                password, auth["salt"], auth["challenge"]
            )
        await self._send(OP_IDENTIFY, d)
        message = await self._recv()
        if message["op"] != OP_IDENTIFIED:
            raise ObsIdentifyError(f"Identified を期待したが op={message['op']} が来た")

    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        await self._send(
            OP_REQUEST,
            {
                "requestType": request_type,
                "requestId": request_id,
                "requestData": request_data or {},
            },
        )
        while True:
            message = await self._recv()
            # イベント (op 5) や他リクエストの応答は捨てる。
            if message["op"] != OP_REQUEST_RESPONSE:
                continue
            d = message["d"]
            if d.get("requestId") != request_id:
                continue
            status = d["requestStatus"]
            if not status.get("result"):
                code = status.get("code", 0)
                comment = status.get("comment", "")
                if code == STATUS_RESOURCE_NOT_FOUND:
                    raise ObsResourceNotFoundError(request_type, code, comment)
                raise ObsRequestError(request_type, code, comment)
            return d.get("responseData") or {}
