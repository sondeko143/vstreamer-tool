import json
from collections import deque

import pytest

from vspeech.lib.obs_ws import OP_IDENTIFY
from vspeech.lib.obs_ws import OP_REQUEST
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsRequestError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.obs_ws import ObsWsClient
from vspeech.lib.obs_ws import build_auth_string

# obs-websocket 5.x の認証アルゴリズム:
#   1. base64(sha256(password + salt))    -> base64 secret
#   2. base64(sha256(secret + challenge))
# 期待値は simpleobsws (obs-websocket 本家 IRLToolkit) の実装と行レベルで一致
# することを確認した上で固定した回帰ベクタ (ADR-0043)。password は文字列
# "supersecretpassword"、salt/challenge はこのテスト専用の固定値で、実在の OBS
# のものではない (.gitleaks.toml で値ごと allowlist 済み)。
AUTH_PASSWORD = "supersecretpassword"
AUTH_SALT = "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI="
AUTH_CHALLENGE = "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY="
AUTH_EXPECTED = "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4="


class FakeObsServer:
    """スクリプト化した obs-websocket サーバ。ネットワークも OBS も使わない。

    「送られてきたものに応答する」形にしてある。おかげでテストはクライアントが
    採番した requestId を知る必要がなく、応答は send の時点で積まれるので
    待ち合わせも要らない。
    """

    def __init__(self, *, require_auth: bool = True, greet: bool = True):
        self.sent: list[dict] = []
        self.closed = False
        self._outgoing: deque[str] = deque()
        self._responses: deque[tuple[bool, int, dict]] = deque()
        if greet:
            hello: dict = {"obsWebSocketVersion": "5.5.0", "rpcVersion": 1}
            if require_auth:
                hello["authentication"] = {
                    "salt": AUTH_SALT,
                    "challenge": AUTH_CHALLENGE,
                }
            self._outgoing.append(json.dumps({"op": 0, "d": hello}))

    def script_response(
        self, *, ok: bool = True, code: int = 100, data: dict | None = None
    ):
        """次の Request に返す応答を積む。"""
        self._responses.append((ok, code, data or {}))

    def inject(self, message: dict):
        """次の応答より前に届く生メッセージ (イベント等) を積む。"""
        self._outgoing.append(json.dumps(message))

    async def send(self, message: str) -> None:
        m = json.loads(message)
        self.sent.append(m)
        if m["op"] == OP_IDENTIFY:
            self._outgoing.append(
                json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}})
            )
        elif m["op"] == OP_REQUEST:
            ok, code, data = (
                self._responses.popleft() if self._responses else (True, 100, {})
            )
            self._outgoing.append(
                json.dumps(
                    {
                        "op": 7,
                        "d": {
                            "requestType": m["d"]["requestType"],
                            # 採番された id をそのまま返す = テスト側の受け渡し不要。
                            "requestId": m["d"]["requestId"],
                            "requestStatus": {"result": ok, "code": code},
                            "responseData": data,
                        },
                    }
                )
            )

    async def recv(self) -> str:
        if not self._outgoing:
            raise AssertionError("client recv'd more than the fake scripted")
        return self._outgoing.popleft()

    async def close(self) -> None:
        self.closed = True


def test_build_auth_string_matches_the_reference_vector():
    assert build_auth_string(AUTH_PASSWORD, AUTH_SALT, AUTH_CHALLENGE) == AUTH_EXPECTED


async def test_identify_sends_rpc_version_and_auth():
    server = FakeObsServer()
    await ObsWsClient(server).identify(AUTH_PASSWORD)
    assert server.sent == [
        {"op": 1, "d": {"rpcVersion": 1, "authentication": AUTH_EXPECTED}}
    ]


async def test_identify_omits_auth_when_server_does_not_ask():
    server = FakeObsServer(require_auth=False)
    await ObsWsClient(server).identify("")
    assert server.sent == [{"op": 1, "d": {"rpcVersion": 1}}]


async def test_identify_raises_when_server_wants_auth_but_password_is_empty():
    server = FakeObsServer()
    with pytest.raises(ObsIdentifyError, match="password"):
        await ObsWsClient(server).identify("")


async def test_identify_raises_when_hello_is_not_first():
    server = FakeObsServer(greet=False)
    server.inject({"op": 5, "d": {"eventType": "Surprise"}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("")


async def test_request_sends_the_request_and_returns_response_data():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(data={"inputSettings": {"text": "hi"}})
    got = await client.request("GetInputSettings", {"inputName": "x"})
    assert got == {"inputSettings": {"text": "hi"}}
    assert server.sent[-1]["op"] == 6
    assert server.sent[-1]["d"]["requestType"] == "GetInputSettings"
    assert server.sent[-1]["d"]["requestData"] == {"inputName": "x"}


async def test_request_generates_a_unique_request_id_per_call():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response()
    server.script_response()
    await client.request("A")
    await client.request("B")
    ids = [m["d"]["requestId"] for m in server.sent if m["op"] == 6]
    assert len(set(ids)) == 2


async def test_request_ignores_events_and_other_request_ids():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.inject({"op": 5, "d": {"eventType": "InputNameChanged"}})
    server.inject(
        {
            "op": 7,
            "d": {
                "requestType": "GetInputSettings",
                "requestId": "someone-elses-id",
                "requestStatus": {"result": True, "code": 100},
                "responseData": {"nope": True},
            },
        }
    )
    server.script_response(data={"mine": True})
    assert await client.request("GetInputSettings", {"inputName": "x"}) == {
        "mine": True
    }


async def test_request_raises_resource_not_found_on_600():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(ok=False, code=600)
    with pytest.raises(ObsResourceNotFoundError):
        await client.request("GetInputSettings", {"inputName": "nope"})


async def test_request_raises_generic_error_on_other_failures():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(ok=False, code=400)
    with pytest.raises(ObsRequestError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert e.value.code == 400


async def test_request_returns_empty_dict_when_there_is_no_response_data():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_response(data={})
    assert await client.request("SetInputSettings", {"inputName": "x"}) == {}
