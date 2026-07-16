import asyncio
import json
from collections import deque
from collections.abc import Callable

import pytest

from vspeech.lib.obs_ws import OP_IDENTIFY
from vspeech.lib.obs_ws import OP_REQUEST
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsProtocolError
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

    def __init__(
        self,
        *,
        require_auth: bool = True,
        greet: bool = True,
        hang: bool = False,
    ):
        self.sent: list[dict] = []
        self.closed = False
        self._outgoing: deque[str | bytes] = deque()
        self._responses: deque[tuple[bool, int, dict]] = deque()
        self._malformed_responses = 0
        self._raw_responses: deque[Callable[[str], str | bytes]] = deque()
        # recv() が尽きたときに例外で止まる代わりに永遠に待つ (finding 1:
        # _recv() の wait_for タイムアウトを駆動するためのモード)。
        self._hang = hang
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

    def script_malformed_response(self):
        """次の Request に requestStatus を欠いた不正な応答を返す (finding 2)。"""
        self._malformed_responses += 1

    def script_raw_response(self, builder: Callable[[str], str | bytes]):
        """次の Request に、`builder(requestId)` が返す生ペイロードをそのまま送る
        (JSON エンコードしない)。requestId はクライアントが採番した実際の値が渡って
        くるので、requestStatus 周りの壊れ方 (missing/非 dict) を、クライアントの
        マッチングループ (`requestId` 一致待ち) を素通りさせた上で作れる。壊れ方が
        requestId と無関係な生フレーム (不正 JSON・非 UTF-8・配列) にも同じフックを
        使う (`builder` は引数を無視すればよい)。
        """
        self._raw_responses.append(builder)

    def stop_responding(self):
        """以降、何を送られても応答しなくなる (finding 1 のタイムアウト用)。"""
        self._hang = True

    def inject(self, message: dict):
        """次の応答より前に届く生メッセージ (イベント等) を積む。"""
        self._outgoing.append(json.dumps(message))

    def inject_raw(self, payload: str | bytes):
        """次の応答より前に届く、JSON エンコードしていない生フレームを積む。
        不正 JSON・JSON 配列・非 UTF-8 バイト列など、`inject()` の dict 入力では
        作れない壊れ方を作るためのフック。
        """
        self._outgoing.append(payload)

    async def send(self, message: str) -> None:
        m = json.loads(message)
        self.sent.append(m)
        if self._hang:
            # サーバが無応答になったふりをする (finding 1 用): 何も積まない
            # ので、次の recv() は _outgoing が尽きて hang モードに入る。
            return
        if m["op"] == OP_IDENTIFY:
            self._outgoing.append(
                json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}})
            )
        elif m["op"] == OP_REQUEST:
            if self._raw_responses:
                builder = self._raw_responses.popleft()
                self._outgoing.append(builder(m["d"]["requestId"]))
                return
            if self._malformed_responses:
                self._malformed_responses -= 1
                self._outgoing.append(
                    json.dumps(
                        {
                            "op": 7,
                            "d": {
                                "requestType": m["d"]["requestType"],
                                "requestId": m["d"]["requestId"],
                                # requestStatus を意図的に省く。
                                "responseData": {},
                            },
                        }
                    )
                )
                return
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

    async def recv(self) -> str | bytes:
        if not self._outgoing:
            if self._hang:
                # 何も返さず、呼び出し側の wait_for がタイムアウトで cancel
                # するまで永遠に待つ。
                await asyncio.Event().wait()
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


# --- finding 1: _recv() が wait_for のタイムアウトを型付き例外に包む ---


async def test_identify_raises_obs_protocol_error_on_timeout():
    server = FakeObsServer(greet=False, hang=True)
    client = ObsWsClient(server, timeout=0.05)
    with pytest.raises(ObsProtocolError, match="0.05"):
        await client.identify("")


async def test_request_raises_obs_protocol_error_on_timeout():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.stop_responding()
    with pytest.raises(ObsProtocolError, match="0.05"):
        await client.request("GetInputSettings", {"inputName": "x"})


# --- finding 2: 不正なネスト構造が KeyError ではなく ObsProtocolError になる ---


async def test_recv_raises_obs_protocol_error_when_d_is_missing():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0})  # Hello だが 'd' が無い不正なメッセージ
    with pytest.raises(ObsProtocolError) as e:
        await ObsWsClient(server).identify("")
    assert not isinstance(e.value, ObsIdentifyError)


async def test_recv_raises_obs_protocol_error_when_d_is_not_a_dict():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": "not-a-dict"})
    with pytest.raises(ObsProtocolError) as e:
        await ObsWsClient(server).identify("")
    assert not isinstance(e.value, ObsIdentifyError)


async def test_request_raises_obs_protocol_error_when_request_status_is_missing():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_malformed_response()
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- fix pass 2, finding "Important": identify() 認証情報の生 index アクセス ---
# (finding 2 と同じ穴の class。_recv() が d を dict と保証しても、その中身
# (`authentication`) の形は OBS が選ぶので、さらに一段検査が要る。)


async def test_identify_raises_obs_identify_error_when_authentication_is_not_a_dict():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": "not-a-dict"}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


async def test_identify_raises_obs_identify_error_when_salt_is_missing():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": {"challenge": AUTH_CHALLENGE}}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


async def test_identify_raises_obs_identify_error_when_challenge_is_missing():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": {"salt": AUTH_SALT}}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


# --- fix pass 3: salt/challenge が str だが UTF-8 エンコード不能 (非対 surrogate) ---
# isinstance(salt, str) は通るが .encode("utf-8") で素の UnicodeEncodeError を送出
# しうる穴。JSON のワイヤ上は素の ASCII \uD800 なので websockets のフレームレベル
# UTF-8 検証もこれを止めない (レビューで実機再現済み)。


async def test_identify_raises_obs_identify_error_when_salt_is_not_utf8_encodable():
    server = FakeObsServer(greet=False)
    server.inject(
        {
            "op": 0,
            "d": {"authentication": {"salt": "\ud800", "challenge": AUTH_CHALLENGE}},
        }
    )
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


# --- fix pass 2, finding "Minor": 非 UTF-8 バイト列フレームが decode で漏れる ---


async def test_recv_raises_obs_protocol_error_on_non_utf8_bytes_frame():
    server = FakeObsServer(greet=False)
    server.inject_raw(b"\xff\xfe\x00\x01")
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


# --- fix pass 2: 契約テスト。個々のバグではなく契約そのものを固定する。
# identify()/request() に悪意・破損したサーバメッセージを大量に流し込み、どれも
# 許容集合の外 (KeyError/TypeError/AttributeError/UnicodeDecodeError/TimeoutError
# などの素の例外) に漏れず、必ず ObsProtocolError の階層内に収まることを保証する。
# 次に見つかる「4つ目」を防ぐのはこのテストの役目であって、個別テストの役目では
# ない。---

IDENTIFY_HOSTILE_HELLOS: list[tuple[str, dict]] = [
    ("d_missing", {"op": 0}),
    ("d_not_a_dict", {"op": 0, "d": "not-a-dict"}),
    ("authentication_not_a_dict", {"op": 0, "d": {"authentication": "not-a-dict"}}),
    (
        "authentication_missing_salt",
        {"op": 0, "d": {"authentication": {"challenge": AUTH_CHALLENGE}}},
    ),
    (
        "authentication_missing_challenge",
        {"op": 0, "d": {"authentication": {"salt": AUTH_SALT}}},
    ),
    (
        "authentication_salt_not_utf8_encodable",
        {
            "op": 0,
            "d": {"authentication": {"salt": "\ud800", "challenge": AUTH_CHALLENGE}},
        },
    ),
    ("op_missing", {"d": {}}),
]

IDENTIFY_HOSTILE_RAW_FRAMES: list[tuple[str, str | bytes]] = [
    ("invalid_json", "{not valid json"),
    ("json_array_instead_of_object", "[1, 2, 3]"),
    ("non_utf8_bytes_frame", b"\xff\xfe\x00\x01"),
]


@pytest.mark.parametrize(
    "hello",
    [pytest.param(msg, id=name) for name, msg in IDENTIFY_HOSTILE_HELLOS],
)
async def test_identify_never_leaks_a_raw_exception_on_hostile_hello(hello):
    server = FakeObsServer(greet=False)
    server.inject(hello)
    with pytest.raises(ObsProtocolError):
        # 空でないパスワードを渡し、authentication 系のケースが
        # "password が空" の早期リターンで隠れないようにする。
        await ObsWsClient(server, timeout=0.05).identify("irrelevant-password")


@pytest.mark.parametrize(
    "raw",
    [pytest.param(payload, id=name) for name, payload in IDENTIFY_HOSTILE_RAW_FRAMES],
)
async def test_identify_never_leaks_a_raw_exception_on_hostile_raw_frame(raw):
    server = FakeObsServer(greet=False)
    server.inject_raw(raw)
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server, timeout=0.05).identify("irrelevant-password")


def _request_status_missing(rid: str) -> str:
    return json.dumps(
        {"op": 7, "d": {"requestId": rid, "requestType": "X", "responseData": {}}}
    )


def _request_status_not_a_dict(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {"requestId": rid, "requestType": "X", "requestStatus": "nope"},
        }
    )


def _op_missing(rid: str) -> str:
    return json.dumps({"d": {"requestId": rid}})


def _invalid_json(rid: str) -> str:
    return "{not valid json"


def _json_array_instead_of_object(rid: str) -> str:
    return "[1, 2, 3]"


def _non_utf8_bytes_frame(rid: str) -> bytes:
    return b"\xff\xfe\x00\x01"


REQUEST_HOSTILE_RESPONSES: list[tuple[str, Callable[[str], str | bytes]]] = [
    ("requestStatus_missing", _request_status_missing),
    ("requestStatus_not_a_dict", _request_status_not_a_dict),
    ("op_missing", _op_missing),
    ("invalid_json", _invalid_json),
    ("json_array_instead_of_object", _json_array_instead_of_object),
    ("non_utf8_bytes_frame", _non_utf8_bytes_frame),
]


@pytest.mark.parametrize(
    "builder",
    [pytest.param(fn, id=name) for name, fn in REQUEST_HOSTILE_RESPONSES],
)
async def test_request_never_leaks_a_raw_exception(builder):
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.script_raw_response(builder)
    with pytest.raises(ObsProtocolError):
        await client.request("GetInputSettings", {"inputName": "x"})
