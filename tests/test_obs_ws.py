import asyncio
import json
from collections import deque
from collections.abc import Callable

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from vspeech.lib.obs_ws import OP_IDENTIFY
from vspeech.lib.obs_ws import OP_REQUEST
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsProtocolError
from vspeech.lib.obs_ws import ObsRequestError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.obs_ws import ObsWsClient
from vspeech.lib.obs_ws import build_auth_string

# obs-websocket 5.x's authentication algorithm:
#   1. base64(sha256(password + salt))    -> the base64 secret
#   2. base64(sha256(secret + challenge))
# The expected values are regression vectors pinned after confirming they match
# simpleobsws (obs-websocket's own IRLToolkit implementation) line for line (ADR-0043).
# The password is the string "supersecretpassword" and the salt/challenge are fixed values
# used only by this test, not from any real OBS (allowlisted by value in .gitleaks.toml).
AUTH_PASSWORD = "supersecretpassword"
AUTH_SALT = "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI="
AUTH_CHALLENGE = "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY="
AUTH_EXPECTED = "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4="


class FakeObsServer:
    """A scripted obs-websocket server. Uses neither the network nor OBS.

    It is shaped as "respond to whatever was sent", which means the tests never need to
    know the requestId the client assigned, and since the response is queued at send time
    there is nothing to synchronize on.
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
        # Wait forever instead of stopping with an exception when recv() runs dry
        # (the mode used to drive _recv()'s wait_for timeout).
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
        """Queue the response to return for the next Request."""
        self._responses.append((ok, code, data or {}))

    def script_malformed_response(self):
        """Return a malformed response missing requestStatus for the next Request."""
        self._malformed_responses += 1

    def script_raw_response(self, builder: Callable[[str], str | bytes]):
        """For the next Request, send the raw payload `builder(requestId)` returns, as-is
        (without JSON encoding). The actual requestId the client assigned is passed in, so
        corruption around requestStatus (missing / not a dict) can be produced after
        getting past the client's matching loop (which waits for a `requestId` match). The
        same hook is used for raw frames whose corruption is unrelated to requestId
        (malformed JSON, non-UTF-8, an array); `builder` can simply ignore its argument.
        """
        self._raw_responses.append(builder)

    def stop_responding(self):
        """From now on, never respond to anything (for the timeout tests)."""
        self._hang = True

    def inject(self, message: dict):
        """Queue a raw message (an event, etc.) that arrives before the next response."""
        self._outgoing.append(json.dumps(message))

    def inject_raw(self, payload: str | bytes):
        """Queue a raw, non-JSON-encoded frame that arrives before the next response.
        The hook for producing corruption that `inject()`'s dict input cannot express:
        malformed JSON, a JSON array, non-UTF-8 bytes, and so on.
        """
        self._outgoing.append(payload)

    async def send(self, message: str) -> None:
        m = json.loads(message)
        self.sent.append(m)
        if self._hang:
            # Pretend the server went unresponsive: nothing is queued, so the next recv()
            # exhausts _outgoing and enters hang mode.
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
                                # requestStatus is deliberately omitted.
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
                            # Echo the assigned id = nothing to hand around in the test.
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
                # Return nothing and wait forever, until the caller's wait_for times out
                # and cancels.
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
        {
            "op": 1,
            "d": {
                "rpcVersion": 1,
                "eventSubscriptions": 0,
                "authentication": AUTH_EXPECTED,
            },
        }
    ]


async def test_identify_omits_auth_when_server_does_not_ask():
    server = FakeObsServer(require_auth=False)
    await ObsWsClient(server).identify("")
    assert server.sent == [{"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 0}}]


async def test_identify_unsubscribes_from_all_events():
    """Identify must say eventSubscriptions=0, not just omit it.

    Omitting it is not "no events": obs-websocket documents the default as
    EventSubscription::All, so OBS starts pushing op-5 event frames at a client
    that never reads them -- this module only ever recv()s while a request is in
    flight (ObsWsClient.request), and the subtitle worker spends the rest of its
    life blocked on its own in_queue.

    Those unread frames are what actually killed the connection, and the failure
    looks nothing like its cause: after 16 of them (websockets' default
    max_queue high-water mark) the Assembler calls transport.pause_reading(),
    which stops parsing *every* frame including Pong. keepalive() then gets no
    pong within ping_timeout and fails the connection itself with
    1011 "keepalive ping timeout" -- on a loopback socket, where a real network
    timeout is not a plausible reading. Observed in production as a subtitle
    reconnect after each long idle gap (short gaps survived only because each
    subtitle push happens to drain the backlog).

    Asserting on the exact payload rather than just the key, so that a future
    edit cannot quietly widen the subscription back to a firehose nobody drains.
    """
    server = FakeObsServer(require_auth=False)
    await ObsWsClient(server).identify("")
    assert server.sent == [{"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 0}}]


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


# --- _recv() wraps wait_for's timeout in a typed exception ---


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


# --- a malformed nested structure becomes ObsProtocolError, not KeyError ---


async def test_recv_raises_obs_protocol_error_when_d_is_missing():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0})  # a Hello, but a malformed message with no 'd'
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


# --- raw index access into identify()'s credentials ---
# (Even though _recv() guarantees d is a dict, the shape of what is inside it
# (`authentication`) is chosen by OBS, so one more layer of checking is required.)


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


# --- salt/challenge is a str but not UTF-8 encodable (an unpaired surrogate) ---
# isinstance(salt, str) passes, yet .encode("utf-8") can raise a bare UnicodeEncodeError.
# On the JSON wire it is plain ASCII \uD800, so websockets' frame-level UTF-8 validation
# does not stop it either (reproduced on real hardware).


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


# Fills the gap where build_auth_string() calls .encode("utf-8") twice (for salt and for
# challenge) but only the salt case was covered. The behaviour itself is already correct
# because identify()'s try/except UnicodeError wraps both (this test passes either way; it
# is not a missed edge case).


async def test_identify_raises_obs_identify_error_when_challenge_is_not_utf8_encodable():
    server = FakeObsServer(greet=False)
    server.inject(
        {
            "op": 0,
            "d": {"authentication": {"salt": AUTH_SALT, "challenge": "\ud800"}},
        }
    )
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


# --- `if auth:` proves truthiness, but the question is whether the key is present.
# It would misread `{}` / `[]` / `False` / `0` / `""` as "no auth needed" and send an
# unauthenticated Identify; if OBS really does require auth, that means a 4008 close ->
# WebSocketException and the caller retrying forever (a broken handshake does not get
# better on retry). ---


async def test_identify_raises_when_authentication_present_but_empty_dict_and_password_empty():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": {}}})
    with pytest.raises(ObsIdentifyError, match="password"):
        await ObsWsClient(server).identify("")


async def test_identify_raises_when_authentication_present_but_false_and_password_empty():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": False}})
    with pytest.raises(ObsIdentifyError, match="password"):
        await ObsWsClient(server).identify("")


async def test_identify_raises_when_authentication_present_but_not_a_dict_and_password_set():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": []}})
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


# --- a non-UTF-8 byte frame leaking out of decode ---


async def test_recv_raises_obs_protocol_error_on_non_utf8_bytes_frame():
    server = FakeObsServer(greet=False)
    server.inject_raw(b"\xff\xfe\x00\x01")
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


# --- a RecursionError from json.loads leaking out of _recv().
# JSONDecodeError (a ValueError) is caught, but nesting that is too deep -- constructible
# from an ASCII string alone, such as "[" * N + "]" * N -- raises RecursionError (a
# subclass of RuntimeError, not ValueError). It is transport-valid input well inside
# websockets' default max_size (1 MiB), so nothing upstream rejects it. Reproduced on real
# hardware in both identify() (bare and wrapped in an op-d envelope) and request(). ---

_DEEPLY_NESTED_JSON_ARRAY = "[" * 12000 + "]" * 12000


async def test_identify_raises_obs_protocol_error_on_deeply_nested_json_array():
    server = FakeObsServer(greet=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_ARRAY)
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


async def test_identify_raises_obs_protocol_error_on_deeply_nested_json_in_envelope():
    server = FakeObsServer(greet=False)
    server.inject_raw('{"op":0,"d":{"x":' + _DEEPLY_NESTED_JSON_ARRAY + "}}")
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


async def test_request_raises_obs_protocol_error_on_deeply_nested_json_array():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(lambda rid: _DEEPLY_NESTED_JSON_ARRAY)
    with pytest.raises(ObsProtocolError):
        await client.request("GetInputSettings", {"inputName": "x"})


# --- responseData was unchecked on the return path.
# requestStatus has an isinstance(dict) check but responseData did not, so a peer sending
# list/str/int/bool/float would have its value returned as-is, in violation of the type
# annotation (-> dict[str, Any]). It does not raise here today; it merely defers the same
# bug shape by one frame, to a bare TypeError at the caller's first
# `result["inputSettings"]`. The builder is defined further down
# (_response_data_not_a_dict). ---


async def test_request_raises_obs_protocol_error_when_response_data_is_not_a_dict():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_response_data_not_a_dict)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- ObsRequestError's code/comment were unchecked.
# A peer sending code as "600" (str) makes `code == STATUS_RESOURCE_NOT_FOUND` fail to
# match, demoting ObsResourceNotFoundError to the generic ObsRequestError so the caller's
# dedicated "resource missing" fail-loud path never fires. A comment arriving as dict/None
# makes e.comment.lower() die with a bare AttributeError. The builders are defined further
# down (_code_not_an_int / _comment_not_a_string). ---


async def test_request_raises_obs_protocol_error_when_code_is_not_an_int():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_code_not_an_int)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


async def test_request_raises_obs_protocol_error_when_comment_is_not_a_string():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_comment_not_a_string)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- repr() 自体が RecursionError を漏らす。
# json.loads() の RecursionError 自体を拾うようになった後も、「壊れた
# メッセージを報告する」ためにエラーメッセージの中で repr() を呼んで
# いる箇所 (message!r が 2 箇所 / auth!r が 2 箇所 / d!r / status!r /
# response_data!r / message['op'] が 2 箇所、計 9 箇所) が同じ問題を持って
# いた: ネストの深い値の repr() 自体が RecursionError を送出し、
# ObsProtocolError を組み立てている最中に素の RecursionError (RuntimeError
# のサブクラスで許容集合の外) が漏れる。
#
# 既存のテストはすべて **配列** ("[" * N + "]" * N) のネストを使っていたが、
# 配列は json.loads() 側が先に力尽きる (~11400 段) ので repr() まで生きて
# 届かず、この穴を暴けない。**オブジェクト** ('{"a":' * N + ... + '}' * N)
# は json.loads() が ~9600 段まで生き延びる一方 repr() は ~9000 段から壊れ
# 始めるので、9000-9600 段の窓でだけ再現する (残り C スタックに依存して
# 動く境界なので、これより深くすれば必ず壊れるという「魔法の深さ」として
# 扱わないこと)。9200 段はこの窓の中で実測して選んだ深さ。
_DEEPLY_NESTED_JSON_OBJECT = '{"a":' * 9200 + "1" + "}" * 9200

_DEEPLY_NESTED_JSON_OBJECT_D_NOT_A_DICT = (
    '{"op":0,"d":"not-a-dict","extra":' + _DEEPLY_NESTED_JSON_OBJECT + "}"
)

_DEEPLY_NESTED_JSON_OBJECT_AUTHENTICATION = (
    '{"op":0,"d":{"authentication":' + _DEEPLY_NESTED_JSON_OBJECT + "}}"
)


async def test_recv_raises_obs_protocol_error_on_deeply_nested_json_object_missing_op():
    # 実機で再現した入力そのもの: op キーの無い巨大な JSON オブジェクト。
    # _recv() の `"op" not in message` ガードに落ちて message!r を組み立てる
    # ときに RecursionError が漏れていた。
    server = FakeObsServer(greet=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT)
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


async def test_recv_raises_obs_protocol_error_on_deeply_nested_json_object_d_not_a_dict():
    # 'd' 自体は "not-a-dict" という浅い違反だが、message 全体 (兄弟キー
    # "extra" 配下) に深いネストを仕込むと message!r の repr() が同じ形で
    # 落ちる。
    server = FakeObsServer(greet=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT_D_NOT_A_DICT)
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


async def test_identify_raises_obs_identify_error_on_deeply_nested_authentication():
    # authentication は dict だが salt/challenge を持たない深いネスト。
    # auth!r の repr() が RecursionError で落ちていた。
    server = FakeObsServer(greet=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT_AUTHENTICATION)
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("irrelevant-password")


def _deeply_nested_json_object_requeststatus_not_a_dict(rid: str) -> str:
    # d!r (request() の「requestStatus が無い」ガード) 用: requestStatus は
    # "nope" という浅い違反だが、兄弟キー "extra" に深いネストを仕込む。
    return (
        '{"op":7,"d":{"requestId":"'
        + rid
        + '","requestType":"X","requestStatus":"nope","extra":'
        + _DEEPLY_NESTED_JSON_OBJECT
        + "}}"
    )


def _deeply_nested_json_object_code(rid: str) -> str:
    # status!r (request() の「code/comment が不正な形」ガード) 用: code
    # 自体を深くネストする。
    return (
        '{"op":7,"d":{"requestId":"'
        + rid
        + '","requestType":"X","requestStatus":{"result":false,"code":'
        + _DEEPLY_NESTED_JSON_OBJECT
        + ',"comment":"x"}}}'
    )


async def test_request_raises_obs_protocol_error_on_deeply_nested_request_status():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_deeply_nested_json_object_requeststatus_not_a_dict)
    with pytest.raises(ObsProtocolError):
        await client.request("GetInputSettings", {"inputName": "x"})


async def test_request_raises_obs_protocol_error_on_deeply_nested_code():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_deeply_nested_json_object_code)
    with pytest.raises(ObsProtocolError):
        await client.request("GetInputSettings", {"inputName": "x"})


# 副次効果 (milder problem): auth /
# responseData が dict でないと判定される 2 箇所 (identify() の
# "authentication が不正な形" ガード、request() の "responseData が不正な形"
# ガード) は、判定に落ちる値自体が非 dict (JSON では文字列・数値・真偽値・
# 配列) でなければならない。配列は上の RecursionError の窓を開けない
# (json.loads() 側が先に力尽きる) ので、この 2 箇所は深いネストによる
# RecursionError では再現できない。だが同じ bare repr() は、ピアが巨大な
# 文字列を送れば例外メッセージ (ひいてはログ行) をそのまま巨大化させる、
# という別の (milder な) 形で漏れる。ここは深さではなく大きさを検査する。
async def test_identify_error_message_is_bounded_when_authentication_is_a_huge_string():
    server = FakeObsServer(greet=False)
    server.inject({"op": 0, "d": {"authentication": "x" * 5000}})
    with pytest.raises(ObsIdentifyError) as e:
        await ObsWsClient(server).identify("irrelevant-password")
    assert len(str(e.value)) < 500


def _response_data_is_a_huge_string(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {"result": True, "code": 100},
                "responseData": "x" * 5000,
            },
        }
    )


async def test_request_error_message_is_bounded_when_response_data_is_a_huge_string():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_response_data_is_a_huge_string)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert len(str(e.value)) < 500


# --- code が bool を素通りさせる。
# isinstance(True, int) は True (bool は int のサブクラス) なので、bool を
# 明示的に除外しないと ObsRequestError.code に bool が入ってしまう。今日は
# 600 と一致しないので実害は無いが、このモジュールで繰り返し踏んでいる
# 「ガードが証明していることが足りない」形そのもの。
def _code_is_a_bool(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {"result": False, "code": True, "comment": "nope"},
                "responseData": {},
            },
        }
    )


async def test_request_raises_obs_protocol_error_when_code_is_a_bool():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_code_is_a_bool)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- `op`'s *value* is interpolated raw at
# the Hello guard and the Identified guard in identify() (the two
# `op={message['op']} が来た` f-strings). `_recv()` only proves
# `"op" in message` — nothing constrains the *value*, and an f-string's
# `{x}` calls format() -> dict.__repr__ exactly like `!r` does, so this is
# the same repr()-recursion / unbounded-size hazard as message/auth/d/status/
# response_data, just spelled without `!r`.
#
# All 11 IDENTIFY_HOSTILE_HELLOS rows above vary d/authentication; the only
# row that touches op (`op_missing`) *removes* the key rather than making
# its *value* hostile. These are the first tests that make op itself a
# hostile value, at both sites where it is interpolated into a message.
_DEEPLY_NESTED_JSON_OBJECT_AS_OP = '{"op":' + _DEEPLY_NESTED_JSON_OBJECT + ',"d":{}}'


async def test_identify_raises_obs_identify_error_on_deeply_nested_op_at_hello():
    # identify() の 1 回目の _recv() (Hello ガード) を直撃する。
    server = FakeObsServer(greet=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT_AS_OP)
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("")


async def test_identify_raises_obs_identify_error_on_deeply_nested_op_at_identified():
    # identify() の 2 回目の _recv() (Identified ガード) を直撃する。
    # require_auth=False の FakeObsServer は構築時に正規の Hello (op=0) を
    # 1 通目として積むので identify() の 1 通目はそのまま消費され認証も
    # 不要になり、事前に inject_raw() で積んだ hostile な op が 2 通目として
    # (送信直後にサーバが積む本物の Identified 応答より先に) 消費される。
    server = FakeObsServer(require_auth=False)
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT_AS_OP)
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(server).identify("")


async def test_identify_error_message_is_bounded_when_op_is_a_huge_string_at_hello():
    server = FakeObsServer(greet=False)
    server.inject({"op": "x" * 500000, "d": {}})
    with pytest.raises(ObsIdentifyError) as e:
        await ObsWsClient(server).identify("")
    assert len(str(e.value)) < 500


async def test_identify_error_message_is_bounded_when_op_is_a_huge_string_at_identified():
    server = FakeObsServer(require_auth=False)
    server.inject({"op": "x" * 500000, "d": {}})
    with pytest.raises(ObsIdentifyError) as e:
        await ObsWsClient(server).identify("")
    assert len(str(e.value)) < 500


# --- ObsRequestError.comment is the last
# unbounded peer->message path. A 500 KB comment must not become a 500 KB
# exception (and, downstream, a 500 KB retry-loop log line).
def _comment_is_a_huge_string(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {
                    "result": False,
                    "code": 400,
                    "comment": "x" * 500000,
                },
                "responseData": {},
            },
        }
    )


async def test_request_error_message_is_bounded_when_comment_is_a_huge_string():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_comment_is_a_huge_string)
    with pytest.raises(ObsRequestError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert len(str(e.value)) < 500


# --- Peer-controlled-field enumeration and coverage.
# Fields whose *value* the peer picks that reach a message or a comparison
# in this module:
#   op (message['op'])             -- newly covered above (message, 2 sites)
#                                      and below (comparison, request() side)
#   d (message['d'])                -- already covered:
#                                      comparison (isinstance) + message
#   authentication / auth           -- already covered:
#                                      comparison (isinstance) + message
#   salt / challenge                -- already covered:
#                                      comparison (isinstance); the only
#                                      message built from them is
#                                      UnicodeError's own str(e), which is
#                                      bounded regardless of input length
#   requestId (d.get('requestId'))  -- comparison only, never a message.
#                                      Covered below: a mismatched type
#                                      doesn't crash the `!=` comparison
#                                      (differing types -> NotImplemented ->
#                                      identity fallback, no repr()/no
#                                      nested traversal), it is simply
#                                      ignored like any other mismatch.
#   requestStatus / status          -- already covered:
#                                      comparison (isinstance) + message
#   code                             -- reaches a message (ObsRequestError)
#                                      and a comparison (== 600), but only
#                                      isinstance(int) is checked, not digit
#                                      count. Covered below: JSON integer
#                                      literals go through json.loads(),
#                                      which itself enforces CPython's
#                                      int<->str conversion limit (default
#                                      4300 digits) and raises ValueError
#                                      before `code` can ever be bound to a
#                                      value that large — already caught by
#                                      _recv()'s existing
#                                      `except (ValueError, RecursionError)`.
#   comment                          -- reaches a message; fixed above
#   responseData                    -- already covered:
#                                      comparison (isinstance) + message
#   requestType                     -- caller-controlled (the subtitle
#                                      worker's own literal, e.g.
#                                      "GetInputSettings"), not peer-
#                                      controlled; out of scope
#   result (status.get('result'))   -- only reaches `if not status.get(...)`,
#                                      i.e. bool(). dict/list.__bool__() is
#                                      an O(1) len() check that never
#                                      recurses into the value's contents,
#                                      so it has no repr()-style recursion
#                                      window; no dedicated test needed.


# inject_raw() で「本物の応答より前に届くメッセージ」として積む生フレーム
# なので、実際の requestId を知る必要が無い (どうせ一致しない値にする)。
_REQUEST_ID_IS_DEEPLY_NESTED_OBJECT = (
    '{"op":7,"d":{"requestId":'
    + _DEEPLY_NESTED_JSON_OBJECT
    + ',"requestType":"X","requestStatus":{"result":true},"responseData":{}}}'
)


async def test_request_ignores_a_deeply_nested_request_id_without_crashing():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.inject_raw(_REQUEST_ID_IS_DEEPLY_NESTED_OBJECT)
    server.script_response(data={"mine": True})
    got = await client.request("GetInputSettings", {"inputName": "x"})
    assert got == {"mine": True}


async def test_request_ignores_a_deeply_nested_op_event_without_crashing():
    # request() 側の同じ op 比較 (`message["op"] != OP_REQUEST_RESPONSE`) も
    # 型違いの比較は identity 比較に落ちるだけで安全 *なはず* であることを
    # 固定する: hostile な op のメッセージを無視し、後続の本物の応答を返す。
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.inject_raw(_DEEPLY_NESTED_JSON_OBJECT_AS_OP)
    server.script_response(data={"mine": True})
    got = await client.request("GetInputSettings", {"inputName": "x"})
    assert got == {"mine": True}


def _code_exceeds_int_max_str_digits(rid: str) -> str:
    # sys.get_int_max_str_digits() の既定値 (4300) を超える桁の整数
    # リテラル。json.loads() 自身がここで ValueError を投げる (実測で確認
    # 済み)。code の桁数そのものはこのモジュールでは検査していないが、
    # そこに届く前に _recv() の json.loads() が先に落ちて拾われるはず、
    # という経路を固定する。
    return (
        '{"op":7,"d":{"requestId":"'
        + rid
        + '","requestType":"X","requestStatus":{"result":false,"code":'
        + ("9" * 5000)
        + ',"comment":"x"}}}'
    )


async def test_request_never_leaks_a_raw_value_error_when_code_exceeds_int_max_str_digits():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.script_raw_response(_code_exceeds_int_max_str_digits)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- `code` was the one peer-controlled value that reached a
# message without passing through `_bounded_repr()` — only
# `isinstance(code, int)` was checked, never its digit count. The test above
# (`_code_exceeds_int_max_str_digits`, 5000 digits) only covers the side
# json.loads() already rejects. sys.get_int_max_str_digits() defaults to
# 4300: json.loads() raises ValueError at 4301+ digits but happily parses
# exactly 4300, so 4300 digits is the largest `code` that can ever reach
# `ObsRequestError.__init__` and its unbounded `f"code={code}"`. Both sides
# of that boundary need a test, or a false "already caught by json.loads()"
# belief (true at 4301, false at 4300) can silently regrow here.
def _code_at_the_int_max_str_digits_boundary(rid: str) -> str:
    return (
        '{"op":7,"d":{"requestId":"'
        + rid
        + '","requestType":"X","requestStatus":{"result":false,"code":'
        + ("9" * 4300)
        + ',"comment":"x"}}}'
    )


async def test_request_error_message_is_bounded_when_code_is_at_the_int_max_str_digits_boundary():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.script_raw_response(_code_at_the_int_max_str_digits_boundary)
    with pytest.raises(ObsRequestError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert len(str(e.value)) < 500


# --- `_SAFE_REPR` bounds nesting depth
# (`maxlevel`) and each leaf's size (`maxstring`/`maxother`), but never bounded
# the *total* rendered length. Within `maxlevel=6`, the default width caps
# (`maxlist=6`/`maxdict=4`) still allow up to ~6**6 leaves at ~200 chars each,
# so a wide-but-shallow structure blows past any "leaf is bounded" guarantee
# without ever touching the RecursionError depth window the earlier deeply-
# nested tests already cover. Every existing "deeply nested" fixture in this
# file varies *depth*; this section varies *width* instead, which is the axis
# that let the bug through undetected.
def _wide_json_array(levels: int, branch: int, leaf_json: str) -> str:
    """`levels` 段のネスト配列を、各段 `branch` 個の要素で作る。

    `branch` を reprlib の既定の幅上限 (`maxlist=6`) 以上にしておけば、深さ
    (`levels`) を `maxlevel=6` より大幅に浅く保ったまま出力サイズだけを
    爆発させられる — 深さは `_DEEPLY_NESTED_JSON_ARRAY`/`_DEEPLY_NESTED_JSON_OBJECT`
    系のテストが既に踏んでいる軸なので、ここでは意図的に浅くする。
    """
    x = leaf_json
    for _ in range(levels):
        x = "[" + ",".join([x] * branch) + "]"
    return x


_WIDE_JSON_LEAF = json.dumps("y" * 250)
# 深さ 4 段 (maxlevel=6 を大きく下回る) × 幅 6 (maxlist の既定値) で、実測
# 約 328 KB のフレームから約 262 KB の例外メッセージが組み上がる (実機の
# identify() で再現した「265 KB のフレーム -> 262 KB の例外」と同じ桁数)。
_WIDE_JSON_VALUE = _wide_json_array(4, 6, _WIDE_JSON_LEAF)


async def test_identify_error_message_is_bounded_when_op_is_a_wide_but_shallow_structure_at_hello():
    server = FakeObsServer(greet=False)
    server.inject_raw('{"op":' + _WIDE_JSON_VALUE + ',"d":{}}')
    with pytest.raises(ObsIdentifyError) as e:
        await ObsWsClient(server, timeout=0.05).identify("")
    assert len(str(e.value)) < 500


async def test_recv_error_message_is_bounded_when_message_is_wide_but_shallow():
    # 同じ幅ハザードを、op 単体ではなく _recv() の「'd' が無い」ガード
    # (message 全体を repr する側) でも固定する。
    server = FakeObsServer(greet=False)
    server.inject_raw('{"op":0,"d":"not-a-dict","extra":' + _WIDE_JSON_VALUE + "}")
    with pytest.raises(ObsProtocolError) as e:
        await ObsWsClient(server, timeout=0.05).identify("")
    assert not isinstance(e.value, ObsIdentifyError)
    assert len(str(e.value)) < 500


def _comment_is_a_wide_but_shallow_non_string(rid: str) -> str:
    return (
        '{"op":7,"d":{"requestId":"'
        + rid
        + '","requestType":"X","requestStatus":{"result":false,"code":400,'
        '"comment":' + _WIDE_JSON_VALUE + "}}}"
    )


async def test_request_error_message_is_bounded_when_comment_is_a_wide_but_shallow_non_string():
    # 同じ根っこ、別の枝: comment の 200 文字スライスは
    # `isinstance(comment, str)` を通った場合の枝にしか効かない。comment が
    # 非 str だと `not isinstance(comment, str)` に落ちて
    # `_bounded_repr(status)` の側 (request() の「requestStatus.code/comment
    # が不正な形」ガード) を通るので、comment 自身が wide-but-shallow な値
    # だと同じ幅ハザードでバイパスされていた。message の総量を bound する
    # ことでこちらも一緒に閉じることを確認する。
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server, timeout=0.05)
    await client.identify("")
    server.script_raw_response(_comment_is_a_wide_but_shallow_non_string)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("SetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)
    assert len(str(e.value)) < 500


# --- `ObsRequestError.__init__` now calls
# `len(comment)`, which is not total for a non-`str` `comment`. Both
# peer-reachable construction sites in this module gate on
# `isinstance(comment, str)` first, so a hostile peer can never reach this —
# but the class is public, so a caller constructing it directly with e.g.
# `None` must not get a raw `TypeError` in exchange.
def test_obs_request_error_is_constructible_with_a_non_string_comment():
    e = ObsRequestError("X", 1, None)  # ty: ignore[invalid-argument-type]
    assert "None" in str(e)
    assert len(str(e)) < 500


# --- 契約テスト。個々のバグではなく契約そのものを固定する。
# identify()/request() に悪意・破損したサーバメッセージを大量に流し込み、どれも
# 許容集合の外 (KeyError/TypeError/AttributeError/UnicodeDecodeError/TimeoutError
# などの素の例外) に漏れず、必ず ObsProtocolError の階層内に収まることを保証する。
# 次に見つかる新種の穴を防ぐのはこのテストの役目であって、個別テストの役目では
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
    (
        "authentication_challenge_not_utf8_encodable",
        {
            "op": 0,
            "d": {"authentication": {"salt": AUTH_SALT, "challenge": "\ud800"}},
        },
    ),
    # 値が偽 (falsy) でも authentication キーが present
    # なら「認証必要」と読まなければいけない。password は非空
    # ("irrelevant-password") で呼ばれるので、ここでの ObsIdentifyError は
    # 全部 shape 違反 (`{}`/`False`/`[]` は dict でも str/challenge を持つ dict
    # でもない) 由来。
    ("authentication_present_but_empty_dict", {"op": 0, "d": {"authentication": {}}}),
    ("authentication_present_but_false", {"op": 0, "d": {"authentication": False}}),
    ("authentication_present_but_empty_list", {"op": 0, "d": {"authentication": []}}),
    ("op_missing", {"d": {}}),
    # `op_huge_string` (`{"op": "x" * 500000, "d": {}}`) はここには含めない:
    # `pytest.raises(ObsProtocolError)` しか検査しないが、ObsIdentifyError は
    # ObsProtocolError のサブクラスなので、op の 2 箇所のガード
    # (Hello/Identified) を両方生の interpolation に戻しても素通りする
    # 「絶対に落ちないバッテリー行」になってしまう (確認済み: ガードを両方
    # 消しても 12 行すべて PASS のまま、うち op_huge_string も PASS)。実質の
    # 検査 (`len(str(e.value)) < 500`) は専用テスト
    # test_identify_error_message_is_bounded_when_op_is_a_huge_string_at_hello
    # が既に持っているので、ここに重複させず素通りする行を残さない。深い
    # ネストの方 (op_deeply_nested_object) は RecursionError を経由して実際に
    # 落ちるので、こちらは削除していない (raw frame 側の
    # IDENTIFY_HOSTILE_RAW_FRAMES を参照)。
]

IDENTIFY_HOSTILE_RAW_FRAMES: list[tuple[str, str | bytes]] = [
    ("invalid_json", "{not valid json"),
    ("json_array_instead_of_object", "[1, 2, 3]"),
    ("non_utf8_bytes_frame", b"\xff\xfe\x00\x01"),
    ("deeply_nested_json_array", _DEEPLY_NESTED_JSON_ARRAY),
    (
        "deeply_nested_json_in_envelope",
        '{"op":0,"d":{"x":' + _DEEPLY_NESTED_JSON_ARRAY + "}}",
    ),
    # 配列ではなくオブジェクトのネストで
    # ないと repr() の RecursionError の窓は開かない (定数の定義・解説は上の
    # セクション参照)。
    ("deeply_nested_json_object_missing_op", _DEEPLY_NESTED_JSON_OBJECT),
    (
        "deeply_nested_json_object_d_not_a_dict",
        _DEEPLY_NESTED_JSON_OBJECT_D_NOT_A_DICT,
    ),
    # op の値そのものを深くネストする。
    ("op_deeply_nested_object", _DEEPLY_NESTED_JSON_OBJECT_AS_OP),
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


def _deeply_nested_json_array(rid: str) -> str:
    return _DEEPLY_NESTED_JSON_ARRAY


def _response_data_not_a_dict(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {"result": True, "code": 100},
                "responseData": [1, 2, 3],
            },
        }
    )


def _code_not_an_int(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {"result": False, "code": "600", "comment": "nope"},
                "responseData": {},
            },
        }
    )


def _comment_not_a_string(rid: str) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": rid,
                "requestType": "X",
                "requestStatus": {
                    "result": False,
                    "code": 400,
                    "comment": {"nope": True},
                },
                "responseData": {},
            },
        }
    )


REQUEST_HOSTILE_RESPONSES: list[tuple[str, Callable[[str], str | bytes]]] = [
    ("requestStatus_missing", _request_status_missing),
    ("requestStatus_not_a_dict", _request_status_not_a_dict),
    ("op_missing", _op_missing),
    ("invalid_json", _invalid_json),
    ("json_array_instead_of_object", _json_array_instead_of_object),
    ("non_utf8_bytes_frame", _non_utf8_bytes_frame),
    ("deeply_nested_json_array", _deeply_nested_json_array),
    ("responseData_not_a_dict", _response_data_not_a_dict),
    # `code_not_an_int` / `comment_not_a_string` は含めない: ここは
    # `pytest.raises(ObsProtocolError)` しか検査しないが、未修正でも
    # ObsRequestError (ObsProtocolError のサブクラス) を投げて通ってしまう
    # 「絶対に落ちないバッテリー行」になってしまう (確認済み: ガードを消して
    # 確認)。実質の検査 (`assert not isinstance(e.value, ObsRequestError)`)
    # は専用テスト test_request_raises_obs_protocol_error_when_code_is_not_an_int
    # / ...comment_is_not_a_string が既に持っているので、ここに重複させず
    # 素通りする行を残さない。
    # オブジェクトのネストで repr() の窓を
    # 直接叩く (定数・builder の定義は上のセクション参照)。
    (
        "deeply_nested_json_object_requeststatus_not_a_dict",
        _deeply_nested_json_object_requeststatus_not_a_dict,
    ),
    ("deeply_nested_json_object_code", _deeply_nested_json_object_code),
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


# --- Measured against a real OBS 32.1.2 /
# obs-websocket 5.7.3 with a wrong `subtitle.obs.password`. obs-websocket
# does not reply to a rejected handshake with an error message -- it closes
# the WebSocket with code 4009 ("Authentication failed."). Every fake in
# this file above models auth failure as the client raising
# ObsIdentifyError itself; none of them ever close the socket instead, so
# none reproduced what the real server actually does. The old identify()
# let that ConnectionClosed (a WebSocketException, not an ObsIdentifyError)
# escape uncaught, so the subtitle worker's fail-open outer catch (which
# does catch WebSocketException) swallowed it and retried forever -- ADR-0042
# Alternatives-rejected #1 ("全て fail-open"), shipped by accident.


class _ClosingTransport:
    """A transport whose `recv()` raises a given `ConnectionClosed`
    immediately, instead of returning a Hello/Identified message -- mirrors
    a real obs-websocket server closing the socket on handshake rejection.
    """

    def __init__(self, exc: ConnectionClosed):
        self._exc = exc

    async def send(self, message: str) -> None:
        return None

    async def recv(self) -> str | bytes:
        raise self._exc

    async def close(self) -> None:
        pass


def _closed_with(code: int, reason: str = "") -> ConnectionClosed:
    """Build a `ConnectionClosed` the way `websockets` would for a close
    frame *received* from the peer with the given code/reason (optionally
    followed by this side echoing the same close back, which is what the
    real handshake-rejection log line -- "received 4009 ...; then sent 4009
    ..." -- shows). `rcvd` is what matters for rejection detection; `sent`
    only needs to be present to make `rcvd_then_sent` a valid combination.
    """
    close = Close(code, reason)
    return ConnectionClosed(rcvd=close, sent=close, rcvd_then_sent=True)


async def test_identify_raises_obs_identify_error_when_obs_closes_with_auth_rejected():
    # The measured case: OBS closes with 4009 "Authentication failed." on a
    # wrong password. identify() must convert this to ObsIdentifyError so
    # the subtitle worker's inner fail-loud catch
    # (ObsIdentifyError/ObsResourceNotFoundError) can turn it into
    # WorkerStartupError (ADR-0042) instead of retrying forever.
    transport = _ClosingTransport(_closed_with(4009, "Authentication failed."))
    with pytest.raises(ObsIdentifyError) as e:
        await ObsWsClient(transport).identify("wrong-password")
    assert "4009" in str(e.value)
    assert "Authentication failed." in str(e.value)


@pytest.mark.parametrize("code", [4000, 4999])
async def test_identify_raises_obs_identify_error_at_the_private_use_range_boundaries(
    code,
):
    # The private-use range identify() treats as a deliberate rejection is
    # 4000-4999 inclusive -- pin both ends so an off-by-one in the range
    # comparison (e.g. `4000 < code < 5000`, excluding the measured-adjacent
    # boundary values) cannot silently regress.
    transport = _ClosingTransport(_closed_with(code, "rejected"))
    with pytest.raises(ObsIdentifyError):
        await ObsWsClient(transport).identify("irrelevant-password")


@pytest.mark.parametrize("code", [3999, 5000])
async def test_identify_lets_a_close_outside_the_private_use_range_propagate(code):
    # The mirror case, the other boundary: codes just outside 4000-4999 (a
    # registered code like 3999, or a code past the private-use band like
    # 5000) are not obs-websocket's rejection signal and must stay a
    # retryable ConnectionClosed, not become a fatal ObsIdentifyError.
    transport = _ClosingTransport(_closed_with(code, "not a rejection"))
    with pytest.raises(ConnectionClosed):
        await ObsWsClient(transport).identify("irrelevant-password")


async def test_identify_lets_a_close_without_a_code_propagate():
    # A transport-level drop mid-handshake with no close frame ever received
    # (`rcvd is None`) -- not a rejection (there is nothing to reject with),
    # just a connection that isn't up yet or dropped abnormally. Getting
    # this backwards converts a retryable network blip into a fail-loud
    # WorkerStartupError one layer up, which is the mirror bug ADR-0042
    # warns about in the other direction.
    transport = _ClosingTransport(ConnectionClosed(rcvd=None, sent=None))
    with pytest.raises(ConnectionClosed):
        await ObsWsClient(transport).identify("irrelevant-password")
