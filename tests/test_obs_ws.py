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


# fix pass 4, finding 5: build_auth_string() が .encode("utf-8") を 2 回呼ぶ
# (salt 用と challenge 用) のに、salt 側のケースしか無かったテストの穴を埋める。
# 挙動そのものは fix pass 3 の try/except UnicodeError が両方を包んでいるので
# 既に直っている (このテストは pass/pass、境界ケースの取りこぼしではない)。


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


# --- fix pass 4, finding 4: `if auth:` は「真偽」を証明するが問うべきは「鍵の
# 有無」。`{}` / `[]` / `False` / `0` / `""` を「認証不要」と誤読して無認証の
# Identify を送ってしまい、OBS が実際には認証必須なら 4008 close ->
# WebSocketException で呼び出し側が延々リトライする (壊れたハンドシェイクは
# リトライしても直らない)。 ---


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


# --- fix pass 2, finding "Minor": 非 UTF-8 バイト列フレームが decode で漏れる ---


async def test_recv_raises_obs_protocol_error_on_non_utf8_bytes_frame():
    server = FakeObsServer(greet=False)
    server.inject_raw(b"\xff\xfe\x00\x01")
    with pytest.raises(ObsProtocolError):
        await ObsWsClient(server).identify("")


# --- fix pass 4, finding 1 (Critical): json.loads の RecursionError が _recv()
# から漏れる。JSONDecodeError (ValueError) は拾うが、深すぎるネスト
# ("[" * N + "]" * N のような ASCII 文字列だけで作れる) は RecursionError
# (RuntimeError のサブクラスで ValueError ではない) を投げる。websockets の
# デフォルト max_size (1 MiB) を素通りする transport-valid な入力なので何も
# 上流で弾かれない。identify() (bare / op-d envelope に包んだ両方) と
# request() の両方で実機再現した。---

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


# --- fix pass 4, finding 2 (Important): responseData が return 経路で無検査。
# requestStatus には isinstance(dict) 検査があるのに responseData には無く、
# 相手が list/str/int/bool/float を送ると型注釈 (-> dict[str, Any]) に反した
# 値をそのまま返してしまう。今日はここで例外にならないが、呼び出し側の最初の
# `result["inputSettings"]` で素の TypeError になる、同じバグ形の 1 フレーム
# 先送り。builder は後方で定義 (_response_data_not_a_dict)。---


async def test_request_raises_obs_protocol_error_when_response_data_is_not_a_dict():
    server = FakeObsServer(require_auth=False)
    client = ObsWsClient(server)
    await client.identify("")
    server.script_raw_response(_response_data_not_a_dict)
    with pytest.raises(ObsProtocolError) as e:
        await client.request("GetInputSettings", {"inputName": "x"})
    assert not isinstance(e.value, ObsRequestError)


# --- fix pass 4, finding 3 (Important): ObsRequestError の code/comment が
# 無検査。相手が code を "600" (str) で送ると `code == STATUS_RESOURCE_NOT_FOUND`
# が一致せず ObsResourceNotFoundError が汎用の ObsRequestError に降格し、
# 呼び出し側の「リソースが無い」専用の fail-loud 経路が発火しなくなる。
# comment が dict/None で来ると e.comment.lower() が素の AttributeError で
# 死ぬ。builder は後方で定義 (_code_not_an_int / _comment_not_a_string)。---


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


# --- fix pass 5, finding 1 (Critical): repr() 自体が RecursionError を漏らす。
# fix pass 4 で json.loads() の RecursionError は拾うようになったが、その後
# 「壊れたメッセージを報告する」ためにエラーメッセージの中で repr() を呼んで
# いる箇所 (message!r が 2 箇所 / auth!r が 2 箇所 / d!r / status!r /
# response_data!r、計 7 箇所) が同じ問題を持っていた: ネストの深い値の
# repr() 自体が RecursionError を送出し、ObsProtocolError を組み立てている
# 最中に素の RecursionError (RuntimeError のサブクラスで許容集合の外) が
# 漏れる。(fix pass 6 で message['op'] の 2 箇所が同じ形で見つかり、計 9 箇所
# になった — 該当セクションを参照。)
#
# 既存のテストはすべて **配列** ("[" * N + "]" * N) のネストを使っていたが、
# 配列は json.loads() 側が先に力尽きる (~11400 段) ので repr() まで生きて
# 届かず、この穴を暴けない。**オブジェクト** ('{"a":' * N + ... + '}' * N)
# は json.loads() が ~9600 段まで生き延びる一方 repr() は ~9000 段から壊れ
# 始めるので、9000-9600 段の窓でだけ再現する (残り C スタックに依存して
# 動く境界なので、これより深くすれば必ず壊れるという「魔法の深さ」として
# 扱わないこと)。9200 段は監査がこの窓の中で実測して渡した深さ。
_DEEPLY_NESTED_JSON_OBJECT = '{"a":' * 9200 + "1" + "}" * 9200

_DEEPLY_NESTED_JSON_OBJECT_D_NOT_A_DICT = (
    '{"op":0,"d":"not-a-dict","extra":' + _DEEPLY_NESTED_JSON_OBJECT + "}"
)

_DEEPLY_NESTED_JSON_OBJECT_AUTHENTICATION = (
    '{"op":0,"d":{"authentication":' + _DEEPLY_NESTED_JSON_OBJECT + "}}"
)


async def test_recv_raises_obs_protocol_error_on_deeply_nested_json_object_missing_op():
    # 監査が実機再現した入力そのもの: op キーの無い巨大な JSON オブジェクト。
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


# fix pass 5, finding 1 (Critical), 副次効果 (milder problem): auth /
# responseData が dict でないと判定される 2 箇所 (identify() の
# "authentication が不正な形" ガード、request() の "responseData が不正な形"
# ガード) は、判定に落ちる値自体が非 dict (JSON では文字列・数値・真偽値・
# 配列) でなければならない。配列は上の RecursionError の窓を開けない
# (json.loads() 側が先に力尽きる) ので、この 2 箇所は深いネストによる
# RecursionError では再現できない。だが同じ bare repr() は、ピアが巨大な
# 文字列を送れば例外メッセージ (ひいてはログ行) をそのまま巨大化させる、
# という別の (milder な) 形で漏れる。ここは深さではなく大きさでミューテー
# ションを殺す。
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


# --- fix pass 5, finding 3 (Minor): code が bool を素通りさせる。
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


# --- fix pass 6, finding 1 (Critical): `op`'s *value* is interpolated raw at
# the Hello guard and the Identified guard in identify() (the two
# `op={message['op']} が来た` f-strings). `_recv()` only proves
# `"op" in message` — nothing constrains the *value*, and an f-string's
# `{x}` calls format() -> dict.__repr__ exactly like `!r` does, so this is
# the same repr()-recursion / unbounded-size hazard fix pass 5 closed for
# message/auth/d/status/response_data, just spelled without `!r` (which is
# why an audit shaped around bare `repr()`/`!r` call sites missed it).
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


# --- fix pass 6, finding 2 (Important): ObsRequestError.comment is the last
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


# --- fix pass 6, finding 4: peer-controlled-field enumeration and coverage.
# Fields whose *value* the peer picks that reach a message or a comparison
# in this module:
#   op (message['op'])             -- newly covered above (message, 2 sites)
#                                      and below (comparison, request() side)
#   d (message['d'])                -- already covered (fix pass 4/5):
#                                      comparison (isinstance) + message
#   authentication / auth           -- already covered (fix pass 2-5):
#                                      comparison (isinstance) + message
#   salt / challenge                -- already covered (fix pass 3-4):
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
#   requestStatus / status          -- already covered (fix pass 4/5):
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
#                                      (finding 2)
#   responseData                    -- already covered (fix pass 4/5):
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


# --- fix pass 8: `code` was the one peer-controlled value that reached a
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


# --- fix pass 7, finding 1 (Important): `_SAFE_REPR` bounds nesting depth
# (`maxlevel`) and each leaf's size (`maxstring`/`maxother`), but never bounded
# the *total* rendered length. Within `maxlevel=6`, the default width caps
# (`maxlist=6`/`maxdict=4`) still allow up to ~6**6 leaves at ~200 chars each,
# so a wide-but-shallow structure blows past any "leaf is bounded" guarantee
# without ever touching the RecursionError depth window the fix pass 4/5/6
# tests already cover. Every existing "deeply nested" fixture in this file
# varies *depth*; this section varies *width* instead, which is the axis that
# let the bug through undetected.
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
# 約 328 KB のフレームから約 262 KB の例外メッセージが組み上がる (fix pass 7
# の監査が実機の identify() で再現した「265 KB のフレーム -> 262 KB の
# 例外」と同じ桁数)。
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
    # 同じ根っこ、別の枝: fix pass 6 で足した comment の 200 文字スライスは
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


# --- fix pass 7, finding 3 (Minor): `ObsRequestError.__init__` now calls
# `len(comment)`, which is not total for a non-`str` `comment`. Both
# peer-reachable construction sites in this module gate on
# `isinstance(comment, str)` first, so a hostile peer can never reach this —
# but the class is public, so a caller constructing it directly with e.g.
# `None` (as older code in this exact style used to allow) must not get a
# raw `TypeError` in exchange.
def test_obs_request_error_is_constructible_with_a_non_string_comment():
    e = ObsRequestError("X", 1, None)  # ty: ignore[invalid-argument-type]
    assert "None" in str(e)
    assert len(str(e)) < 500


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
    (
        "authentication_challenge_not_utf8_encodable",
        {
            "op": 0,
            "d": {"authentication": {"salt": AUTH_SALT, "challenge": "\ud800"}},
        },
    ),
    # fix pass 4, finding 4: 値が偽 (falsy) でも authentication キーが present
    # なら「認証必要」と読まなければいけない。password は非空
    # ("irrelevant-password") で呼ばれるので、ここでの ObsIdentifyError は
    # 全部 shape 違反 (`{}`/`False`/`[]` は dict でも str/challenge を持つ dict
    # でもない) 由来。
    ("authentication_present_but_empty_dict", {"op": 0, "d": {"authentication": {}}}),
    ("authentication_present_but_false", {"op": 0, "d": {"authentication": False}}),
    ("authentication_present_but_empty_list", {"op": 0, "d": {"authentication": []}}),
    ("op_missing", {"d": {}}),
    # fix pass 6, finding 1 (Critical) は元々ここに `op_huge_string` 行
    # (`{"op": "x" * 500000, "d": {}}`) を足していたが、fix pass 7, finding 2
    # (Important) で削除した。ここは `pytest.raises(ObsProtocolError)` しか
    # 検査しないが、ObsIdentifyError は ObsProtocolError のサブクラスなので、
    # op の 2 箇所のガード (Hello/Identified) を両方 fix pass 6 前の生
    # interpolation に revert しても素通りする「絶対に落ちないバッテリー行」
    # だったとミューテーションテストで確認済み (ガードを両方消して確認:
    # 12 行すべて PASS のまま、うち op_huge_string も PASS)。実質の検査
    # (`len(str(e.value)) < 500`) は専用テスト
    # test_identify_error_message_is_bounded_when_op_is_a_huge_string_at_hello
    # が既に持っているので、ここに重複させず素通りする行を残さない
    # (fix pass 5, finding 2 で code_not_an_int/comment_not_a_string に
    # 下した判断と同じ形)。深いネストの方 (op_deeply_nested_object) は
    # RecursionError を経由して実際に落ちるので、こちらは削除していない
    # (raw frame 側の IDENTIFY_HOSTILE_RAW_FRAMES を参照)。
]

IDENTIFY_HOSTILE_RAW_FRAMES: list[tuple[str, str | bytes]] = [
    ("invalid_json", "{not valid json"),
    ("json_array_instead_of_object", "[1, 2, 3]"),
    ("non_utf8_bytes_frame", b"\xff\xfe\x00\x01"),
    # fix pass 4, finding 1.
    ("deeply_nested_json_array", _DEEPLY_NESTED_JSON_ARRAY),
    (
        "deeply_nested_json_in_envelope",
        '{"op":0,"d":{"x":' + _DEEPLY_NESTED_JSON_ARRAY + "}}",
    ),
    # fix pass 5, finding 1 (Critical): 配列ではなくオブジェクトのネストで
    # ないと repr() の RecursionError の窓は開かない (定数の定義・解説は上の
    # fix pass 5 セクション参照)。
    ("deeply_nested_json_object_missing_op", _DEEPLY_NESTED_JSON_OBJECT),
    (
        "deeply_nested_json_object_d_not_a_dict",
        _DEEPLY_NESTED_JSON_OBJECT_D_NOT_A_DICT,
    ),
    # fix pass 6, finding 1 (Critical): op の値そのものを深くネストする。
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
    # fix pass 5, finding 2 (Important): code_not_an_int / comment_not_a_string
    # は削除した。ここは `pytest.raises(ObsProtocolError)` しか検査しないが、
    # 未修正でも ObsRequestError (ObsProtocolError のサブクラス) を投げて
    # 通ってしまう「絶対に落ちないバッテリー行」だったとミューテーション
    # テストで確認済み (ガードを消して確認)。実質の検査
    # (`assert not isinstance(e.value, ObsRequestError)`) は専用テスト
    # test_request_raises_obs_protocol_error_when_code_is_not_an_int /
    # ...comment_is_not_a_string が既に持っているので、ここに重複させず
    # 素通りする行を残さない。
    # fix pass 5, finding 1 (Critical): オブジェクトのネストで repr() の窓を
    # 直接叩く (定数・builder の定義は上の fix pass 5 セクション参照)。
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
