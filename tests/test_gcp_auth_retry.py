"""GCP の OAuth トークン更新が、プールされた死んだ接続を生き延びること。

本番で観測した失敗:

    ConnectionResetError(10054, '既存の接続はリモート ホストに強制的に切断されました。')
    -> google.auth.exceptions.TransportError
    -> 503 Getting metadata from plugin failed

トークンの有効期限ぶん (約 1 時間) idle した TLS 接続がプールに残り、Google 側は
既に閉じているので、次の更新でそれを掴む。urllib3 は貸し出し前に
`is_connection_dropped()` で死活を見るが、それで拾えるのは「貸す前に既に死んで
いた」接続だけで、本番の traceback は `getresponse()` -> `recv_into` で落ちて
いる = リクエストを書いた後に RST が来ている。この窓は死活チェックでは塞げず、
トランスポート層の retry でしか救えない。
"""

import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

import pytest
import requests

pytest.importorskip("google.cloud.translate_v3")

from vspeech.lib.gcp import build_auth_session  # noqa: E402

TOKEN_BODY = b'{"access_token": "ok", "expires_in": 3599, "token_type": "Bearer"}'


class _ResetOnSecondRequest(BaseHTTPRequestHandler):
    """1 回目は普通に応答して keep-alive 接続をプールに残し、2 回目は
    リクエストを受け切ってから応答せずに RST する (= 本番の窓)。3 回目以降
    (retry が張り直した新しい接続) は普通に応答する。"""

    protocol_version = "HTTP/1.1"
    counter: list[int] = []

    def do_POST(self) -> None:
        type(self).counter.append(1)
        n = len(type(self).counter)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if n == 2:
            # SO_LINGER 0 にして FIN ではなく RST を送る -- 本番と同じ
            # WinError 10054 をクライアントに観測させるため。
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.connection.close()
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(TOKEN_BODY)))
        self.end_headers()
        self.wfile.write(TOKEN_BODY)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def token_endpoint():
    _ResetOnSecondRequest.counter = []
    # ThreadingHTTPServer でないと keep-alive 接続を掴んだまま retry の新規
    # 接続を accept できず、テストがデッドロックする。
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResetOnSecondRequest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/token"
    finally:
        server.shutdown()
        server.server_close()


def _refresh_twice(session, url):
    """更新 2 回。1 回目で接続がプールに入り、2 回目がその死んだ接続を掴む。"""
    body = {"grant_type": "refresh_token"}
    session.post(url, data=body, timeout=5).raise_for_status()
    return session.post(url, data=body, timeout=5)


def test_plain_session_fails_on_a_reset_pooled_connection(token_endpoint):
    """前提の確認: google.auth が既定で作る素の Session はここで落ちる。

    これが落ちなくなったら (requests/urllib3 側がこの窓を塞いだら)、下の
    テストは何も証明しなくなるので、一緒に置いておく。
    """
    with pytest.raises(requests.exceptions.ConnectionError):
        _refresh_twice(requests.Session(), token_endpoint)


def test_auth_session_survives_a_reset_pooled_connection(token_endpoint):
    response = _refresh_twice(build_auth_session(), token_endpoint)
    assert response.status_code == 200
    # 3 = 成功, RST, retry が新しい接続で成功。retry が起きた証拠。
    assert len(_ResetOnSecondRequest.counter) == 3


def test_auth_plugin_refreshes_over_the_retrying_session():
    """plugin が実際に retry 付き session でトークンを更新すること。

    これが無いと、修正の本体である
    `Request(session=build_auth_session())` を `Request()` に戻しても
    全テストが GREEN のままになる (実測) -- つまり本番のバグをそのまま
    復活させても誰も気付かない。他のテストは
    「session 単体の性質」と「チャネルの同一性」しか見ておらず、その 2 つを
    繋ぐ継ぎ目がどこにも縛られていなかった。

    アダプタは **https:// で** 引く。他のテストはローカルの http:// サーバを
    叩くので、https:// 側だけ retry の無いアダプタに差し替えても
    素通りしてしまう (これも実測)。実運用のトークンエンドポイントは https。
    """
    from google.auth.credentials import AnonymousCredentials

    from vspeech.lib.gcp import _AUTH_RETRY
    from vspeech.lib.gcp import create_auth_metadata_plugin

    plugin = create_auth_metadata_plugin(
        AnonymousCredentials(),
        host="translate.googleapis.com",
        scopes=("https://www.googleapis.com/auth/cloud-translation",),
    )
    adapter = plugin._request.session.get_adapter("https://oauth2.googleapis.com/token")
    assert adapter.max_retries is _AUTH_RETRY


def test_auth_plugin_passes_the_default_host():
    """`default_host` が plugin まで届いていること。

    これを落としてもサービスアカウントの self-signed JWT 経路が変わるだけで
    他のテストは全部 GREEN のまま通る (実測)。`create_auth_channel` の呼び出し
    引数を見ているテストはあるが、それは「渡した値」を見ているだけで、
    plugin に届いたことは誰も確かめていなかった。
    """
    from google.auth.credentials import AnonymousCredentials

    from vspeech.lib.gcp import create_auth_metadata_plugin

    plugin = create_auth_metadata_plugin(
        AnonymousCredentials(), host="translate.googleapis.com", scopes=()
    )
    assert plugin._default_host == "translate.googleapis.com"


class _ServiceUnavailableWithRetryAfter(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    BODY = b'{"error": "unavailable"}'

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Retry-After", "1")
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()
        self.wfile.write(self.BODY)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_auth_session_does_not_swallow_retry_after_responses():
    """Retry-After 付きの 503 を、応答のまま google.auth へ返すこと。

    urllib3 の `is_retry()` は respect_retry_after_header (既定 True) が有効で
    Retry-After ヘッダが付いていると、status_forcelist を一切見ずに「retry
    すべき」と判断する。そこへ status=0 が重なると 0 -> -1 で即 is_exhausted に
    なり、応答が「retry もされず本文も失われた RetryError」に化ける。

    429/503 は google.auth 自身が `_client` の ExponentialBackoff で retry する
    対象なので、これを握り潰すと接続層の窓を塞ぐ代わりに google.auth が持って
    いた retry を奪う -- 直したつもりで可用性を下げる取り違え。
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ServiceUnavailableWithRetryAfter)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        response = build_auth_session().post(
            f"http://127.0.0.1:{server.server_address[1]}/token",
            data={"grant_type": "refresh_token"},
            timeout=5,
        )
        assert response.status_code == 503
        assert b"unavailable" in response.content
    finally:
        server.shutdown()
        server.server_close()


def test_auth_session_caps_each_attempt_timeout():
    """1 試行あたりの時間に上限をかけていること。

    google.auth はトークン更新の POST に timeout を渡さず、`Request.__call__` の
    既定 120 秒が効く。そこへ retry を足すと最悪時間が試行回数ぶん伸び、応答を
    返さないエンドポイント相手では gRPC の認証スレッドが積み上がる。この上限が、
    retry を足しても最悪時間を有界に保つ当の仕掛けなので、ここで固定する。
    """
    from unittest.mock import patch

    from requests import Request as RequestsRequest
    from requests.adapters import HTTPAdapter

    from vspeech.lib.gcp import _AUTH_REQUEST_TIMEOUT_SEC
    from vspeech.lib.gcp import build_auth_session

    adapter = build_auth_session().get_adapter("https://oauth2.googleapis.com/token")
    prepared = RequestsRequest("POST", "https://oauth2.googleapis.com/token").prepare()

    seen: list[float] = []
    with patch.object(HTTPAdapter, "send", return_value=None) as parent_send:
        # google.auth が実際に渡してくる形 (既定 120 秒) と、渡されない形。
        for given in (120.0, None, (120.0, 120.0)):
            adapter.send(prepared, timeout=given)
        for call in parent_send.call_args_list:
            value = call.kwargs["timeout"]
            seen.extend(value if isinstance(value, tuple) else [value])

    assert seen, "adapter.send never reached the parent"
    assert all(v <= _AUTH_REQUEST_TIMEOUT_SEC for v in seen)
    # 上限が実質無しに緩められたら落ちるよう、絶対値でも縛る。
    assert _AUTH_REQUEST_TIMEOUT_SEC <= 30.0


def test_compute_engine_id_token_credentials_use_the_retrying_session(monkeypatch):
    """CE credentials の Request も retry 付きであること。

    ここの Request は `iam.Signer` に抱え込まれ、更新のたびにメタデータ GET と
    iamcredentials への signBlob POST を、トークン POST より **先に** 行う。
    素の Request() だとその 2 本が retry 無しのままで、1 回の更新に走る 3 本の
    うち 1 本しか直っていない状態になる (実測でそうなっていた)。
    """
    from typing import Any

    from vspeech.config import GcpConfig
    from vspeech.lib import gcp
    from vspeech.lib.gcp import _AUTH_RETRY

    captured: dict[str, Any] = {}

    def fake_ce_id_token_credentials(request, target_audience):
        captured["request"] = request
        return object()

    monkeypatch.setattr(gcp, "CeIdTokenCredentials", fake_ce_id_token_credentials)
    config = GcpConfig()
    config.use_ce_credentials = True
    gcp.get_id_token_credentials(config)

    session = captured["request"].session
    adapter = session.get_adapter("https://iamcredentials.googleapis.com/")
    assert adapter.max_retries is _AUTH_RETRY
    # メタデータサーバは http:// なので、そちらにも同じ adapter が要る。
    assert session.get_adapter("http://metadata.google.internal/") is adapter


def test_create_auth_channel_requires_a_running_loop():
    """実行中のループが無いまま呼んだら声を上げること。

    grpc.aio のチャネルは生成時点のループに束縛されるが、ループが走って
    いなくても例外にはならず、走っていない新しいループが黙って結び付き、
    以後の RPC が永久に返らなくなる。preflight (ADR-0045) は同期なので、
    そこに GCP の生存確認を足した誰かが静かに踏みうる。
    """
    from google.auth.credentials import AnonymousCredentials

    from vspeech.lib.gcp import create_auth_channel

    with pytest.raises(RuntimeError):
        create_auth_channel(
            AnonymousCredentials(), host="translate.googleapis.com", scopes=()
        )


def test_auth_plugin_applies_the_service_scopes():
    """api_core と同じ scopes を credentials に適用していること。

    `_create_composite_credentials` を手で組み直した以上、api_core がやって
    いて我々が落とした処理があれば本番だけで壊れる。中でも
    `with_scopes_if_required` は、落としてもここまでの他のテストが全部 GREEN の
    まま (AnonymousCredentials は Scoped ではないので何も起きない) なのに、
    実際のサービスアカウントではスコープ無しトークンになって拒否される --
    このプロジェクトの認証情報を持たないマシンでは実接続で検出できない種類の
    退行なので、型どおりに検査しておく。
    """
    from google.auth.credentials import Credentials as BaseCredentials
    from google.auth.credentials import Scoped

    from vspeech.lib.gcp import create_auth_metadata_plugin

    class FakeScopedCredentials(BaseCredentials, Scoped):
        def __init__(self):
            super().__init__()
            self.asked_default_scopes = None

        @property
        def requires_scopes(self):
            return True

        def with_scopes(self, scopes, default_scopes=None):
            self.asked_default_scopes = default_scopes
            return self

        def refresh(self, request):  # pragma: no cover - never called here
            raise AssertionError("refresh must not happen at construction time")

    credentials = FakeScopedCredentials()
    scopes = ("https://www.googleapis.com/auth/cloud-translation",)
    create_auth_metadata_plugin(
        credentials, host="translate.googleapis.com", scopes=scopes
    )

    assert credentials.asked_default_scopes == scopes


async def test_translation_client_authenticates_through_the_retrying_channel(
    monkeypatch,
):
    """翻訳クライアントの認証が `create_auth_channel` の産物であること。

    上の 2 つは session 単体の性質しか見ていないので、これが無いと
    `TranslationServiceAsyncClient(credentials=...)` の 1 行に戻されたときに
    (retry 付き session が誰にも使われなくなっても) 全部 GREEN のままになる。
    チャネル生成には動いているイベントループが要るので async テスト。
    """
    from typing import Any

    from google.auth.credentials import AnonymousCredentials
    from google.cloud.translate_v3.services.translation_service.transports import (
        TranslationServiceGrpcAsyncIOTransport,
    )

    from vspeech.worker import translation

    built: dict[str, Any] = {}
    original = translation.create_auth_channel

    def spy(credentials, host, scopes, options=()):
        channel = original(credentials, host=host, scopes=scopes, options=options)
        built.update(channel=channel, host=host, scopes=scopes, options=options)
        return channel

    monkeypatch.setattr(translation, "create_auth_channel", spy)
    client = translation.create_translation_client(AnonymousCredentials())

    transport = client.transport
    assert isinstance(transport, TranslationServiceGrpcAsyncIOTransport)
    assert transport.grpc_channel is built["channel"]
    assert built["host"] == "translate.googleapis.com"
    assert any("cloud-translation" in s for s in built["scopes"])
    # transport が自前でチャネルを作るときの options は、チャネルを渡した
    # 時点で適用されなくなる。渡し直さないと受信上限が gRPC 既定の 4 MiB に
    # 戻るので、ここで運ばれていることを縛る。
    assert dict(built["options"])["grpc.max_receive_message_length"] == -1


async def test_speech_client_authenticates_through_the_retrying_channel(monkeypatch):
    """transcription の GCP バックエンドも同じ認証チャネルに載っていること。

    translation と同型。`SpeechAsyncClient(credentials=...)` の 1 行に戻されても
    他のテストは全部 GREEN のままなので、ここで縛る。
    """
    from typing import Any

    from google.auth.credentials import AnonymousCredentials
    from google.cloud.speech_v1.services.speech.transports import (
        SpeechGrpcAsyncIOTransport,
    )

    from vspeech.worker import transcription

    built: dict[str, Any] = {}
    original = transcription.create_auth_channel

    def spy(credentials, host, scopes, options=()):
        channel = original(credentials, host=host, scopes=scopes, options=options)
        built.update(channel=channel, host=host, scopes=scopes, options=options)
        return channel

    monkeypatch.setattr(transcription, "create_auth_channel", spy)
    client = transcription.create_speech_client(AnonymousCredentials())

    transport = client.transport
    assert isinstance(transport, SpeechGrpcAsyncIOTransport)
    assert transport.grpc_channel is built["channel"]
    assert built["host"] == "speech.googleapis.com"
    # scopes まで見ること。Speech は cloud-platform だけ、Translate は
    # cloud-translation も持つ別物なので、host だけ見ていると Translate の
    # scopes を渡す取り違えが素通りする (共有定数の導入でその取り違えは
    # 起きやすくなっている)。
    assert built["scopes"] == SpeechGrpcAsyncIOTransport.AUTH_SCOPES
    assert dict(built["options"])["grpc.max_receive_message_length"] == -1


async def _start_and_stop(generator):
    """ワーカーの async generator をクライアント構築の先まで進めて止める。

    どのワーカーも `worker_startup` の中でクライアントを作り、直後に
    `in_queue.get()` で待つので、短い timeout で 1 回 __anext__ を試せば
    「構築だけ済んだ状態」に到達できる。
    """
    from asyncio import wait_for

    # timeout はどれだけ短くてもよい: ワーカーは最初の await に達する前に
    # 同期でクライアントを構築し、その後 `in_queue.get()` で永久に待つので、
    # この wait_for は必ずタイムアウトする (= 決まった長さの sleep)。1 秒に
    # すると 2 テストで 2 秒を捨てるだけなので短く取る。
    try:
        await wait_for(generator.__anext__(), timeout=0.05)
    except TimeoutError:
        pass
    finally:
        await generator.aclose()


async def test_translation_worker_builds_its_client_through_the_factory(monkeypatch):
    """ワーカーが実際に `create_translation_client` を通ること。

    ファクトリ単体のテストだけでは、**呼び出し側**が
    `TranslationServiceAsyncClient(credentials=...)` の 1 行に戻されたときに
    素通りする (実測: ファクトリを残したまま呼び出し箇所だけ戻すと全部 GREEN)。
    直った経路が実際に使われていることまで縛る。
    """
    from asyncio import Queue

    from vspeech.config import GcpConfig
    from vspeech.config import TranslationConfig
    from vspeech.worker import translation

    used: dict[str, bool] = {}
    monkeypatch.setattr(translation, "get_credentials", lambda cfg: (object(), "proj"))
    monkeypatch.setattr(
        translation,
        "create_translation_client",
        lambda credentials: used.setdefault("via_factory", True) and object(),
    )
    await _start_and_stop(
        translation.translation_worker_google(
            config=TranslationConfig(), gcp_config=GcpConfig(), in_queue=Queue()
        )
    )
    assert used.get("via_factory") is True


async def test_transcription_worker_builds_its_client_through_the_factory(monkeypatch):
    """transcription 側も同じ (上のテストと同じ理由)。"""
    from asyncio import Queue

    from vspeech.config import GcpConfig
    from vspeech.config import TranscriptionConfig
    from vspeech.worker import transcription

    used: dict[str, bool] = {}
    monkeypatch.setattr(transcription, "get_credentials", lambda cfg: (object(), ""))
    monkeypatch.setattr(
        transcription,
        "create_speech_client",
        lambda credentials: used.setdefault("via_factory", True) and object(),
    )
    config = TranscriptionConfig()
    config.vad_gate = False  # モデル実体を要求しない
    await _start_and_stop(
        transcription.transcript_worker_google(
            config=config, gcp_config=GcpConfig(), in_queue=Queue()
        )
    )
    assert used.get("via_factory") is True


async def test_sender_secure_channel_authenticates_with_the_retrying_session(
    monkeypatch,
):
    """sender の ID トークン経路を、**本物の** チャネル組み立てごと検証する。

    差し替えるのは grpc の `secure_channel` だけに留め、認証プラグインの組み立て
    (`AuthMetadataPlugin(credentials, request)`) は本物を走らせる。こうしないと
    その組み立てが assertion の下を通らず、`AuthMetadataPlugin(credentials,
    Request())` のような本番のバグを取りこぼす。

    同時に「認証がチャネルに載っていること」も見る: `composite_credentials` を
    `ssl_credentials` だけにする (= 資格情報を一切送らなくなる) 変異を捕まえる。
    """
    from typing import Any

    from vspeech.lib.gcp import _AUTH_RETRY
    from vspeech.worker import sender

    captured: dict[str, Any] = {}
    real_plugin = sender.AuthMetadataPlugin
    real_composite = sender.composite_channel_credentials

    def spy_plugin(credentials, request):
        captured["plugin_request"] = request
        return real_plugin(credentials, request)

    def spy_composite(ssl_credentials, call_credentials):
        composed = real_composite(ssl_credentials, call_credentials)
        captured["composed"] = composed
        return composed

    def fake_secure_channel(target, credentials, options=None):
        captured["channel_credentials"] = credentials
        return object()

    monkeypatch.setattr(sender, "AuthMetadataPlugin", spy_plugin)
    monkeypatch.setattr(sender, "composite_channel_credentials", spy_composite)
    monkeypatch.setattr(sender, "secure_channel", fake_secure_channel)

    class FakeIdTokenCredentials:
        def with_target_audience(self, audience):
            return self

        def refresh(self, request):
            captured["refresh_request"] = request

    from typing import cast

    from vspeech.lib.gcp import GcpIDTokenCredentials

    await sender.get_channel(
        "https://securehost/", cast(GcpIDTokenCredentials, FakeIdTokenCredentials())
    )

    # 以後の更新を行う plugin が retry 付き session を持っていること。
    plugin_request = captured["plugin_request"]
    adapter = plugin_request.session.get_adapter("https://oauth2.googleapis.com/token")
    assert adapter.max_retries is _AUTH_RETRY
    # 初回 refresh と plugin が同じ Request を共有していること。
    assert captured["refresh_request"] is plugin_request
    # 認証がチャネルに実際に載っていること (ssl だけに落ちていない)。
    assert captured["channel_credentials"] is captured["composed"]
