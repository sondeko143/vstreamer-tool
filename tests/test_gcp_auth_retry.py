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
