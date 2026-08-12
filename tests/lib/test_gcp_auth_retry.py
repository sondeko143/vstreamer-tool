"""GCP's OAuth token refresh survives a dead pooled connection.

The failure observed in production (the quoted message is the Windows text verbatim):

    ConnectionResetError(10054, '既存の接続はリモート ホストに強制的に切断されました。')
    -> google.auth.exceptions.TransportError
    -> 503 Getting metadata from plugin failed

A TLS connection that sat idle for the token lifetime (about an hour) remains in the pool
and Google has already closed it, so the next refresh grabs it. urllib3 checks liveness
with `is_connection_dropped()` before handing it out, but that only catches connections
that were already dead before the loan, whereas the production traceback dies in
`getresponse()` -> `recv_into` = the RST arrives after the request was written. That
window cannot be closed by a liveness check; only a transport-layer retry can save it.
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
    """The first request is answered normally, leaving a keep-alive connection in the
    pool; the second reads the request fully and then RSTs without answering (= the
    production window). From the third on (the fresh connection the retry establishes) it
    answers normally again."""

    protocol_version = "HTTP/1.1"
    counter: list[int] = []

    def do_POST(self) -> None:
        type(self).counter.append(1)
        n = len(type(self).counter)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if n == 2:
            # Set SO_LINGER 0 so an RST is sent instead of a FIN -- so the client observes
            # the same WinError 10054 as in production.
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
    # Without a ThreadingHTTPServer it cannot accept the retry's new connection while
    # still holding the keep-alive one, and the test deadlocks.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResetOnSecondRequest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/token"
    finally:
        server.shutdown()
        server.server_close()


def _refresh_twice(session, url):
    """Two refreshes. The first puts a connection in the pool; the second grabs that dead
    connection."""
    body = {"grant_type": "refresh_token"}
    session.post(url, data=body, timeout=5).raise_for_status()
    return session.post(url, data=body, timeout=5)


def test_plain_session_fails_on_a_reset_pooled_connection(token_endpoint):
    """Confirms the premise: the bare Session google.auth builds by default dies here.

    If this stops failing (i.e. requests/urllib3 closes the window), the test below proves
    nothing any more, so it is kept alongside it.
    """
    with pytest.raises(requests.exceptions.ConnectionError):
        _refresh_twice(requests.Session(), token_endpoint)


def test_auth_session_survives_a_reset_pooled_connection(token_endpoint):
    response = _refresh_twice(build_auth_session(), token_endpoint)
    assert response.status_code == 200
    # 3 = success, RST, then the retry succeeding on a new connection. Proof a retry
    # happened.
    assert len(_ResetOnSecondRequest.counter) == 3


def test_auth_plugin_refreshes_over_the_retrying_session():
    """The plugin really does refresh the token over the session with retries.

    Without this, reverting the substance of the fix --
    `Request(session=build_auth_session())` back to `Request()` -- leaves every test GREEN
    (measured), i.e. the production bug could be reinstated and nobody would notice. The
    other tests only look at "properties of the session alone" and "identity of the
    channel", leaving the seam between those two unpinned.

    Look the adapter up **under https://**. The other tests hit a local http:// server, so
    replacing only the https:// side with a retry-less adapter would slip through (also
    measured). The real token endpoint is https.
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
    """`default_host` reaches the plugin.

    Dropping it only changes the service account's self-signed JWT path, and every other
    test still passes GREEN (measured). A test does look at `create_auth_channel`'s call
    arguments, but that only observes "the value passed in" -- nobody was checking that it
    reached the plugin.
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
    """A 503 with Retry-After is handed back to google.auth as the response it is.

    When respect_retry_after_header (default True) is on and a Retry-After header is
    present, urllib3's `is_retry()` decides "should retry" without consulting
    status_forcelist at all. Combined with status=0 that goes 0 -> -1 and is immediately
    exhausted, so the response mutates into "a RetryError that was never retried and lost
    its body".

    429/503 are exactly what google.auth itself retries through `_client`'s
    ExponentialBackoff, so swallowing them steals the retry google.auth had in exchange
    for closing the connection-layer window -- a mix-up that lowers availability while
    appearing to fix something.
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
    """A ceiling is applied to the time of each attempt.

    google.auth passes no timeout to the token-refresh POST, so `Request.__call__`'s
    default of 120 seconds applies. Adding retries multiplies the worst case by the number
    of attempts, and against an endpoint that never answers, gRPC's auth threads pile up.
    This ceiling is the very mechanism that keeps the worst case bounded despite the
    retries, so it is pinned here.
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
        # The shapes google.auth actually passes (the 120-second default) and the absent
        # one.
        for given in (120.0, None, (120.0, 120.0)):
            adapter.send(prepared, timeout=given)
        for call in parent_send.call_args_list:
            value = call.kwargs["timeout"]
            seen.extend(value if isinstance(value, tuple) else [value])

    assert seen, "adapter.send never reached the parent"
    assert all(v <= _AUTH_REQUEST_TIMEOUT_SEC for v in seen)
    # Also bound it in absolute terms so relaxing the ceiling to effectively nothing
    # fails.
    assert _AUTH_REQUEST_TIMEOUT_SEC <= 30.0


def test_compute_engine_id_token_credentials_use_the_retrying_session(monkeypatch):
    """The CE credentials' Request carries retries too.

    That Request is held by `iam.Signer` and, on every refresh, issues a metadata GET and a
    signBlob POST to iamcredentials **before** the token POST. With a bare Request() those
    two have no retries and only one of the three calls per refresh is fixed (which is
    what was measured).
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
    # The metadata server is http://, so it needs the same adapter too.
    assert session.get_adapter("http://metadata.google.internal/") is adapter


def test_create_auth_channel_requires_a_running_loop():
    """Calling it with no running loop must speak up.

    A grpc.aio channel binds to the loop at construction time, but with no loop running it
    does not raise: a new, not-running loop is silently bound and every later RPC hangs
    forever. preflight is synchronous, so whoever adds a GCP liveness check
    there could quietly step into it.
    """
    from google.auth.credentials import AnonymousCredentials

    from vspeech.lib.gcp import create_auth_channel

    with pytest.raises(RuntimeError):
        create_auth_channel(
            AnonymousCredentials(), host="translate.googleapis.com", scopes=()
        )


def test_auth_plugin_applies_the_service_scopes():
    """The same scopes api_core applies are applied to the credentials.

    Having rebuilt `_create_composite_credentials` by hand, anything api_core does that we
    dropped breaks in production only. `with_scopes_if_required` in particular can be
    dropped with every other test above still GREEN (AnonymousCredentials is not Scoped, so
    nothing happens), while a real service account would fetch a scopeless token and be
    rejected -- a regression that cannot be detected by a real connection on a machine
    without this project's credentials, so it is checked structurally.
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
    """The translation client's auth is the product of `create_auth_channel`.

    The two tests above only look at properties of the session alone, so without this,
    reverting to the one-line `TranslationServiceAsyncClient(credentials=...)` leaves
    everything GREEN even though nobody uses the retrying session any more. Channel
    construction needs a running event loop, hence an async test.
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
    # The options the transport uses when it builds the channel itself stop applying the
    # moment we hand it a channel. Without passing them again the receive limit reverts to
    # gRPC's default 4 MiB, so this pins that they are carried through.
    assert dict(built["options"])["grpc.max_receive_message_length"] == -1


async def test_speech_client_authenticates_through_the_retrying_channel(monkeypatch):
    """The transcription GCP backend rides the same auth channel.

    The same shape as translation. Reverting to the one-line
    `SpeechAsyncClient(credentials=...)` leaves every other test GREEN, so it is pinned
    here.
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
    # Check the scopes too. Speech has cloud-platform only while Translate also has
    # cloud-translation, so watching the host alone would let a mix-up that passes
    # Translate's scopes slip through (and introducing a shared constant makes that mix-up
    # more likely).
    assert built["scopes"] == SpeechGrpcAsyncIOTransport.AUTH_SCOPES
    assert dict(built["options"])["grpc.max_receive_message_length"] == -1


async def _start_and_stop(generator):
    """Advance the worker's async generator past client construction and stop it.

    Every worker builds its client inside `worker_startup` and then immediately waits in
    `in_queue.get()`, so a single __anext__ attempt with a short timeout reaches the state
    where construction alone has happened.
    """
    from asyncio import wait_for

    # The timeout can be arbitrarily short: the worker builds the client synchronously
    # before reaching its first await and then waits forever in `in_queue.get()`, so this
    # wait_for always times out (i.e. it is a sleep of a fixed length). Making it 1 second
    # would just throw away 2 seconds across the two tests, so keep it short.
    try:
        await wait_for(generator.__anext__(), timeout=0.05)
    except TimeoutError:
        pass
    finally:
        await generator.aclose()


async def test_translation_worker_builds_its_client_through_the_factory(monkeypatch):
    """The worker really does go through `create_translation_client`.

    A test of the factory alone slips through when the **call site** is reverted to the
    one-line `TranslationServiceAsyncClient(credentials=...)` (measured: leaving the
    factory in place and reverting only the call site keeps everything GREEN). This pins
    that the fixed path is actually used.
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
    """The same on the transcription side (for the same reason as the test above)."""
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
    config.vad_gate = False  # requires no actual model
    await _start_and_stop(
        transcription.transcript_worker_google(
            config=config, gcp_config=GcpConfig(), in_queue=Queue()
        )
    )
    assert used.get("via_factory") is True


async def test_sender_secure_channel_authenticates_with_the_retrying_session(
    monkeypatch,
):
    """Verify the sender's ID-token path together with the **real** channel construction.

    Only grpc's `secure_channel` is substituted; the auth plugin construction
    (`AuthMetadataPlugin(credentials, request)`) runs for real. Otherwise that construction
    never passes under an assertion and a production bug such as
    `AuthMetadataPlugin(credentials, Request())` is missed.

    It also checks that the credentials really ride on the channel: it catches a mutation
    that reduces `composite_credentials` to `ssl_credentials` alone (i.e. sends no
    credentials at all).
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

    # The plugin that performs later refreshes carries the session with retries.
    plugin_request = captured["plugin_request"]
    adapter = plugin_request.session.get_adapter("https://oauth2.googleapis.com/token")
    assert adapter.max_retries is _AUTH_RETRY
    # The first refresh and the plugin share the same Request.
    assert captured["refresh_request"] is plugin_request
    # The credentials really ride on the channel (it has not fallen back to ssl alone).
    assert captured["channel_credentials"] is captured["composed"]
