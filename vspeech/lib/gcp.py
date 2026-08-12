from asyncio import get_running_loop
from collections.abc import Sequence
from typing import Any

import grpc
from google.auth import default as google_auth_default
from google.auth.compute_engine import Credentials as CeCredentials
from google.auth.compute_engine import IDTokenCredentials as CeIdTokenCredentials
from google.auth.credentials import Credentials as BaseCredentials
from google.auth.credentials import with_scopes_if_required
from google.auth.exceptions import TransportError
from google.auth.transport.grpc import AuthMetadataPlugin
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.service_account import IDTokenCredentials
from grpc import aio
from requests import Session
from requests.adapters import HTTPAdapter
from requests.adapters import Retry

from vspeech.config import GcpConfig
from vspeech.config import ServiceAccountInfo

type GcpIDTokenCredentials = IDTokenCredentials | CeIdTokenCredentials

# Retry policy for token refresh (oauth2.googleapis.com).
#
# A refresh only happens roughly once per token lifetime (about an hour), so by then
# the TLS connection left in the pool has long since been closed by Google. urllib3
# calls `is_connection_dropped()` before handing a connection out, but that only catches
# connections that were already dead before the loan; what we observed in production is
# the window where the RST arrives *after* the POST was written (the traceback died in
# `getresponse()` -> `recv_into`). It dies between the check and the write, so a
# liveness check cannot close that window -- only a retry can.
#
# Putting POST in allowed_methods is the crux: urllib3's default
# (DEFAULT_ALLOWED_METHODS) excludes the non-idempotent POST from retries, and a token
# refresh is exactly a POST, so with the default nothing would be retried at all. A
# duplicate send here merely means "receive one more token" and is harmless, whereas
# without the retry this window can never be closed. GET is included too (the metadata
# server path). urllib3 also accepts allowed_methods=False meaning "all methods", but
# the type stubs do not have it under Collection[str] | None so ty rejects it, and being
# explicit also reads as "POST was included on purpose".
_AUTH_RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    # Retrying on HTTP status is google.auth's own job (`_client`'s ExponentialBackoff
    # interprets 429/503 and friends and retries, and turns invalid_grant into the
    # proper exception), so here we only watch the connection layer.
    status=0,
    # status=0 alone is not enough. When respect_retry_after_header (default True) is on
    # and a Retry-After header is present, urllib3's `Retry.is_retry()` decides to retry
    # without consulting status_forcelist at all. Combined with status=0 that goes
    # 0 -> -1 and is immediately exhausted, so a 429/503 mutates into "a RetryError that
    # was never retried and lost the response body" (measured: a 503 with Retry-After
    # became a RetryError at attempts=1). Far from leaving it to google.auth, that steals
    # the retry google.auth already had. Turn the header respect off as well and hand
    # status handling back to google.auth entirely.
    respect_retry_after_header=False,
    allowed_methods=frozenset({"GET", "POST"}),
    backoff_factor=0.25,
)

# Upper bound in seconds for a single token refresh.
#
# google.auth passes no timeout to the POST against the token endpoint, so
# `Request.__call__`'s default (_DEFAULT_TIMEOUT = 120 seconds) applies as-is. Adding
# retries multiplies the worst case by the number of attempts: against an endpoint that
# never answers (a blackhole) that is 4 x 120 + backoff = about 481 seconds, and since
# gRPC runs the auth plugin on a separate thread per call and that thread outlives the
# RPC deadline, stuck threads pile up.
#
# Cutting a single attempt at 20 seconds bounds the worst case at 4 x 20 + 1.5 = about
# 81.5 seconds, which is shorter than before the fix (120 seconds) even with the retries
# added. A healthy response from Google's token endpoint takes under a second, so 20
# seconds is plenty of slack.
_AUTH_REQUEST_TIMEOUT_SEC = 20.0


def _cap(value: float | None) -> float:
    return (
        _AUTH_REQUEST_TIMEOUT_SEC
        if value is None
        else min(value, _AUTH_REQUEST_TIMEOUT_SEC)
    )


class _BoundedTimeoutAdapter(HTTPAdapter):
    """Cap whatever timeout the caller passed at `_AUTH_REQUEST_TIMEOUT_SEC`.

    google.auth offers no way to pass a timeout in (the refresh call is entirely
    contained in `_client._token_endpoint_request`), so the cap is applied on the
    session side. requests hands the timeout over either as a float or as a
    (connect, read) tuple, so both shapes are handled.
    """

    def send(self, request, stream=False, timeout=None, *args, **kwargs):
        if isinstance(timeout, tuple):
            connect, read = timeout
            timeout = (_cap(connect), _cap(read))
        else:
            timeout = _cap(timeout)
        return super().send(request, stream=stream, timeout=timeout, *args, **kwargs)


def build_auth_session() -> Session:
    """The `requests.Session` used for the token endpoint.

    `google.auth.transport.requests.Request()` builds a bare `requests.Session()` (no
    retries) unless a session is passed in. That is where the window of grabbing a dead
    pooled connection lives, so we build a session with retries attached and pass it in
    ourselves.
    """
    session = Session()
    adapter = _BoundedTimeoutAdapter(max_retries=_AUTH_RETRY)
    # http:// is a production path too, not a test-only one: compute engine credentials
    # hit the metadata server (http://metadata.google.internal) through this session.
    #
    # Beyond that, **mounting the very same adapter on both** is itself a requirement:
    # mounting separate ones allows a state where only the http:// side the tests
    # exercise has retries while the production https:// side is bare -- and the tests
    # stay green.
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# The options a GAPIC transport passes when it builds the channel itself.
#
# Passing our own channel skips that branch entirely and the receive limit reverts to
# gRPC's default 4 MiB (measured: a 5 MiB response came back RESOURCE_EXHAUSTED). Both
# the Translate and Speech transports were confirmed to pass these two, but **when
# adding a new client, check that its transport really passes the same ones** -- if it
# passes something else, reusing this constant will silently drop data.
GAPIC_DEFAULT_CHANNEL_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)


def create_auth_metadata_plugin(
    credentials: BaseCredentials, host: str, scopes: Sequence[str]
) -> AuthMetadataPlugin:
    """Build gRPC's auth metadata plugin so that it refreshes through a session with
    retries.

    This is what `google.api_core.grpc_helpers._create_composite_credentials` does by
    default, except that it constructs `Request()` with no arguments (there is no hook
    to inject a session), so we rebuild it by hand purely to replace that one piece.

    `with_scopes_if_required` and `default_host` follow api_core: dropping the former
    makes a service account fetch a token with no scopes and be rejected, and dropping
    the latter disables the service account's self-signed JWT path.
    """
    scoped = with_scopes_if_required(credentials, scopes=None, default_scopes=scopes)
    request = Request(session=build_auth_session())
    return AuthMetadataPlugin(scoped, request, default_host=host)


def create_auth_channel(
    credentials: BaseCredentials,
    host: str,
    scopes: Sequence[str],
    options: Sequence[tuple[str, Any]] = (),
) -> aio.Channel:
    """A grpc.aio channel carrying the credentials from `create_auth_metadata_plugin`.

    The caller (whoever uses the GAPIC transport) must always pass `options`. The
    options the transport passes when it builds the channel itself stop applying the
    moment we hand it a channel -- the receive limit reverts to the default 4 MiB.
    """
    # A channel binds to the loop running at construction time (`get_working_loop()` in
    # `grpc/aio/_channel.py`). With no loop running this does not raise: a new,
    # not-running loop is silently bound and every later RPC hangs forever. Today the
    # only callers are async functions, but preflight is synchronous, so
    # whoever adds a GCP liveness check there could quietly step into this trap. Make it
    # speak up.
    get_running_loop()
    call_credentials = grpc.metadata_call_credentials(
        create_auth_metadata_plugin(credentials, host=host, scopes=scopes)
    )
    channel_credentials = grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(), call_credentials
    )
    return aio.secure_channel(f"{host}:443", channel_credentials, options=list(options))


def unescape_private_key(service_account_info: ServiceAccountInfo):
    decoded = {k: v.get_secret_value() for k, v in service_account_info.items()}
    if "private_key" in service_account_info:
        return {
            **decoded,
            "private_key": decoded["private_key"].replace("\\n", "\n"),
        }
    return decoded


def get_credentials(config: GcpConfig) -> tuple[Credentials | CeCredentials, str]:
    """Never pass `scopes=` when loading a service account (ADR-0048).

    A service account with no scopes builds its token locally as an audience-based
    self-signed JWT -- it never touches the network. The moment `scopes=` is passed it
    leaves that branch and starts a round trip to oauth2.googleapis.com every hour. In
    other words, the very path ADR-0048 exists to avoid comes back silently just from
    adding a setting (see `_use_self_signed_jwt` in `service_account.py`).
    """
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        cred = Credentials.from_service_account_file(file_path)
        return cred, cred.project_id
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        cred = Credentials.from_service_account_info(decoded)
        return cred, cred.project_id
    elif config.use_ce_credentials:
        cred = CeCredentials()
        return cred, ""
    else:
        cred, project_id = google_auth_default()
        return cred, project_id or ""


def get_id_token_credentials(
    config: GcpConfig,
) -> GcpIDTokenCredentials | None:
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        return IDTokenCredentials.from_service_account_file(
            filename=file_path, target_audience=""
        )
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        return IDTokenCredentials.from_service_account_info(
            info=decoded, target_audience=""
        )
    elif config.use_ce_credentials:
        # Constructing CeIdTokenCredentials synchronously probes the GCE
        # metadata server (metadata.google.internal). On non-GCE hosts that
        # blocks for several seconds while DNS/connection retries exhaust
        # before raising TransportError, stalling sender worker startup. Only
        # pay that cost when the user has explicitly opted into CE credentials.
        try:
            # This Request sits on a path that makes three HTTP calls per refresh:
            # `iam.Signer` holds on to it (`iam.py`) and, on every refresh, issues a GET
            # to the metadata server and a signBlob POST to
            # iamcredentials.googleapis.com **before** the token POST that follows. With
            # a bare Request() those two would have no retries and the same stale-pool
            # window would remain (ADR-0048).
            return CeIdTokenCredentials(
                request=Request(session=build_auth_session()), target_audience=""
            )
        except TransportError:
            return None
    else:
        return None
