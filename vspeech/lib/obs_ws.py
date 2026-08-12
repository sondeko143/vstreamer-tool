"""obs-websocket 5.x client (ADR-0043).

All we need is the Hello(0)/Identify(1)/Identified(2) handshake and the
Request(6)/RequestResponse(7) round trip; event subscriptions, batches and msgpack are
unused. Callers only ever touch it through ObsTransport, so that the blast radius of a
websockets API change stays inside this file.
"""

import base64
import hashlib
import json
import reprlib
from asyncio import wait_for
from typing import Any
from typing import Protocol
from uuid import uuid4

from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

RPC_VERSION = 1

# The value put in Identify's eventSubscriptions: EventSubscription::None (ADR-0047).
#
# This module uses no events at all (see the docstring at the top), but that only holds
# once OBS is told we are not subscribing: obs-websocket treats an omitted key as
# EventSubscription::All (from the protocol document's Identify:
# `"eventSubscriptions": number(optional) = (EventSubscription::All)`). Stay silent and
# OBS keeps pushing op 5 events, while we only read while request() is in recv() waiting
# for a response (the rest of the time the subtitle worker is parked on its own
# in_queue), so unread frames pile up without bound.
#
# What they pile up into is the problem, and the symptom looks nothing like the cause:
# past 16 frames (websockets' default max_queue) the Assembler calls
# transport.pause_reading() and parsing of *every* frame stops -- Pongs included. Then
# keepalive() never receives a pong within ping_timeout and closes the connection itself
# with 1011 "keepalive ping timeout". All that is left is a log line saying "the ping
# timed out" against a localhost peer, which points nowhere near the cause.
EVENT_SUBSCRIPTION_NONE = 0

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7

STATUS_RESOURCE_NOT_FOUND = 600

# A bare repr() is dangerous: it consumes Python call stack in proportion to a JSON
# nesting depth the peer (OBS) gets to choose, so there is a window where json.loads()
# itself survives but repr() alone dies with RecursionError (roughly depth 9000-9600 for
# object nesting, moving with whatever C stack is left; array nesting does not open this
# window because json.loads() gives out first). This module calls exactly that repr()
# while composing the body of an ObsProtocolError, so a bare repr() would leak another
# (unacceptable) exception while trying to build an exception. reprlib.Repr cuts off the
# depth with its own counter and is therefore safe.
#
# maxstring/maxother only bound the length of a single leaf, not the total output
# length. Even under maxlevel=6, maxlist=6/maxdict=4 allow "width", so up to
# 6^6 ~ 46656 leaves (each <= 200 chars) can line up within 6 levels of nesting.
# Measured: an input 4 levels deep and 6 wide (a frame of about 328 KB) composes a
# 262 KB exception string -- without touching the depth where json.loads() survives (the
# RecursionError window) or the length of any single leaf. In other words
# maxstring/maxother do not have the side effect of "keeping a 1 MiB peer frame from
# becoming a 1 MiB exception string (and hence log line)". Bounding the total output is
# _bounded_repr()'s job, so every exception message in this module must go through
# _bounded_repr() rather than calling _SAFE_REPR.repr() directly.
_SAFE_REPR = reprlib.Repr(maxlevel=6, maxstring=200, maxother=200)

# Upper bound on the string _bounded_repr() returns. The longest prefix among its nine
# call sites (the one containing request_type) is about 60 characters, so even adding
# 300 + len("…(truncated)") keeps the whole exception message comfortably under 500
# characters.
_BOUNDED_REPR_MAX_CHARS = 300


def _bounded_repr(x: Any) -> str:
    """Return `_SAFE_REPR.repr(x)` further truncated to a fixed total character count,
    applied to the rendered string itself.

    `_SAFE_REPR` alone bounds only the depth (`maxlevel`) and the length of a single leaf
    (`maxstring`/`maxother`), so once the width (`maxlist`/`maxdict`) lines leaves up the
    total easily reaches hundreds of KB regardless of depth (see the module comment
    above). Truncating the final rendered string here guarantees a total-size bound that
    depends on neither width nor depth.
    """
    rendered = _SAFE_REPR.repr(x)
    if len(rendered) <= _BOUNDED_REPR_MAX_CHARS:
        return rendered
    return rendered[:_BOUNDED_REPR_MAX_CHARS] + "…(truncated)"


class ObsProtocolError(Exception):
    """The exchange with obs-websocket took an unexpected shape."""


class ObsIdentifyError(ObsProtocolError):
    """Identify did not succeed (auth failure, an op other than Hello/Identified, ...).

    An RPC version mismatch is deliberately not checked here: obs-websocket closes the
    connection itself when it detects one, so by the time control reaches this function
    it cannot happen and a check would be dead code.

    This kind of failure does not get better on retry, so callers treat it fail-loud
    (ADR-0042).
    """


class ObsRequestError(ObsProtocolError):
    def __init__(self, request_type: str, code: int, comment: str):
        self.request_type = request_type
        self.code = code
        self.comment = comment
        # comment is a free-form string from OBS with no agreed length limit -- the last
        # unbounded peer->message path: the caller (the subtitle worker) logs it on every
        # pass of its retry loop, so a malicious (or merely verbose) peer can make it
        # emit an enormous log line per retry. The attribute (self.comment) stays
        # untouched so callers can read the raw value; only the exception message is
        # truncated. To keep a normal-length comment readable, it is sliced rather than
        # run through _SAFE_REPR's repr() (which adds quotes and hurts readability).
        #
        # comment is annotated str, but this is a public class and callers outside this
        # module (including ones that compose it themselves) can construct it directly
        # with any value. Both peer-driven construction sites check
        # isinstance(comment, str) before calling, so a non-str never arrives here, yet
        # calling `len(comment)` unchecked would make e.g.
        # `ObsRequestError("X", 1, None)` die with a bare TypeError. Branch on isinstance
        # so the constructor stays total and never leaks an exception, non-str included.
        if isinstance(comment, str):
            bounded_comment = (
                comment if len(comment) <= 200 else comment[:200] + "…(truncated)"
            )
        else:
            bounded_comment = _bounded_repr(comment)
        super().__init__(
            f"{request_type} failed: code={_bounded_repr(code)} {bounded_comment}"
        )


class ObsResourceNotFoundError(ObsRequestError):
    """The named input (or similar) does not exist in OBS (code 600)."""


class ObsTransport(Protocol):
    """The minimal surface that websockets' ClientConnection satisfies."""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


def build_auth_string(password: str, salt: str, challenge: str) -> str:
    """Build the obs-websocket 5.x authentication string.

    Exactly the steps from the spec:
      1. sha256 over password + salt, base64-encoded -> the base64 secret
      2. sha256 over secret + challenge, base64-encoded
    """
    secret = base64.b64encode(
        hashlib.sha256((password + salt).encode("utf-8")).digest()
    )
    return base64.b64encode(
        hashlib.sha256(secret + challenge.encode("utf-8")).digest()
    ).decode("utf-8")


# The dedicated close-code band obs-websocket uses to declare that it rejected the
# handshake itself (the 5.x protocol document's WebSocketCloseCode assigns every
# rejection reason in here; a wrong password = 4009 is the only case observed in
# practice, but an RPC version mismatch or a malformed Identify use the same band). A
# close in this band is the only signal that can state in the type system "reconnecting
# will not help", and identify() converts only these into ObsIdentifyError.
#
# Filtering by the band is the crux; `ConnectionClosed` must not be treated as a
# rejection wholesale: quitting OBS normally closes with 1001 (going away) (measured).
# Treating that as a rejection would fire the fail-loud path every time the user closes
# OBS and kill the audio pipeline over a subtitle concern -- exactly the reason ADR-0042
# exists. This granularity could only be settled by measuring both 4009 and 1001 on real
# hardware.
_HANDSHAKE_REJECTION_CLOSE_CODES = range(4000, 5000)


def _handshake_rejection(e: ConnectionClosed) -> Close | None:
    """Return the close frame (code/reason) if `e` is a handshake rejection issued by
    obs-websocket itself, else None.

    `e.rcvd` is the close frame actually received from the peer (OBS). `e.sent` (the
    close we sent) is not used for the decision -- only the peer declares a rejection
    reason. When the connection dropped without any close frame at all (a raw transport
    disconnect, `e.rcvd is None`; e.g. the 1006 equivalent, or the process dying
    instantly so only TCP fell over mid-handshake) there is nothing to judge by, so it is
    not treated as a rejection -- it may just be "not connected yet" and get better on
    retry.
    """
    rcvd = e.rcvd
    if rcvd is None:
        return None
    if rcvd.code not in _HANDSHAKE_REJECTION_CLOSE_CODES:
        return None
    return rcvd


class ObsWsClient:
    """A thin obs-websocket 5.x client.

    Supports only the Hello(0)/Identify(1)/Identified(2) handshake and the
    Request(6)/RequestResponse(7) round trip. Event subscriptions, batch requests and the
    msgpack serializer are out of scope (ADR-0043). Every unexpected response is raised
    as an exception of this module (`ObsProtocolError` and its subclasses) so callers can
    treat them separately from `OSError` and `websockets` exceptions (ADR-0042).
    """

    def __init__(self, transport: ObsTransport, timeout: float = 5.0):
        self._transport = transport
        self._timeout = timeout

    async def _send(self, op: int, d: dict[str, Any]) -> None:
        await self._transport.send(json.dumps({"op": op, "d": d}))

    async def _recv(self) -> dict[str, Any]:
        try:
            raw = await wait_for(self._transport.recv(), timeout=self._timeout)
        except TimeoutError as e:
            raise ObsProtocolError(
                f"OBS からの応答が {self._timeout} 秒以内に来なかった"
            ) from e
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ObsProtocolError(f"OBS から UTF-8 でないフレーム: {e}") from e
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError) as e:
            # json.loads breaks in two ways: malformed JSON syntax (JSONDecodeError, a
            # subclass of ValueError) and nesting that is too deep (RecursionError, a
            # subclass of RuntimeError that ValueError does not catch). The latter can be
            # built from a few tens of KB of ASCII such as "[" * N + "]" * N, which is
            # transport-valid input well inside websockets' default max_size (1 MiB), so
            # catch it here too.
            raise ObsProtocolError(f"OBS から不正な JSON: {e}") from e
        if not isinstance(message, dict) or "op" not in message:
            raise ObsProtocolError(
                f"OBS から不正なメッセージ: {_bounded_repr(message)}"
            )
        if not isinstance(message.get("d"), dict):
            raise ObsProtocolError(
                f"OBS からのメッセージに 'd' が無い: {_bounded_repr(message)}"
            )
        return message

    async def identify(self, password: str) -> None:
        """Receive the Hello that follows connection, authenticate if required, send
        Identify and wait for Identified.

        Intended to be called exactly once per connection. obs-websocket reports neither
        a malformed Hello/Identified nor a rejection of the handshake itself (auth
        failure, RPC version mismatch, malformed Identify, ...) as an error message --
        instead it closes the WebSocket with a close code in 4000-4999 (private use)
        (measured on OBS 32.1.2 / obs-websocket 5.7.3: a wrong password gives code 4009
        "Authentication failed."). This function raises every failure it can detect as
        `ObsIdentifyError` (a subclass of `ObsProtocolError`) so the caller (ADR-0042) can
        tell "will not get better on retry" apart by type.

        Conversely, a disconnect with no close code (`ConnectionClosed.rcvd is None`, e.g.
        the connection dropping at the raw TCP level mid-handshake) or a close outside the
        4000-4999 band (a transport-level abnormal close such as 1006) is not a handshake
        rejection and may simply mean OBS is not up yet -- it can get better on retry, so
        it is not converted to `ObsIdentifyError` and propagates to the caller as
        `websockets`' `ConnectionClosed` (a subclass of `WebSocketException`). The caller
        treats that fail-open (reconnect with backoff, ADR-0042).
        """
        try:
            message = await self._recv()
            if message["op"] != OP_HELLO:
                # message['op'] is a raw peer value that _recv() only guarantees to
                # "exist as a key". An f-string's {x} still reaches dict.__repr__ through
                # format(), so it hits both the recursion hazard and the unbounded-length
                # hazard of repr() even without !r. _bounded_repr() seals both (depth via
                # reprlib's maxlevel, total width via _bounded_repr()'s own truncation).
                raise ObsIdentifyError(
                    f"Hello を期待したが op={_bounded_repr(message['op'])} が来た"
                )
            hello_data = message["d"]
            d: dict[str, Any] = {
                "rpcVersion": RPC_VERSION,
                # Omitting this means All, not "no subscription". See the comment on
                # EVENT_SUBSCRIPTION_NONE -- staying silent piles up events nobody reads
                # and eventually surfaces as a keepalive ping timeout.
                "eventSubscriptions": EVENT_SUBSCRIPTION_NONE,
            }
            # Distinguish "no authentication key" from "present but falsy":
            # obs-websocket includes this key only when authentication is required.
            # Testing truthiness (`if auth:`) would misread `{}` / `[]` / `0` / `false` /
            # `""` as "no auth needed" and send an unauthenticated Identify; if the peer
            # really does require auth it closes with 4008 and the caller retries forever
            # (a broken handshake does not get better on retry). The question to ask is
            # whether the key is present, not whether its value is truthy.
            if "authentication" in hello_data:
                auth = hello_data["authentication"]
                if not password:
                    raise ObsIdentifyError(
                        "OBS が認証を要求していますが subtitle.obs.password が空です"
                    )
                if not isinstance(auth, dict):
                    raise ObsIdentifyError(
                        f"OBS の authentication が不正な形: {_bounded_repr(auth)}"
                    )
                salt = auth.get("salt")
                challenge = auth.get("challenge")
                if not isinstance(salt, str) or not isinstance(challenge, str):
                    raise ObsIdentifyError(
                        "OBS の authentication に salt/challenge が無い:"
                        f" {_bounded_repr(auth)}"
                    )
                try:
                    d["authentication"] = build_auth_string(password, salt, challenge)
                except UnicodeError as e:
                    # isinstance(salt, str) / isinstance(challenge, str) do not guarantee
                    # the string is UTF-8 encodable (e.g. json.loads('"\\ud800"') returns
                    # a str containing an unpaired surrogate). build_auth_string() itself
                    # stays a pure function outside this concern (it has its own unit
                    # tests), so wrap it here and fail loud as a failure of identify
                    # (ADR-0042).
                    raise ObsIdentifyError(
                        f"OBS の authentication の salt/challenge が UTF-8 として不正: {e}"
                    ) from e
            await self._send(OP_IDENTIFY, d)
            message = await self._recv()
            if message["op"] != OP_IDENTIFIED:
                # The same hazard as the Hello guard above.
                raise ObsIdentifyError(
                    f"Identified を期待したが op={_bounded_repr(message['op'])} が来た"
                )
        except ConnectionClosed as e:
            # obs-websocket sends no error message; it declares the rejection through the
            # close code (see this method's docstring).
            rejection = _handshake_rejection(e)
            if rejection is None:
                # Not a handshake rejection, just a (retryable) disconnect. Do not convert
                # it to ObsIdentifyError; hand it to the caller's fail-open path as
                # ConnectionClosed.
                raise
            raise ObsIdentifyError(
                f"OBS がハンドシェイクを拒否した: {rejection}"
            ) from e

    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a Request to obs-websocket, wait for the matching RequestResponse and
        return its `responseData`.

        **Not safe for concurrent calls.** Two calls running at once both consume the
        same `_recv()` stream and can pick up each other's responses. The design assumes
        the caller (the subtitle worker) issues one request at a time; this is a
        documented constraint, not a bug to fix.
        """
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
            # No aggregate deadline (only the per-recv self._timeout). A peer that keeps
            # sending unrelated messages faster than the timeout could stall this loop,
            # but the peer is a local OBS and the risk was judged low, so it was left
            # out. Add an aggregate deadline here if a deployment ever reproduces it.
            message = await self._recv()
            # Discard events (op 5) and responses belonging to other requests.
            if message["op"] != OP_REQUEST_RESPONSE:
                continue
            d = message["d"]
            if d.get("requestId") != request_id:
                continue
            status = d.get("requestStatus")
            if not isinstance(status, dict):
                raise ObsProtocolError(
                    f"{request_type} の応答に requestStatus が無い: {_bounded_repr(d)}"
                )
            if not status.get("result"):
                code = status.get("code", 0)
                comment = status.get("comment", "")
                # Type checks that keep ObsRequestError.code/comment's annotations
                # (int/str) honest. Without them two real harms follow:
                # (1) `code == STATUS_RESOURCE_NOT_FOUND` only matches the int 600, so a
                #     peer sending "600" (str) quietly demotes ObsResourceNotFoundError
                #     to the generic ObsRequestError and the caller's dedicated
                #     "resource missing" fail-loud path never fires.
                # (2) Unchecked, e.code / e.comment can be str/list/dict/None, and caller
                #     code written to the annotations (such as `e.comment.lower()`) dies
                #     with a bare AttributeError.
                # This is the case where requestStatus itself is broken, so fail loud
                # with ObsProtocolError rather than ObsRequestError (fabricating
                # `code`/`comment` to build an ObsRequestError would only defer the same
                # lie by one step).
                # isinstance(True, int) is True (bool is a subclass of int), so a bare
                # isinstance(code, int) would let a peer send JSON true/false as code and
                # put a bool -- which can never equal 600 -- into ObsRequestError.code.
                # That is exactly the "the guard proves less than it appears to" shape
                # this module keeps running into, so exclude bool explicitly.
                if (
                    not isinstance(code, int)
                    or isinstance(code, bool)
                    or not isinstance(comment, str)
                ):
                    raise ObsProtocolError(
                        f"{request_type} の応答の requestStatus.code/comment が"
                        f" 不正な形: {_bounded_repr(status)}"
                    )
                if code == STATUS_RESOURCE_NOT_FOUND:
                    raise ObsResourceNotFoundError(request_type, code, comment)
                raise ObsRequestError(request_type, code, comment)
            response_data = d.get("responseData")
            if response_data is None:
                return {}
            if not isinstance(response_data, dict):
                # Do not make this asymmetric with the isinstance checks on
                # requestStatus: leaving responseData unchecked defers the same bug shape
                # by one frame, into an access like `result["inputSettings"]` in the
                # caller (the subtitle worker), where it becomes a bare TypeError.
                raise ObsProtocolError(
                    f"{request_type} の応答の responseData が不正な形:"
                    f" {_bounded_repr(response_data)}"
                )
            return response_data
