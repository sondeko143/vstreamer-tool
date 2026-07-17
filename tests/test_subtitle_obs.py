import asyncio
from asyncio import Queue
from uuid import uuid4

import pytest
from websockets.exceptions import ConnectionClosedError

import vspeech.worker.subtitle_obs as subtitle_obs_mod
from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import SubtitleWorkerType
from vspeech.exceptions import WorkerShutdown
from vspeech.exceptions import WorkerStartupError
from vspeech.lib.obs_text_settings import hex_color_to_obs_int
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsRequestError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.subtitle_obs import make_panels
from vspeech.worker.subtitle_obs import push_text
from vspeech.worker.subtitle_obs import validate_sources


class FakeObsClient:
    """ObsWsClient の狭い口だけを真似る。ネットワークも OBS も無し。"""

    def __init__(self, missing: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.missing = missing or set()

    async def request(self, request_type: str, request_data=None) -> dict:
        data = request_data or {}
        self.calls.append((request_type, data))
        name = data.get("inputName")
        if name in self.missing:
            raise ObsResourceNotFoundError(request_type, 600, "not found")
        if request_type == "GetInputSettings":
            return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
        return {}

    def settings_for(self, source: str) -> list[dict]:
        return [
            d["inputSettings"]
            for t, d in self.calls
            if t == "SetInputSettings" and d.get("inputName") == source
        ]


def make_config() -> Config:
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = "vspeech-translated"
    return config


def make_message(text: str, position=None) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(
            EventType.subtitle, params=Params(position=position)
        ),
        following_events=[],
        text=text,
        sound=SoundInput.invalid(),
        file_path="",
        filters=[],
    )


class SucceedingClient(FakeObsClient):
    """`FakeObsClient` を `ObsWsClient(ws)` の代わりに差し込めるようにする
    最小拡張。`identify()` を追加するだけで、`subtitle_obs_worker` レベルの
    結合テストで `ObsWsClient` の代わりに使える (ネットワークも OBS も無し)。
    """

    def __init__(self, _ws=None, missing: set[str] | None = None):
        super().__init__(missing=missing)

    async def identify(self, password: str) -> None:
        return None


class AuthRejectedClient:
    """`identify()` always raises `ObsIdentifyError` -- a password typo, the
    kind of failure ADR-0042 says must fail loud (retrying it forever would
    never succeed). fix pass 4, fail-loud finding 1 (Critical).
    """

    def __init__(self, _ws):
        pass

    async def identify(self, password: str) -> None:
        raise ObsIdentifyError("simulated auth rejection")

    async def request(self, request_type: str, request_data=None) -> dict:
        return {}


def make_missing_source_client(missing: set[str]):
    """`ObsWsClient` replacement whose `identify()` succeeds but whose
    `GetInputSettings` (via `validate_sources`) rejects the given source
    names -- a source-name typo, the second fail-loud case (ADR-0042).
    Thin wrapper over `SucceedingClient`/`FakeObsClient`'s existing
    `missing` support; needed as a factory (not a plain class) because
    `subtitle_obs_worker` constructs `ObsWsClient(ws)` positionally, with no
    way to also pass `missing` through that single-argument call site.
    """

    class Client(SucceedingClient):
        def __init__(self, _ws=None):
            super().__init__(_ws, missing=missing)

    return Client


class RetryableIdentifyTimeoutClient:
    """`identify()` always raises a *bare* `ObsProtocolError` (not the
    `ObsIdentifyError`/`ObsResourceNotFoundError` subclasses) -- an identify
    round-trip that timed out or got a malformed response, the kind of
    failure ADR-0042 says must stay fail-open (it is retryable; the next
    attempt might just work). fix pass 4, fail-loud finding 1's third case:
    proves the inner catch must stay narrow (`ObsIdentifyError`,
    `ObsResourceNotFoundError`) and not widen to `ObsProtocolError`.
    """

    def __init__(self, _ws):
        pass

    async def identify(self, password: str) -> None:
        raise subtitle_obs_mod.ObsProtocolError("identify timed out")

    async def request(self, request_type: str, request_data=None) -> dict:
        return {}


class AlwaysFailingSetClient:
    """identify/GetInputSettings は毎回成功するが、SetInputSettings は毎回
    失敗する -- typo ではなく持続的な OBS 側の拒否を模す。fix pass 1,
    finding 2 (Important, measured) の再現専用: 「識別できた直後に毎回
    壊れる」失敗が、backoff/warn-once を無効化しないことを確かめる。
    """

    def __init__(self, _ws):
        pass

    async def identify(self, password: str) -> None:
        return None

    async def request(self, request_type: str, request_data=None) -> dict:
        if request_type == "GetInputSettings":
            return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
        if request_type == "SetInputSettings":
            raise ObsRequestError("SetInputSettings", 500, "boom")
        return {}


def make_tick_hoist_client(pushed: list[tuple[int, str, str]], crash_text: str):
    """セッションごとに `pushed` へ (session_index, source, text) を記録する
    `ObsWsClient` 代替のファクトリ。セッション 0 で `crash_text` そのものを
    push しようとした瞬間に切断を起こす -- 「字幕が表示された直後に OBS が
    落ちた」を模す。fix pass 1, finding 3 (Important, measured) 専用。
    """
    session_counter = {"n": -1}

    class Client:
        def __init__(self, _ws):
            session_counter["n"] += 1
            self.session = session_counter["n"]

        async def identify(self, password: str) -> None:
            return None

        async def request(self, request_type: str, request_data=None) -> dict:
            data = request_data or {}
            if request_type == "GetInputSettings":
                return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
            if request_type == "SetInputSettings":
                settings = data.get("inputSettings", {})
                if "text" in settings:
                    text = settings["text"]
                    pushed.append((self.session, data["inputName"], text))
                    if self.session == 0 and text == crash_text:
                        raise ObsRequestError("SetInputSettings", 500, "boom")
                return {}
            return {}

    return Client


def make_mid_session_disconnect_client(after_n_text_pushes: int):
    """Raises `ConnectionClosedError` (a `WebSocketException` subclass) on
    the Nth `SetInputSettings` text push -- simulating OBS closing the
    connection *partway through a live session* (e.g. the user restarts OBS
    while streaming), not at connect/identify time like every other
    disconnect fake in this file. fix pass 4, fail-open finding 2 (Critical):
    every existing fail-open test reaches the outer catch via
    `ObsRequestError` only, so dropping `WebSocketException` from
    `subtitle_obs_worker`'s outer catch previously stayed green.
    """
    counter = {"n": 0}

    class Client(SucceedingClient):
        async def request(self, request_type: str, request_data=None) -> dict:
            data = request_data or {}
            if request_type == "SetInputSettings" and "text" in data.get(
                "inputSettings", {}
            ):
                counter["n"] += 1
                if counter["n"] == after_n_text_pushes:
                    raise ConnectionClosedError(None, None)
            return await super().request(request_type, request_data)

    return Client


def make_recording_client(pushed: list[tuple[str, str]]):
    """`pushed` へ (source, text) を記録するだけの `ObsWsClient` 代替。切断も
    クラッシュもしない -- fix pass 2, finding 4 (pause gate) 専用。
    """

    class Client(SucceedingClient):
        async def request(self, request_type: str, request_data=None) -> dict:
            result = await super().request(request_type, request_data)
            if request_type == "SetInputSettings":
                settings = (request_data or {}).get("inputSettings", {})
                if "text" in settings:
                    pushed.append(((request_data or {})["inputName"], settings["text"]))
            return result

    return Client


def make_session_health_client(clock: FakeClock):
    """セッション 0 は接続直後 (elapsed=0) に即死する。セッション 1 は、死ぬ
    直前に clock を `SESSION_HEALTHY_SEC` 超まで進めてから同じように死ぬ --
    「健全に生き延びてから落ちた」を実時間を待たずに表現する。fix pass 2,
    finding 1 (Blocker, 未カバーだったリセット枝) 専用。
    """
    session_counter = {"n": -1}

    class Client:
        def __init__(self, _ws):
            session_counter["n"] += 1
            self.session = session_counter["n"]
            self._raised = False

        async def identify(self, password: str) -> None:
            return None

        async def request(self, request_type: str, request_data=None) -> dict:
            data = request_data or {}
            if request_type == "GetInputSettings":
                return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
            if request_type == "SetInputSettings" and "text" in data.get(
                "inputSettings", {}
            ):
                if not self._raised:
                    self._raised = True
                    if self.session == 1:
                        clock.advance(subtitle_obs_mod.SESSION_HEALTHY_SEC + 1.0)
                    raise ObsRequestError("SetInputSettings", 500, "boom")
                return {}
            return {}

    return Client


def make_style_recording_client(pushed: list[tuple[str, dict]]):
    """Records every `SetInputSettings` call as `(inputName, inputSettings)`
    into `pushed` -- unlike `make_recording_client`, this keeps the *whole*
    settings dict (not just `text`), so a test can inspect style fields like
    `color`. No crash/disconnect. fix pass 4, finding 4 (reload-rebind)
    only -- needs to see the pushed colour, not just whether the worker
    crashed.
    """

    class Client(SucceedingClient):
        async def request(self, request_type: str, request_data=None) -> dict:
            result = await super().request(request_type, request_data)
            if request_type == "SetInputSettings":
                data = request_data or {}
                pushed.append(
                    (data.get("inputName", ""), data.get("inputSettings", {}))
                )
            return result

    return Client


def make_ingest_before_aging_push_crash_client(pushed: list[tuple[int, str, str]]):
    """Records every *text* `SetInputSettings` push as `(session_index,
    inputName, text)`, same shape as `make_tick_hoist_client`. Session 0
    raises the instant it sees its 3rd text push to `vspeech-text`
    (`"n"`'s source) -- the 1st is the initial connect's empty push, the 2nd
    is an earlier message being ingested and pushed, and the 3rd is the
    *aging* push that clears that earlier message once it expires. fix pass
    5, item 1: pins `_run_session`'s ingest-before-either-push order by
    making the aging push (not the ingest push) the one that fails, on a
    turn where a message and an expiry land together.
    """
    session_counter = {"n": -1}

    class Client:
        def __init__(self, _ws):
            session_counter["n"] += 1
            self.session = session_counter["n"]
            self._n_text_pushes = 0

        async def identify(self, password: str) -> None:
            return None

        async def request(self, request_type: str, request_data=None) -> dict:
            data = request_data or {}
            if request_type == "GetInputSettings":
                return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
            if request_type == "SetInputSettings":
                settings = data.get("inputSettings", {})
                if "text" in settings:
                    name = data["inputName"]
                    text = settings["text"]
                    pushed.append((self.session, name, text))
                    if name == "vspeech-text":
                        self._n_text_pushes += 1
                        if self.session == 0 and self._n_text_pushes == 3:
                            raise ObsRequestError("SetInputSettings", 500, "boom")
                return {}
            return {}

    return Client


def make_style_recording_crash_on_text_client(
    pushed: list[tuple[int, str, dict]], crash_text: str
):
    """Records every `SetInputSettings` call (style *and* text alike) as
    `(session_index, inputName, inputSettings)` -- unlike
    `make_style_recording_client`, which doesn't tag a session index because
    fix pass 4's reload-rebind test only ever needed one session. Session 0
    raises the instant `crash_text` is pushed as a *text* push
    (`"text" in settings`), forcing a reconnect -- fix pass 5, item 3 needs
    to see the *style* dict a fresh session pushes at connect time, before
    any reload-triggered self-heal has a chance to run.
    """
    session_counter = {"n": -1}

    class Client:
        def __init__(self, _ws):
            session_counter["n"] += 1
            self.session = session_counter["n"]

        async def identify(self, password: str) -> None:
            return None

        async def request(self, request_type: str, request_data=None) -> dict:
            data = request_data or {}
            if request_type == "GetInputSettings":
                return {"inputKind": "text_gdiplus_v3", "inputSettings": {}}
            if request_type == "SetInputSettings":
                settings = data.get("inputSettings", {})
                pushed.append((self.session, data.get("inputName", ""), settings))
                if self.session == 0 and settings.get("text") == crash_text:
                    raise ObsRequestError("SetInputSettings", 500, "boom")
                return {}
            return {}

    return Client


class RecordingLogger:
    """`vspeech.worker.subtitle_obs.logger` の代わりに差し込む記録専用の偽
    logger。標準 logging のハンドラ/レベル設定に依存せず、warn/backoff の
    呼び出し内容を直接アサートできるようにする。
    """

    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.debugs: list[str] = []

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        return msg % args if args else msg

    def info(self, msg, *args, **kwargs):
        self.infos.append(self._fmt(msg, args))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(self._fmt(msg, args))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(self._fmt(msg, args))


class FakeClock:
    """`monotonic` の代わりに差し込める、テストが手で進める時計。"""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeConnection:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_fake_connect():
    """`websockets.asyncio.client.connect` の代わりに差し込む。実ネットワーク
    に触れず、常に (中身の無い) 接続として振る舞う `async with` 対応の
    コンテキストマネージャを返す。
    """

    def fake_connect(url: str):
        return _FakeConnection()

    return fake_connect


class _RefusingConnection:
    """`async with` entry that raises a given exception immediately, instead
    of returning a connection -- simulates OBS being unreachable at all
    (e.g. not running, wrong host) without touching the network. fix pass 4,
    fail-open finding 2.
    """

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_refusing_connect(exc: BaseException):
    """`connect()` replacement whose `async with` entry always raises `exc`."""

    def fake_connect(url: str):
        return _RefusingConnection(exc)

    return fake_connect


def make_fake_wait_for(in_queue: Queue[WorkerInput], clock: FakeClock):
    """`wait_for` replacement for the aging/expiry tests below.

    `_run_session` times its wait against `next_expiry_sec(panels)`, a real
    wall-clock timeout we can neither wait out for real (tests must not
    sleep real seconds) nor drive via `monotonic` alone (`wait_for`'s own
    deadline comes from the event loop's clock, not the `monotonic` this
    module patches for `age_panels`/`last_tick`). This plays the same role
    `fake_sleep` plays for the backoff tests above:

    - if `timeout` is `None` (nothing is currently displayed, so there is
      nothing to expire), it really awaits `in_queue.get()` -- there is no
      timeout to fake, so this is not a fake at all, just a pass-through
      that blocks for real until a message arrives (or the test cancels the
      task), exactly like the real `wait_for(..., timeout=None)`.
    - otherwise, if a message is already queued, it returns it immediately
      (as if it arrived before the timeout fired) with no clock movement;
      if not, it advances `clock` by exactly `timeout` (so `age_panels`'
      elapsed-time math sees exactly what a real expiry would have produced)
      and raises `TimeoutError`, matching `wait_for`'s real behaviour on
      expiry without any real wait.
    """

    async def fake_wait_for(coro, timeout: float | None):
        if timeout is None:
            return await coro
        coro.close()
        if not in_queue.empty():
            return in_queue.get_nowait()
        clock.advance(timeout)
        raise TimeoutError()

    return fake_wait_for


async def _settle(ticks: int = 200) -> None:
    """バックグラウンドタスクへ、実時間を待たずにイベントループの制御を
    何度か明け渡す。ここで使う fake はどれも実際に await でブロックしない
    (キューが空でタイムアウト無しに待つ場合を除く) ので、数十 tick もあれば
    十分ここまで進む。
    """
    for _ in range(ticks):
        await asyncio.sleep(0)


def test_make_panels_uses_the_same_two_panels_as_tk():
    panels = make_panels(make_config().subtitle)
    assert set(panels) == {"n", "s"}
    assert panels["n"].anchor == "s"
    assert panels["s"].anchor == "n"


async def test_validate_sources_passes_when_both_exist():
    client = FakeObsClient()
    await validate_sources(client, make_config().subtitle.obs)
    assert [d["inputName"] for _, d in client.calls] == [
        "vspeech-text",
        "vspeech-translated",
    ]


async def test_validate_sources_raises_when_a_source_is_missing():
    client = FakeObsClient(missing={"vspeech-translated"})
    with pytest.raises(ObsResourceNotFoundError):
        await validate_sources(client, make_config().subtitle.obs)


async def test_push_text_sends_the_joined_panel_text_to_its_source():
    config = make_config()
    panels = make_panels(config.subtitle)
    from vspeech.lib.subtitle_state import ingest_text

    ts = ingest_text(panels, make_message("こんにちは"))
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, ts)
    assert client.settings_for("vspeech-text") == [{"text": "こんにちは"}]


async def test_push_text_routes_the_s_panel_to_the_translated_source():
    config = make_config()
    panels = make_panels(config.subtitle)
    from vspeech.lib.subtitle_state import ingest_text

    ts = ingest_text(panels, make_message("hello", position="s"))
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, ts)
    assert client.settings_for("vspeech-translated") == [{"text": "hello"}]


async def test_push_text_sends_empty_string_when_the_panel_drained():
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, panels["n"])
    assert client.settings_for("vspeech-text") == [{"text": ""}]


async def test_push_text_sets_overlay_true_on_its_own_call():
    # fix pass 4, finding 6 (Minor): renamed from
    # `test_push_text_uses_overlay_so_it_does_not_clobber_style`. That name
    # and its `all(... for t, d in client.calls if t == "SetInputSettings")`
    # assertion *read* as covering "does not clobber style" -- i.e. every
    # `SetInputSettings` call this worker can make -- but this test only
    # ever calls `push_text`, so `client.calls` holds exactly one
    # `SetInputSettings` and `all()` ranges over a single element. Proven:
    # mutating `_push_panel_style` (the *style*-push function, a different
    # code path entirely) from `overlay: True` to `overlay: False` left this
    # test green. Narrowed the name and assertion to what this test actually
    # exercises (`push_text`'s own `overlay` flag, asserted directly rather
    # than via a misleadingly-broad `all()`);
    # `test_push_panel_style_sets_overlay_true_so_it_does_not_clobber_other_settings`
    # below covers the style-push path this test's old name implied but
    # never touched.
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, panels["n"])
    calls = [d for t, d in client.calls if t == "SetInputSettings"]
    assert len(calls) == 1
    assert calls[0]["overlay"] is True


async def test_push_panel_style_sets_overlay_true_so_it_does_not_clobber_other_settings():
    # fix pass 4, finding 6 (Minor), the other half: the mutation
    # `_push_panel_style: overlay True -> False` survived every pre-existing
    # test because none of them call `_push_panel_style` and check its
    # `overlay` field -- `test_push_panel_style_sends_a_single_panels_config_values`
    # below checks colour/valign/font fields but not `overlay`. This drives
    # `_push_panel_style` (the style-push path) directly and pins its
    # `overlay` flag, same reasoning as `push_text`'s: OBS's `SetInputSettings`
    # replaces the *whole* input's settings unless `overlay: True` asks it to
    # merge, so a style push without it would clobber whatever `push_text`
    # (or a human, via the OBS UI) had already set.
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await subtitle_obs_mod._push_panel_style(client, config.subtitle, "n", panels["n"])
    calls = [d for t, d in client.calls if t == "SetInputSettings"]
    assert len(calls) == 1
    assert calls[0]["overlay"] is True


async def test_push_panel_style_sends_a_single_panels_config_values():
    # fix pass 3, finding 2 (Minor): this test used to call the unguarded
    # public wrapper `push_styles`, which had no caller left in `vspeech/` --
    # `_push_styles_or_warn` loops `_push_panel_style` itself (fix pass 2,
    # finding 5), so `push_styles` was dead code kept alive only by this
    # test, sitting on the obvious public name for a future caller to
    # reintroduce the exact bare-`ValueError`-escape Critical this worker
    # spent two passes closing. Deleted `push_styles`; this test now pins the
    # same per-panel field mapping directly against `_push_panel_style`, the
    # guarded path's actual building block. Iterating over *both* panels in
    # one pass (and surviving a bad first panel) is covered separately by
    # `test_a_bad_first_panel_color_does_not_block_the_second_panels_good_style`
    # (via `_push_styles_or_warn`, the only remaining caller).
    config = make_config()
    config.subtitle.text.font_color = "#ff8000"
    config.subtitle.translated.font_size = 22
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await subtitle_obs_mod._push_panel_style(client, config.subtitle, "n", panels["n"])
    await subtitle_obs_mod._push_panel_style(client, config.subtitle, "s", panels["s"])
    text_settings = client.settings_for("vspeech-text")[0]
    translated_settings = client.settings_for("vspeech-translated")[0]
    # BGR, not the un-reversed hex value: hex_color_to_obs_int("#ff8000") ==
    # 0x0080FF, hardware-measured and already asserted by
    # tests/test_obs_text_settings.py (ADR-0041). The brief's draft asserted
    # 0xFF8000 here, which is the un-reversed value and would only pass for a
    # color whose hex digits happen to be a BGR/RGB palindrome.
    assert text_settings["color"] == 0x0080FF
    assert text_settings["valign"] == "bottom"
    assert translated_settings["font"]["size"] == 22
    assert translated_settings["valign"] == "top"


# --- fix pass 1: the 8 tests above only ever exercise pure helpers through
# FakeObsClient. Nothing below this line existed before fix pass 1 -- these
# are the first tests to drive _run_session / subtitle_obs_worker / backoff /
# warn-once / reload. (This comment originally also claimed "the pause gate"
# as covered here -- it wasn't; none of fix pass 1's tests touch
# context.running. Fixed in fix pass 2, finding 4: see
# test_pause_gate_holds_a_queued_message_until_context_running_is_set below,
# which is the actual pause-gate coverage.)


async def test_push_styles_or_warn_swallows_a_tk_only_color_and_warns(monkeypatch):
    # fix pass 1, finding 1 (Critical): direct, fast proof of the shared fix
    # both trigger sites (connect-time push_styles and the reload path)
    # delegate to. A Tk-valid colour name must not raise past this function.
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    config = make_config()
    # push_styles iterates panels in insertion order ("n" then "s"), so
    # breaking the *second* (translated) panel's colour proves the first
    # panel's style still reached OBS before the failure -- not just that
    # nothing crashed.
    config.subtitle.translated.font_color = "white"
    panels = make_panels(config.subtitle)
    client = FakeObsClient()

    await subtitle_obs_mod._push_styles_or_warn(
        client, config.subtitle, panels, style_warned={}
    )

    assert any("white" in w for w in fake_logger.warnings)
    # the "n" panel's colours are still valid defaults and was pushed first,
    # so its style already reached OBS before the "s" panel's push raised.
    assert client.settings_for("vspeech-text")


async def test_a_bad_first_panel_color_does_not_block_the_second_panels_good_style(
    monkeypatch,
):
    # fix pass 2, finding 5 (Minor): before this fix, `_push_styles_or_warn`
    # wrapped the *whole* `push_styles` panel loop in one try/except, so a
    # bad "n" (text) colour -- the panel iterated *first* -- aborted the loop
    # before the "s" (translated) panel, even though its colour was fine.
    # Each panel is now guarded independently: breaking the first panel must
    # not stop the second, still-valid panel from getting its update.
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    config = make_config()
    config.subtitle.text.font_color = "white"  # the *first*-iterated panel
    panels = make_panels(config.subtitle)
    client = FakeObsClient()

    await subtitle_obs_mod._push_styles_or_warn(
        client, config.subtitle, panels, style_warned={}
    )

    assert any("white" in w for w in fake_logger.warnings)
    # unfixed, this would be empty: the loop would have aborted at "n"
    # before ever reaching "s".
    assert client.settings_for("vspeech-translated")


async def test_subtitle_obs_worker_does_not_let_a_bad_color_escape_at_first_connect(
    monkeypatch,
):
    # fix pass 1, finding 1 (Critical), the *other* trigger site named in the
    # review: push_styles at the very first connect (subtitle_obs.py:217
    # pre-fix), reached whenever a reload changed the colour while OBS was
    # still down and it then comes up -- preflight never re-runs, so this is
    # the only remaining guard. A bare ValueError here used to take the
    # whole TaskGroup (and the live voice pipeline) down with it.
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", SucceedingClient)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    config = make_config()
    config.subtitle.text.font_color = "green"
    context = SharedContext(config=config)
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )
    assert any("green" in w for w in fake_logger.warnings)

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_reload_with_a_tk_only_color_warns_and_keeps_the_session_running(
    monkeypatch,
):
    # fix pass 1, finding 1 (Critical): the reload trigger site
    # (subtitle_obs.py:173 pre-fix). A config edit that lands on a *live*
    # session must degrade (warn + keep the previous style + keep running),
    # not kill vc/playback along with it.
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", SucceedingClient)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    config = make_config()  # valid colours at first connect
    context = SharedContext(config=config)
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # let the first connect finish; it blocks on the empty queue

    # a reload lands on the *live* session, only now turning the colour bad.
    context.config.subtitle.text.font_color = "white"
    worker_meta.need_reload = True
    await in_queue.put(make_message("hello"))  # unblocks the current wait_for
    await _settle()  # the next loop iteration applies the reload

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )
    assert any("white" in w for w in fake_logger.warnings)
    assert worker_meta.need_reload is False

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_backoff_and_warn_once_survive_a_recurring_post_identify_failure(
    monkeypatch,
):
    # fix pass 1, finding 2 (Important, measured). The old code reset
    # backoff/warned as soon as identify+validate_sources succeeded, so a
    # failure that recurs *after* that point (SetInputSettings rejected
    # every single session) never actually backed off, and warned on every
    # attempt instead of once. Measured pre-fix: sleeps == [0.5] * 8,
    # warnings == 8.
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 8:
            raise asyncio.CancelledError()

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", AlwaysFailingSetClient)

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerShutdown):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0, 5.0]
    reach_warnings = [w for w in fake_logger.warnings if "cannot reach" in w]
    assert len(reach_warnings) == 1


async def test_display_clock_advances_across_an_outage_so_stale_text_is_not_re_pushed(
    monkeypatch,
):
    # fix pass 1, finding 3 (Important, measured). `last_tick` used to be a
    # `_run_session` local, re-initialised on every reconnect, so the
    # elapsed outage time never reached `age_panels` and a minute-old
    # subtitle got re-pushed verbatim on recovery. Measured pre-fix with a
    # 60s outage: the reconnect's push still carried the stale text.
    pushed: list[tuple[int, str, str]] = []
    stale_text = "これは古い字幕です"
    clock = FakeClock(0.0)

    async def fake_sleep(sec: float) -> None:
        # Simulate a single 60s outage between session 0 dying and session 1
        # connecting, without any real delay.
        clock.advance(60.0)

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_tick_hoist_client(pushed, stale_text)
    )
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "monotonic", clock)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()
    await in_queue.put(make_message(stale_text))

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()

    session1_texts = {
        (source, text) for session, source, text in pushed if session == 1
    }
    assert session1_texts == {("vspeech-text", ""), ("vspeech-translated", "")}
    assert (1, "vspeech-text", stale_text) not in pushed
    assert (1, "vspeech-translated", stale_text) not in pushed

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


# --- fix pass 2: the review that closed fix pass 1 found the tests above
# genuinely proved their three findings, but flagged one blocker (the
# SESSION_HEALTHY_SEC reset branch had zero coverage) and four minors. The
# tests below are fix pass 2's additions.


async def test_a_healthy_session_dying_resets_backoff_and_warns_again(monkeypatch):
    # fix pass 2, finding 1 (Blocker, measured): the >= SESSION_HEALTHY_SEC
    # reset branch (subtitle_obs.py ~364-369) had zero coverage -- both fix
    # pass 1 failure tests above kill the session at ~0 elapsed, so that arm
    # never ran. A re-review forced the condition permanently false and
    # 475/475 still passed. Session 0 here dies instantly (elapsed=0, same
    # shape as the crash-loop test above, so it must NOT reset). Session 1 is
    # made to look like it lived past SESSION_HEALTHY_SEC before dying --
    # the fake client advances the fake clock immediately before raising, no
    # real sleep -- which must reset backoff/warned, so the second outage
    # warns again (not silenced by warn-once) and the next sleep restarts at
    # INITIAL_BACKOFF_SEC instead of continuing to climb from where session
    # 0 left off.
    clock = FakeClock(0.0)
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_session_health_client(clock)
    )
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "monotonic", clock)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerShutdown):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    # unfixed (reset branch dead/false), this would be [0.5, 1.0]: session 1's
    # death would just keep climbing session 0's backoff instead of
    # restarting from INITIAL_BACKOFF_SEC.
    assert sleeps == [
        subtitle_obs_mod.INITIAL_BACKOFF_SEC,
        subtitle_obs_mod.INITIAL_BACKOFF_SEC,
    ]
    reach_warnings = [w for w in fake_logger.warnings if "cannot reach" in w]
    # one warning for session 0's quick death, a second *distinct* one after
    # session 1's healthy-then-dead cycle -- unfixed, warn-once never gets
    # released and this would be 1 (later outages silently hidden).
    assert len(reach_warnings) == 2


async def test_push_styles_or_warn_only_warns_once_for_a_persisting_bad_value(
    monkeypatch,
):
    # fix pass 2, finding 3 (Minor, measured): a bad colour plus a flapping
    # OBS used to log this warning on every single reconnect (measured: 20
    # attempts -> 20 style warnings, against 1 correctly-gated "cannot reach"
    # warning) even though the colour never changed. `style_warned` gates it
    # per panel, the same warn-once shape as the connection `warned` flag --
    # but only while the *same* bad value persists.
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    config = make_config()
    config.subtitle.text.font_color = "white"
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    style_warned: dict[str, bool] = {}

    for _ in range(3):
        await subtitle_obs_mod._push_styles_or_warn(
            client, config.subtitle, panels, style_warned
        )

    style_warnings = [w for w in fake_logger.warnings if "invalid style" in w]
    assert len(style_warnings) == 1

    # fixed, then broken again with a *different* value -- must warn again,
    # not stay silenced forever (finding 3's explicit second requirement).
    config.subtitle.text.font_color = "#ff8000"
    await subtitle_obs_mod._push_styles_or_warn(
        client, config.subtitle, panels, style_warned
    )
    config.subtitle.text.font_color = "green"
    await subtitle_obs_mod._push_styles_or_warn(
        client, config.subtitle, panels, style_warned
    )

    style_warnings = [w for w in fake_logger.warnings if "invalid style" in w]
    assert len(style_warnings) == 2


async def test_pause_gate_holds_a_queued_message_until_context_running_is_set(
    monkeypatch,
):
    # fix pass 2, finding 4 (Minor): the "fix pass 1" section comment above
    # claimed these tests covered "the pause gate" -- none of them touch
    # context.running. This drives it directly. A message already in flight
    # when the gate closes still reaches OBS (_run_session checks
    # context.running *after* processing a message, not before -- see
    # subtitle_obs.py's `if not context.running.is_set(): await
    # context.running.wait()` at the bottom of the loop body), but a second
    # message queued while the gate is closed must wait for
    # context.running.set() before it is pushed.
    pushed: list[tuple[str, str]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", make_recording_client(pushed))

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles; worker is parked on the empty queue

    context.running.clear()
    await in_queue.put(make_message("kick"))
    await _settle()
    assert ("vspeech-text", "kick") in pushed  # already in flight, still lands

    await in_queue.put(make_message("blocked"))
    await _settle()
    # gate holds it: "blocked" must not have reached OBS in any form yet.
    # (Panel "n" accumulates history and joins entries with a delimiter, so
    # a released push would carry "blocked" combined with "kick" rather than
    # the bare string -- checking containment, not equality, is what actually
    # matches the panel's real join behaviour.)
    assert not any("blocked" in text for _, text in pushed)

    context.running.set()
    await _settle()
    assert any("blocked" in text for _, text in pushed)  # released once resumed

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


# --- fix pass 3: a review that closed fix pass 2 confirmed all prior
# findings genuinely fixed and flagged two last items: one Important
# (below), one Minor (the push_styles deletion above -- retargeted, not
# added, so it has no new test of its own).


async def test_style_warn_once_persists_across_reconnects_not_just_within_a_session(
    monkeypatch,
):
    # fix pass 3, finding 1 (Important, measured). `style_warned` is created
    # once in `subtitle_obs_worker`, *above* the `while True:` reconnect
    # loop, and threaded into every session -- `_push_styles_or_warn`'s own
    # docstring says that placement is *why* a flapping OBS doesn't
    # reproduce fix pass 2's finding 3 (20 reconnects -> 20 style warnings).
    # But nothing actually drove a reconnect loop with a persistently bad
    # colour through `subtitle_obs_worker` before this test:
    # `test_push_styles_or_warn_only_warns_once_for_a_persisting_bad_value`
    # only proves the *mechanism* by calling `_push_styles_or_warn` directly
    # with a hand-shared dict -- it never touches the worker's own wiring, so
    # it would still pass even if `subtitle_obs_worker` created a fresh dict
    # per session. This test drives the real reconnect loop instead.
    #
    # Reuses `AlwaysFailingSetClient` (originally built for fix pass 1,
    # finding 2's backoff-reset repro) for a second purpose here: every
    # `SetInputSettings` it receives fails, which -- once the "n" panel's bad
    # colour has already been handled locally by `build_text_settings`
    # raising before any request is sent -- makes the *next* panel's
    # (valid-but-doomed) style push the thing that forces each reconnect.
    # That gives a clean "one style warn-once decision per session, followed
    # by a forced disconnect" shape without a bespoke fake.
    #
    # Measured with `style_warned` moved inside the loop (this test's own
    # mutation-proof, see the task report): 6 reconnects -> 6 "invalid
    # style" warnings, reproducing finding 3's exact original signature,
    # while the adjacent "cannot reach" warn-once correctly stays at 1
    # either way (sessions here die near-instantly, so they never cross
    # SESSION_HEALTHY_SEC and reset it).
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 6:
            raise asyncio.CancelledError()

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", AlwaysFailingSetClient)

    config = make_config()
    config.subtitle.text.font_color = "white"  # persistently bad, every reconnect
    context = SharedContext(config=config)
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerShutdown):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert len(sleeps) == 6  # 6 full sessions actually ran, not fewer

    style_warnings = [w for w in fake_logger.warnings if "invalid style" in w]
    assert len(style_warnings) == 1

    reach_warnings = [w for w in fake_logger.warnings if "cannot reach" in w]
    assert len(reach_warnings) == 1


# --- fix pass 4: a coverage audit (not a code review) ran 49 mutations
# against this file and found the worker's own logic already correct -- 10
# of 49 survived while all 18 pre-fix-pass-4 tests stayed green. Verdict:
# the suite pinned the *mechanisms* the three prior fix passes added (the
# DEGRADE colour path, the backoff/warn-once reset, the aging clock's
# cross-reconnect lifetime) and left the *file's reason for existing*
# unpinned -- ADR-0042's fail-loud/fail-open tiers (docs/adr/0042-subtitle-
# obs-failure-tiers.md), the aging/expiry loop, and the reload-rebind
# semantics real reloads actually use. See task-7-report.md's "Fix pass 4"
# section for the mutation -> test table and RED/GREEN proof for each test
# below.


async def test_an_auth_rejection_becomes_a_worker_startup_error(monkeypatch):
    # ADR-0042 fail-loud, finding 1 (Critical): WorkerStartupError and
    # ObsIdentifyError appear nowhere in this test file before fix pass 4 --
    # every fake's identify() returns None. Dropping ObsIdentifyError from
    # subtitle_obs_worker's inner `except (ObsIdentifyError,
    # ObsResourceNotFoundError)` lets a password typo fall through to the
    # outer fail-open catch (ObsIdentifyError IS an ObsProtocolError) and
    # retry forever instead of raising WorkerStartupError -- exactly the
    # "infinite silent retry" ADR-0042's Alternatives-rejected section names.
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", AuthRejectedClient)

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerStartupError):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert sleeps == []  # fatal on the very first attempt -- never retried


async def test_a_missing_source_becomes_a_worker_startup_error(monkeypatch):
    # ADR-0042 fail-loud, finding 1 (Critical), the other half:
    # ObsResourceNotFoundError from validate_sources must also become
    # WorkerStartupError. Also kills "delete validate_sources entirely":
    # without the upfront check, the missing source's ObsResourceNotFoundError
    # would instead surface later from a push_text/_push_panel_style call --
    # a code path only the *outer* fail-open catch covers (it's still an
    # ObsProtocolError), turning a permanent typo into an infinite retry.
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_missing_source_client({"vspeech-text"})
    )

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerStartupError):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert sleeps == []


async def test_a_retryable_identify_timeout_is_fail_open_not_fatal(monkeypatch):
    # ADR-0042 fail-loud, finding 1's third case (Critical): a bare
    # ObsProtocolError from identify() (e.g. a response timeout) must stay
    # fail-open -- it is exactly the case the module docstring (L18-21)
    # names as the reason worker_startup's blanket `except Exception` isn't
    # used here. Widening subtitle_obs_worker's inner catch from
    # `(ObsIdentifyError, ObsResourceNotFoundError)` to `ObsProtocolError`
    # would make this fatal instead -- this is the test that kills that
    # specific mutation (the two tests above don't: they'd stay green even
    # if the inner catch were widened, since ObsIdentifyError/
    # ObsResourceNotFoundError are still caught either way).
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError()

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", RetryableIdentifyTimeoutClient)

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerShutdown):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert len(sleeps) == 3  # kept retrying instead of raising WorkerStartupError
    reach_warnings = [w for w in fake_logger.warnings if "cannot reach" in w]
    assert len(reach_warnings) == 1


async def test_a_refused_connection_is_fail_open_not_fatal(monkeypatch):
    # ADR-0042 fail-open, finding 2 (Critical): "OBS is not running" is the
    # ADR's headline scenario, but every pre-fix-pass-4 test reaches
    # fail-open through ObsRequestError only -- none ever makes connect()
    # itself fail. Dropping OSError from subtitle_obs_worker's outer catch
    # would let a refused connection escape unguarded and kill the
    # TaskGroup.
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)
    monkeypatch.setattr(
        subtitle_obs_mod,
        "connect",
        make_refusing_connect(ConnectionRefusedError("OBS is not running")),
    )

    context = SharedContext(config=make_config())
    context.add_worker(event=EventType.subtitle, configs_depends_on=["subtitle"])
    in_queue: Queue[WorkerInput] = Queue()

    with pytest.raises(WorkerShutdown):
        await subtitle_obs_mod.subtitle_obs_worker(context, in_queue)

    assert sleeps == [
        subtitle_obs_mod.INITIAL_BACKOFF_SEC,
        subtitle_obs_mod.INITIAL_BACKOFF_SEC * 2,
    ]
    reach_warnings = [w for w in fake_logger.warnings if "cannot reach" in w]
    assert len(reach_warnings) == 1


async def test_a_mid_session_disconnect_is_fail_open_not_fatal(monkeypatch):
    # ADR-0042 fail-open, finding 2 (Critical): every pre-fix-pass-4 test
    # that reaches fail-open does so at connect/identify time. Dropping
    # WebSocketException from subtitle_obs_worker's outer catch would let a
    # *mid-session* ConnectionClosedError (OBS restarting while the pipeline
    # is live) escape the worker and kill the whole TaskGroup -- taking the
    # live voice pipeline down with it, the exact Critical this file exists
    # to prevent (module docstring L7-9).
    fake_logger = RecordingLogger()
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_mid_session_disconnect_client(3)
    )
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect's 2 initial (empty) text pushes land;
    # parked on the empty queue

    await in_queue.put(make_message("hello"))  # the 3rd text push -> disconnect
    await _settle()

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )
    assert any("cannot reach" in w for w in fake_logger.warnings)

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_a_displayed_subtitle_is_cleared_from_obs_once_its_display_time_expires(
    monkeypatch,
):
    # ADR-0042, finding 3 (Critical): no pre-fix-pass-4 test ever shows a
    # subtitle appearing and then disappearing -- every fake's clock only
    # ever advances inside fake_sleep (the reconnect backoff), so
    # within-session elapsed time was always 0 and _run_session's aging push
    # (subtitle_obs.py's `for ts in age_panels(...): await push_text(...)`)
    # never actually ran against a real expiry in any prior test. Kills both
    # "remove the aging push" and "timeout = next_expiry_sec(...) -> None":
    # either leaves the expired subtitle on screen in OBS forever.
    clock = FakeClock(0.0)
    pushed: list[tuple[str, str]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", make_recording_client(pushed))
    monkeypatch.setattr(subtitle_obs_mod, "monotonic", clock)

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()
    monkeypatch.setattr(
        subtitle_obs_mod, "wait_for", make_fake_wait_for(in_queue, clock)
    )

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles; parked on the empty queue

    await in_queue.put(make_message("hello"))
    await _settle()  # "hello" is pushed, then the next loop's timeout fires
    # (fake_wait_for advances the clock by min_display_sec) and the aging
    # push should clear it -- all without any real sleep.

    hello_idx = pushed.index(("vspeech-text", "hello"))
    cleared_idx = next(
        (
            i
            for i, (source, text) in enumerate(pushed)
            if source == "vspeech-text" and text == "" and i > hello_idx
        ),
        None,
    )
    assert cleared_idx is not None, (
        f"'hello' was never cleared from OBS after expiring: {pushed}"
    )

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_two_messages_within_min_display_sec_coexist_instead_of_the_second_wiping_the_first(
    monkeypatch,
):
    # ADR-0042, finding 3 (Critical): `drop last_tick[0] = now`
    # (subtitle_obs.py's _run_session, right after the aging push) stays
    # green across every pre-fix-pass-4 test. Its real effect: last_tick[0]
    # freezes at its value from connect time, so every later iteration's
    # elapsed-time computation (now - last_tick[0]) grows across the *whole*
    # session instead of since the previous iteration -- once a session has
    # been alive longer than min_display_sec (2.5s, i.e. essentially
    # always), the very next message's aging push sees a huge fake "elapsed"
    # and wipes whatever is currently on screen the instant a new message
    # arrives, even though it was displayed a moment ago. Simulates a
    # session that's been connected-but-idle for 3s (> min_display_sec)
    # before its first message ever arrives, then a second message
    # immediately behind the first with no gap -- unfixed, this destroys
    # "AAA" the instant "BBB" arrives instead of letting both coexist and
    # join (panel "n" has anchor "s", so Texts.texts joins newest-first:
    # "BBB AAA").
    clock = FakeClock(0.0)
    pushed: list[tuple[str, str]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(subtitle_obs_mod, "ObsWsClient", make_recording_client(pushed))
    monkeypatch.setattr(subtitle_obs_mod, "monotonic", clock)

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()
    monkeypatch.setattr(
        subtitle_obs_mod, "wait_for", make_fake_wait_for(in_queue, clock)
    )

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles; parked on the empty queue

    clock.advance(3.0)  # session already alive past min_display_sec
    await in_queue.put(make_message("AAA"))
    await in_queue.put(make_message("BBB"))  # queued before AAA is even
    # processed, so fake_wait_for hands it back with zero elapsed time --
    # "arrives right behind AAA".
    await _settle()

    text_pushes = [text for source, text in pushed if source == "vspeech-text"]
    # Only check the first 3 pushes: with no more messages ever arriving,
    # _settle()'s tick budget is generous enough for the loop to keep
    # spinning through subsequent *real* expiries afterward (BBB's own
    # min_display_sec eventually elapsing too) -- that continuation is
    # correct, expected behaviour, not the thing under test. unfixed
    # (last_tick[0] = now dropped), this prefix is instead
    # ["", "AAA", "BBB"] (measured): AAA gets destroyed the instant BBB
    # arrives, so BBB is pushed alone instead of joined with AAA.
    assert text_pushes[:3] == ["", "AAA", "BBB AAA"]

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_reload_rebinds_context_config_so_the_worker_actually_picks_up_the_new_value(
    monkeypatch,
):
    # ADR-0042, finding 4: a real reload (process_command, lib/command.py's
    # `context.config = new_config`) *rebinds* context.config to a brand-new
    # Config instance -- it does not mutate the old one in place. The
    # pre-fix-pass-4 reload test
    # (test_reload_with_a_tk_only_color_warns_and_keeps_the_session_running)
    # instead mutates context.config.subtitle.text.font_color directly on
    # the *same* object make_panels already built panels["n"].config from,
    # so ts.config sees the edit with zero reload logic ever running.
    # `_apply_reload: drop _refresh_panel_configs` stays green under that
    # test, because dropping the call that re-points panels["n"].config at
    # the new Config's subtitle.text changes nothing when there IS no new
    # Config object. This test rebinds context.config the way a real reload
    # does, and checks the *pushed* style value, not just "did it crash".
    fake_logger = RecordingLogger()
    pushed: list[tuple[str, dict]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_style_recording_client(pushed)
    )
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    config = make_config()  # default font_color "#ffffff" at first connect
    context = SharedContext(config=config)
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles

    # a real reload rebinds context.config to a brand-new instance (mirrors
    # lib/command.py's `context.config = new_config`), not an in-place
    # mutation of the object panels["n"].config already points at.
    new_config = config.model_copy(deep=True)
    new_config.subtitle.text.font_color = "#0080ff"
    context.config = new_config
    worker_meta.need_reload = True
    await in_queue.put(make_message("hello"))  # unblocks the current wait_for
    await _settle()  # the next loop iteration applies the reload

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )
    assert worker_meta.need_reload is False

    text_style_pushes = [
        settings
        for name, settings in pushed
        if name == "vspeech-text" and "color" in settings
    ]
    # unfixed (_refresh_panel_configs dropped), panels["n"].config is still
    # the *old* Config's subtitle.text object -- this would stay the default
    # font_color's int forever, never observing the reload.
    assert text_style_pushes[-1]["color"] == hex_color_to_obs_int("#0080ff")

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


# --- fix pass 5: an audit of the fix pass 4 reorder (moving ingest_text
# ahead of both pushes in _run_session) found the reorder itself
# mutation-invisible -- reverting it, in full or in part, left 27/27 green,
# because neither aging test ever has a message and an expiry land in the
# *same* turn. It also found two pre-existing survivors outside that diff:
# _apply_reload's own _push_all_text call, and the connect-time
# _refresh_panel_configs call. See task-7-report.md's "Fix pass 5" section
# for the RED/GREEN proof of the reorder test below and the experiments
# behind the other two.


async def test_ingest_survives_a_same_turn_aging_push_failure(monkeypatch):
    # ADR-0042 / fix pass 4, finding 5 (audit, fix pass 5): a message and an
    # expiry landing in the *same* turn, where the *aging* push (not the
    # ingest push) is the one that fails. `_run_session` ingests -- a pure
    # state update that cannot raise -- before either push specifically so a
    # push failure (which kills the session and jumps straight to the outer
    # fail-open catch) can no longer un-ingest anything already applied: the
    # next session's reconnect (_push_all_text) still carries it.
    #
    # This is the *position* of `ingest_text` relative to the aging push,
    # not an incidental detail: both a full revert to the old order (aging
    # push before ingest_text ever runs) and a partial revert that keeps the
    # `aged`/`ingested` variable shape but still moves the `ingest_text`
    # call to after the aging-push loop lose "NEWTEXT" the same way, because
    # neither ever calls ingest_text before the aging push raises. (Swapping
    # only the two *push* calls, leaving ingest_text's position alone, is an
    # equivalent mutant -- by the time either push runs, ingest_text has
    # already applied either way, so the message survives regardless of push
    # order; not tested here, per the brief.)
    #
    # "OLDTEXT" (panel "n") is made to expire in the same turn "NEWTEXT"
    # (panel "s") is ingested, by using the `context.running` gate as a real
    # suspend point: without it, `_run_session`'s turns run back-to-back
    # with no genuine `await` between them (none of the fakes here truly
    # block), so there would be no way to advance the clock and queue the
    # next message strictly *between* two turns from the test.
    clock = FakeClock(0.0)
    pushed: list[tuple[int, str, str]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod,
        "ObsWsClient",
        make_ingest_before_aging_push_crash_client(pushed),
    )
    monkeypatch.setattr(subtitle_obs_mod, "monotonic", clock)

    async def fake_sleep(sec: float) -> None:
        return None

    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)

    context = SharedContext(config=make_config())
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()
    monkeypatch.setattr(
        subtitle_obs_mod, "wait_for", make_fake_wait_for(in_queue, clock)
    )

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles; parked on the empty queue
    # (timeout=None, a real block -- nothing displayed yet)

    context.running.clear()  # forces a real suspend point after this turn
    await in_queue.put(make_message("OLDTEXT"))  # unblocks the parked wait
    await _settle()  # "OLDTEXT" ingested into "n" and pushed; the turn then
    # blocks for real on context.running.wait()

    clock.advance(100.0)  # far past OLDTEXT's display time, whatever it is
    await in_queue.put(make_message("NEWTEXT", position="s"))
    context.running.set()
    await _settle()  # this turn ages "n" (OLDTEXT expires) and ingests
    # "NEWTEXT" into "s" together; the aging push (for "n") then raises

    assert not task.done(), (
        f"worker crashed instead of failing open: "
        f"{task.exception() if task.done() else None}"
    )

    # the message must have survived in panel state even though its own
    # push never happened in session 0 -- the aging push that killed the
    # session ran first. Session 1's reconnect re-pushes every panel's
    # *current* text (_push_all_text), so if "NEWTEXT" made it into panel
    # "s"'s state, it shows up there.
    session1_pushes = {
        (source, text) for session, source, text in pushed if session == 1
    }
    assert ("vspeech-translated", "NEWTEXT") in session1_pushes

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_apply_reload_re_pushes_current_text_so_a_delimiter_change_does_not_stay_stale(
    monkeypatch,
):
    # ADR-0042 / fix pass 5, item 2 (C1, audit): does `_apply_reload`'s own
    # `_push_all_text` call matter, or does the next natural event always
    # cover it? Experiment: a reload can change *how* an already-displayed,
    # otherwise-unchanged panel renders (delimiter here; anchor is the other
    # case) without any message or expiry following it. Nothing else in the
    # loop re-renders a panel on its own -- only a new message or an expiry
    # does -- so without this call, OBS keeps showing the pre-reload
    # rendering until one of those happens, which could be a long time (or
    # never, for a panel that's gone quiet). Measured: dropping this one
    # line leaves the pre-existing 27/27 green, because none of them check
    # the panel's *rendered* text immediately after a reload that changes
    # delimiter/anchor with no message riding along. Conclusion: load-bearing
    # -- pinned here, not deleted.
    fake_logger = RecordingLogger()
    pushed: list[tuple[str, dict]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod, "ObsWsClient", make_style_recording_client(pushed)
    )
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    config = make_config()  # default delimiter " "
    context = SharedContext(config=config)
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # first connect settles; parked on the empty queue

    await in_queue.put(make_message("AAA"))
    await _settle()

    context.running.clear()  # a real suspend point strictly *between*
    # "BBB" landing and the reload's need_reload check -- otherwise the
    # reload and "BBB" would be indistinguishable from a single combined
    # turn, and this test wouldn't isolate _apply_reload's own push.
    await in_queue.put(make_message("BBB"))
    await _settle()  # "BBB" ingested and pushed; the turn then blocks for
    # real on context.running.wait()

    text_pushes_before_reload = [
        s["text"] for n, s in pushed if n == "vspeech-text" and "text" in s
    ]
    # panel "n" has anchor "s" (reversed join): newest first.
    assert text_pushes_before_reload[-1] == "BBB AAA"

    new_config = config.model_copy(deep=True)  # mirrors lib/command.py's
    # `context.config = new_config` rebind
    new_config.subtitle.text.delimiter = "|"
    context.config = new_config
    worker_meta.need_reload = True
    context.running.set()  # release the gate; the next loop iteration's
    # need_reload check fires before any new message or expiry, isolating
    # _apply_reload's own effect
    await _settle()

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )
    assert worker_meta.need_reload is False  # confirms _apply_reload ran

    text_pushes_after_reload = [
        s["text"] for n, s in pushed if n == "vspeech-text" and "text" in s
    ]
    # unfixed (_apply_reload: drop _push_all_text), this stays "BBB AAA" --
    # the reload's style push still happens, but nothing re-renders the
    # *text* with the new delimiter until a message or expiry, neither of
    # which happened here.
    assert text_pushes_after_reload[-1] == "BBB|AAA"

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task


async def test_connect_time_refresh_repoints_panels_to_config_changed_while_connected_and_unflagged(
    monkeypatch,
):
    # ADR-0042 / fix pass 5, item 3 (C2, audit): the earlier "redundant"
    # reasoning was that _run_session's first turn always runs
    # _apply_reload and converges -- but _apply_reload only ever fires
    # `if context.need_reload`, and lib/command.py's reload handler
    # (lines 60-74) resets and re-evaluates *each* worker's need_reload
    # fresh on every single reload event, diffed only against whatever
    # context.config already is at that moment. So a *second* reload that
    # doesn't touch `subtitle` -- landing before this worker ever consumes
    # the first one's flag, e.g. while OBS is down -- clears a
    # still-unconsumed True back to False even though
    # `context.config.subtitle` has, cumulatively, changed since
    # `make_panels()` built `panels[key].config`. That state (config
    # rebound, need_reload never set) is reproduced directly here instead of
    # replaying two full reload cycles through a temp config file.
    #
    # Experiment: with need_reload never True, `_apply_reload`'s self-heal
    # can never fire in the new session either -- so if the connect-time
    # `_refresh_panel_configs` call is the *only* thing that re-points
    # `panels["n"].config` at the live config, dropping it means the next
    # session pushes the *stale* style forever, not just transiently.
    fake_logger = RecordingLogger()
    pushed: list[tuple[int, str, dict]] = []
    monkeypatch.setattr(subtitle_obs_mod, "connect", make_fake_connect())
    monkeypatch.setattr(
        subtitle_obs_mod,
        "ObsWsClient",
        make_style_recording_crash_on_text_client(pushed, "disconnect-me"),
    )
    monkeypatch.setattr(subtitle_obs_mod, "logger", fake_logger)

    async def fake_sleep(sec: float) -> None:
        return None

    monkeypatch.setattr(subtitle_obs_mod, "sleep", fake_sleep)

    config = make_config()  # default font_color "#ffffff"
    context = SharedContext(config=config)
    worker_meta = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    in_queue: Queue[WorkerInput] = Queue()

    task = asyncio.create_task(
        subtitle_obs_mod.subtitle_obs_worker(context, in_queue),
        name=worker_meta.event.name,
    )
    await _settle()  # session 0 connects, pushes the default style, parks
    # on the empty queue

    new_config = config.model_copy(deep=True)  # mirrors lib/command.py's
    # `context.config = new_config` rebind
    new_config.subtitle.text.font_color = "#0080ff"
    context.config = new_config
    assert worker_meta.need_reload is False  # the race this test reproduces

    await in_queue.put(make_message("disconnect-me"))
    await _settle()  # session 0 dies pushing "disconnect-me"; session 1
    # reconnects

    assert not task.done(), (
        f"worker crashed: {task.exception() if task.done() else None}"
    )

    session1_text_styles = [
        settings
        for session, name, settings in pushed
        if session == 1 and name == "vspeech-text" and "color" in settings
    ]
    assert session1_text_styles, "session 1 never pushed a style for vspeech-text"
    # unfixed (connect-time _refresh_panel_configs dropped), panels["n"]
    # .config is still the *old* Config's subtitle.text object -- and
    # need_reload is False, so _apply_reload never runs to fix it either.
    # This would stay the default font_color's int forever.
    assert session1_text_styles[0]["color"] == hex_color_to_obs_int("#0080ff")

    task.cancel()
    with pytest.raises(WorkerShutdown):
        await task
