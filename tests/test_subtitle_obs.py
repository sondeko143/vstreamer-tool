import asyncio
from asyncio import Queue
from uuid import uuid4

import pytest

import vspeech.worker.subtitle_obs as subtitle_obs_mod
from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import SubtitleWorkerType
from vspeech.exceptions import WorkerShutdown
from vspeech.lib.obs_ws import ObsRequestError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.subtitle_obs import make_panels
from vspeech.worker.subtitle_obs import push_styles
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


async def test_push_text_uses_overlay_so_it_does_not_clobber_style():
    config = make_config()
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_text(client, config.subtitle.obs, panels, panels["n"])
    assert all(d["overlay"] is True for t, d in client.calls if t == "SetInputSettings")


async def test_push_styles_sends_both_panels_with_config_values():
    config = make_config()
    config.subtitle.text.font_color = "#ff8000"
    config.subtitle.translated.font_size = 22
    panels = make_panels(config.subtitle)
    client = FakeObsClient()
    await push_styles(client, config.subtitle, panels)
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
# warn-once / reload / the pause gate.


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

    await subtitle_obs_mod._push_styles_or_warn(client, config.subtitle, panels)

    assert any("white" in w for w in fake_logger.warnings)
    # the "n" panel's colours are still valid defaults and was pushed first,
    # so its style already reached OBS before the "s" panel's push raised.
    assert client.settings_for("vspeech-text")


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
