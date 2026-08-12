"""The OBS backend of subtitle (ADR-0040 / 0041 / 0042).

Pushes subtitles into OBS's Text (GDI+) sources as an obs-websocket client. It never
touches OBS's structure (scenes, the existence of inputs, their placement) and only
updates the settings of the inputs the user created (ADR-0041).

Failures follow "only die on what we could observe" (ADR-0042):
  - cannot connect / disconnected / timeout / malformed message -> fail-open
    (warn once + reconnect with backoff). Subtitles drop but the audio stays alive.
  - auth failure / missing source -> fail-loud (WorkerStartupError)
  - a broken color setting (a Tk-only color name that is not #rrggbb, etc.) -> DEGRADE
    (warn + keep the previous style and continue). The values present at startup are
    rejected fail-loud by preflight (layer A, `preflight._check_subtitle`), but a reload
    does not go through there, so a broken value can only survive after a reload (or on
    the reconnect right after one), and that is what is caught here.
The first two cannot be told apart until we connect, so we wait until we do.

ADR-0038's layer B normally uses exceptions.worker_startup, which is deliberately not
used here: its except Exception turns everything into WorkerStartupError, which would
also make a timeout during identify (which retrying fixes) fail loud. Only the two types
observed to be unrecoverable are caught and raised as WorkerStartupError.

On an axis separate from those three tiers, `subtitle.obs.translated_source` may be empty
(`preflight._check_subtitle` requires only text_source). For a pipeline that uses no
translation, nothing about the "s" panel is pushed -- neither validation, nor style, nor
text (validate_sources / _push_styles_or_warn / _push_text_if_routed). That is a
legitimate configuration rather than an observed failure and belongs to none of the three
tiers above -- except that when a p=s message actually arrives (i.e. a translation that
should have been shown disappears), it is reported exactly once, warn-once (for the same
reason as ADR-0041's "changing a setting and having nothing happen is worse than the
setting not existing").
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import sleep
from asyncio import wait_for
from time import monotonic
from typing import Any
from typing import Protocol

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleObsConfig
from vspeech.exceptions import WorkerStartupError
from vspeech.exceptions import shutdown_worker
from vspeech.lib.obs_text_settings import build_text_settings
from vspeech.lib.obs_ws import ObsIdentifyError
from vspeech.lib.obs_ws import ObsProtocolError
from vspeech.lib.obs_ws import ObsResourceNotFoundError
from vspeech.lib.obs_ws import ObsWsClient
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import age_panels
from vspeech.lib.subtitle_state import ingest_text
from vspeech.lib.subtitle_state import next_expiry_sec
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput

INITIAL_BACKOFF_SEC = 0.5
MAX_BACKOFF_SEC = 5.0
# A session that lasted at least this many seconds counts as "was healthy" and resets the
# backoff/warn state. MAX_BACKOFF_SEC is reused: the design already waits at most this
# long before the next reconnect, so a session that outlived it is qualitatively different
# from one that "breaks instantly and loops on reconnect".
SESSION_HEALTHY_SEC = MAX_BACKOFF_SEC


class ObsRequester(Protocol):
    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


def make_panels(config: SubtitleConfig) -> dict[str, Texts]:
    """The same two panels as the tk backend.

    bb_width/bb_height exist for tk's real Canvas dimensions; OBS expresses that with
    extents and does not use them for layout. They are required fields of the shared
    dataclass, so the config's window dimensions go in as-is.
    """
    return {
        "n": Texts(
            tag="text",
            anchor=config.text.anchor,
            config=config.text,
            bb_width=config.window_width,
            bb_height=config.window_height,
        ),
        "s": Texts(
            tag="translated",
            anchor=config.translated.anchor,
            config=config.translated,
            bb_width=config.window_width,
            bb_height=config.window_height,
        ),
    }


def _source_of(panel_key: str, obs: SubtitleObsConfig) -> str:
    return obs.text_source if panel_key == "n" else obs.translated_source


def _panel_key(panels: dict[str, Texts], ts: Texts) -> str:
    for key, panel in panels.items():
        if panel is ts:
            return key
    raise KeyError("panel not found")


async def validate_sources(client: ObsRequester, obs: SubtitleObsConfig) -> None:
    """Verify that text_source (always required) and translated_source (when configured)
    actually exist in OBS.

    When translated_source is empty it is never pushed to in the first place, so its
    existence is not checked either (`_check_subtitle` made only text_source required,
    ADR-0041) -- passing an empty string to GetInputSettings is not a meaningful check,
    and such a user has no obligation to prepare a `translated_source` in OBS.

    If a source is missing, ObsResourceNotFoundError is raised. The caller catches it by
    name alongside ObsIdentifyError and converts it into WorkerStartupError (fail-loud,
    ADR-0042). Those two types are the only failures that are "observable once connected
    and not fixed by retrying"; every other ObsProtocolError falls into the fail-open
    reconnect. exceptions.worker_startup is not used -- its except Exception is too broad
    here and would make a recoverable timeout fatal.
    """
    sources = [obs.text_source]
    if obs.translated_source:
        sources.append(obs.translated_source)
    for source in sources:
        await client.request("GetInputSettings", {"inputName": source})


async def _push_panel_style(
    client: ObsRequester, config: SubtitleConfig, key: str, ts: Texts
) -> None:
    """Send one panel's style to OBS (ADR-0041: the config is the authority).

    Call this only from `_push_styles_or_warn` (the guarded path) -- a broken color
    setting (a Tk-only color name, etc.) makes `build_text_settings` raise `ValueError`,
    so anyone calling directly takes on the responsibility of handling it. Do not add an
    unguarded public wrapper that merely calls it for both panels -- a bare `ValueError`
    would take the audio pipeline down along with the `TaskGroup`.
    """
    await client.request(
        "SetInputSettings",
        {
            "inputName": _source_of(key, config.obs),
            "inputSettings": build_text_settings(ts.config, config),
            "overlay": True,
        },
    )


async def _push_styles_or_warn(
    client: ObsRequester,
    config: SubtitleConfig,
    panels: dict[str, Texts],
    style_warned: dict[str, bool],
) -> None:
    """Try `_push_panel_style` per panel without taking the process down when a color
    value is broken.

    preflight (layer A) only looks at the config as it was at startup, so a broken
    `#rrggbb` (a Tk color name, etc.) introduced by a reload cannot be stopped there --
    the gap in ADR-0038/0042 that arises because the color fields alone are shared between
    TK and OBS while preflight only runs at startup. The `ValueError` reaching here is
    only the kind `hex_color_to_obs_int` returns -- observable and fixable by correcting
    the config -- and is no reason to bring down a working audio pipeline: DEGRADE (warn +
    keep the previous style and continue). It was not OBS that rejected the value: it is a
    broken input detected locally by `hex_color_to_obs_int` before any request was sent,
    so the message names the config value rather than OBS.

    Each panel is guarded by its own try/except: keeping them independent lets each panel
    satisfy "reflect the newest value if valid, keep the previous value if broken" on its
    own. Wrapping both in a single try/except would leave every panel after the first
    broken one unpushed (even with valid values) and stuck on its previous style.

    `style_warned` is a per-panel-key warn-once flag: no matter how many times the push
    fails, only one warning is emitted until the same broken value pushes successfully
    (i.e. the config is fixed). While OBS flaps (connects and drops repeatedly), the same
    broken value is re-sent on every reconnect and used to warn every time until it was
    fixed (measured: 20 warnings across 20 reconnects, versus the neighbouring
    disconnection warn-once, which correctly emitted once). On a successful push the
    panel's flag is cleared -- so that if a different value breaks later, it warns again
    (rather than falling silent forever).

    There are two call sites (right after connecting on first connect/reconnect, and on
    reload) and neither may leak a ValueError for the same reason, so they share this one
    function. `style_warned` is a single dict held across reconnects by
    `subtitle_obs_worker` and passed to both call sites -- rebuilding it per session would
    reset the warn-once on every flap.

    The "s" panel with an empty translated_source has nowhere to push at all, so it is
    skipped entirely (ADR-0041). Unlike the push_text side's skip
    (`_push_text_if_routed`), nothing is lost here (a style is meaningless with nothing to
    display), so there is no warn-once -- the thing the user genuinely needs to hear, "a
    p=s subtitle body disappeared", can only happen on the push_text side.
    """
    for key, ts in panels.items():
        if key == "s" and not config.obs.translated_source:
            continue
        try:
            await _push_panel_style(client, config, key, ts)
        except ValueError as e:
            if not style_warned.get(key, False):
                logger.warning(
                    "subtitle worker [obs] invalid style value for %s (%s); "
                    "keeping the previous style and continuing.",
                    ts.tag,
                    e,
                )
                style_warned[key] = True
            else:
                logger.debug(
                    "subtitle worker [obs] still invalid style value for %s: %s",
                    ts.tag,
                    e,
                )
        else:
            style_warned[key] = False


async def push_text(
    client: ObsRequester,
    obs: SubtitleObsConfig,
    panels: dict[str, Texts],
    ts: Texts,
) -> None:
    """Send the panel's current string to the corresponding source.

    `Texts.texts` has already done the separator join and the reversal for the "s" anchor,
    so this only sends. When empty it sends an empty string, clearing the display.
    """
    await client.request(
        "SetInputSettings",
        {
            "inputName": _source_of(_panel_key(panels, ts), obs),
            "inputSettings": {"text": ts.texts},
            "overlay": True,
        },
    )


def _translated_dest_missing(
    obs: SubtitleObsConfig, panels: dict[str, Texts], ts: Texts
) -> bool:
    """True when `ts` is the "s" (translated) panel and translated_source
    is empty -- there is nowhere in OBS to send it (ADR-0041: an empty
    translated_source means this pipeline has no translation step, not a
    typo; `_check_subtitle` (preflight) only requires text_source).
    """
    return ts is panels.get("s") and not obs.translated_source


async def _push_text_if_routed(
    client: ObsRequester,
    obs: SubtitleObsConfig,
    panels: dict[str, Texts],
    ts: Texts,
) -> None:
    """`push_text`, but silently skips the "s" panel when it has no
    destination, instead of calling `push_text` (which would resolve
    `_source_of` to the empty string and hand OBS an empty `inputName`).

    Every push_text call site in this module goes through here rather than
    calling push_text directly, so a missing translated_source can never
    reach OBS as an empty inputName. This runs on every connect, reload,
    and aging tick regardless of whether a translation was ever routed
    here, so it stays silent -- warning on all of those would spam the log
    on every reconnect for a pipeline that simply has no translation step.
    The one case worth telling the user about (a routed p=s message with
    nowhere to go) is handled separately, at the point of ingest in
    `_run_session`, where it can be told apart from this housekeeping
    no-op.
    """
    if _translated_dest_missing(obs, panels, ts):
        return
    await push_text(client, obs, panels, ts)


async def _push_all_text(
    client: ObsRequester, obs: SubtitleObsConfig, panels: dict[str, Texts]
) -> None:
    for ts in panels.values():
        await _push_text_if_routed(client, obs, panels, ts)


def _refresh_panel_configs(context: SharedContext, panels: dict[str, Texts]) -> None:
    panels["n"].config = context.config.subtitle.text
    # Not redundant with the line above:
    # Texts.texts (the join order push_text sends) reads ts.anchor directly,
    # while build_text_settings (the style push) reads ts.config.anchor --
    # two different attributes that happen to start in sync. Re-pointing
    # .config alone fixes the style but leaves the join order on the
    # pre-reload anchor.
    panels["n"].anchor = context.config.subtitle.text.anchor
    panels["s"].config = context.config.subtitle.translated
    # See the "n" panel's comment above -- same reason, same non-redundancy.
    panels["s"].anchor = context.config.subtitle.translated.anchor


async def _apply_reload(
    context: SharedContext,
    client: ObsRequester,
    panels: dict[str, Texts],
    style_warned: dict[str, bool],
) -> None:
    """Take in the new config on reload and re-push the styles and the current text.

    Even with a broken color, `_push_styles_or_warn` swallows the ValueError, so this
    always runs to the end and the text is always updated.
    """
    context.reset_need_reload()
    _refresh_panel_configs(context, panels)
    await _push_styles_or_warn(client, context.config.subtitle, panels, style_warned)
    await _push_all_text(client, context.config.subtitle.obs, panels)


def _age_across_outage(
    panels: dict[str, Texts], last_tick: list[float], now: float
) -> None:
    """Age the subtitles by the time elapsed since the previous session ended (or since
    startup), before the bulk push that follows a connect.

    `last_tick` is a one-element list shared by `subtitle_obs_worker` and `_run_session`.
    As a plain local it would be lost on every reconnect, so the time the reconnect took
    (i.e. how long OBS was down) would not be reflected in the aging and stale subtitles
    would reappear as-is right after recovery.
    `age_panels` only ages each panel's `values[0]`, but the `_push_all_text` right after
    re-sends every panel unconditionally, so there is no need to use `age_panels`' return
    value (the list of panels that changed) here.
    """
    age_panels(panels, now - last_tick[0])
    last_tick[0] = now


async def _run_session(
    context: SharedContext,
    client: ObsRequester,
    in_queue: Queue[WorkerInput],
    panels: dict[str, Texts],
    last_tick: list[float],
    style_warned: dict[str, bool],
    dest_warned: list[bool],
) -> None:
    """The main loop while connected. There is no 30fps busy loop.

    It waits with a timeout set to the moment the next subtitle expires, so while nothing
    is happening it never wakes up.

    `dest_warned` is the flag (a one-element list) for warning exactly once that "a p=s
    message arrived but translated_source is empty, so it has nowhere to go". For the same
    reason as `style_warned`/`last_tick`, `subtitle_obs_worker` holds the object across
    reconnects and passes the same one in -- rebuilding it here would warn again on every
    flap.
    """
    while True:
        if context.need_reload:
            await _apply_reload(context, client, panels, style_warned)
        timeout = next_expiry_sec(panels)
        message: WorkerInput | None = None
        try:
            message = await wait_for(in_queue.get(), timeout=timeout)
        except TimeoutError:
            pass
        now = monotonic()
        # Ingest before either push: ingest_text is
        # a pure state update and cannot raise, unlike push_text below. The
        # alternative order (aging push, then ingest) would mean a message already
        # dequeued could vanish without ever reaching OBS in any session and
        # without being re-queued, if the *aging* push raised (killing the
        # session) before the ingest ever ran -- narrow (needs a message and
        # an expiry in the same turn, and that push to fail), but the
        # surrounding code already works hard to make *ingested* text
        # survive an outage (see _age_across_outage), so a message arriving
        # 1ms earlier would have survived and one arriving now wouldn't.
        # Ingesting first means the aging push's own failure can no longer
        # lose the message -- the next session's reconnect (_push_all_text)
        # still carries it. In the rare case the aged and newly-ingested
        # text land on the same panel in the same turn, that panel is pushed
        # twice in a row with identical (already up-to-date) content --
        # redundant, not incorrect.
        aged = age_panels(panels, now - last_tick[0])
        last_tick[0] = now
        ingested = ingest_text(panels, message) if message is not None else None
        obs = context.config.subtitle.obs
        for ts in aged:
            await _push_text_if_routed(client, obs, panels, ts)
        if ingested is not None:
            # ingest_text has no notion of push config -- it routes a p=s
            # message into the "s" panel regardless of whether
            # translated_source is set, so this is the one place that can
            # tell a genuine drop (a translated subtitle just vanished)
            # apart from _push_text_if_routed's routine no-op skip above
            # (connect/reload/aging housekeeping with nothing new to lose).
            if _translated_dest_missing(obs, panels, ingested) and not dest_warned[0]:
                logger.warning(
                    "subtitle worker [obs] received a translated (p=s) "
                    "subtitle but subtitle.obs.translated_source is empty; "
                    "dropping it and continuing without it.",
                )
                dest_warned[0] = True
            await _push_text_if_routed(client, obs, panels, ingested)
        if not context.running.is_set():
            await context.running.wait()


async def subtitle_obs_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    panels = make_panels(context.config.subtitle)
    backoff = INITIAL_BACKOFF_SEC
    warned = False
    # Keeps the display clock advancing across session boundaries.
    # subtitle_obs_worker and _run_session share and mutate the same one-element list --
    # see _age_across_outage's docs for the details.
    last_tick: list[float] = [monotonic()]
    # Per-panel-key style warn-once flags.
    # The same dict is reused across sessions (reconnects) -- rebuilding it on every flap
    # would warn again on every reconnect even for the same broken value
    # (see test_style_warn_once_persists_across_reconnects_not_just_within_a_session
    # in tests/worker/test_subtitle_obs.py, which goes RED if this dict is moved
    # inside the loop below).
    style_warned: dict[str, bool] = {}
    # The warn-once flag for "a p=s (translated) message arrived but translated_source is
    # empty, so it has nowhere to go". For the same reason as style_warned it is created
    # once here (outside the while True) and the same one-element list is reused across
    # reconnects
    # (see test_missing_translated_source_warn_once_persists_across_reconnects,
    # which goes RED if this list is moved inside the loop below).
    dest_warned: list[bool] = [False]
    try:
        while True:
            obs = context.config.subtitle.obs
            # The start time used to measure whether the session lasted long enough to
            # count as "was healthy". It stays None until identify plus source validation
            # finish, which distinguishes it from a session that "breaks instantly and
            # loops on reconnect".
            session_started: float | None = None
            try:
                async with connect(obs.url) as ws:
                    client = ObsWsClient(ws)
                    try:
                        await client.identify(obs.password.get_secret_value())
                        await validate_sources(client, obs)
                    except (ObsIdentifyError, ObsResourceNotFoundError) as e:
                        # Observable once connected and not fixed by retrying. This is the
                        # only fail-loud path (ADR-0042). Other ObsProtocolErrors
                        # (timeouts, malformed messages) fall to the except below.
                        raise WorkerStartupError("subtitle", str(e)) from e
                    logger.info("subtitle worker [obs] connected to %s", obs.url)
                    session_started = monotonic()
                    _refresh_panel_configs(context, panels)
                    await _push_styles_or_warn(
                        client, context.config.subtitle, panels, style_warned
                    )
                    _age_across_outage(panels, last_tick, monotonic())
                    await _push_all_text(client, obs, panels)
                    await _run_session(
                        context,
                        client,
                        in_queue,
                        panels,
                        last_tick,
                        style_warned,
                        dest_warned,
                    )
            except (OSError, WebSocketException, ObsProtocolError) as e:
                # OBS not running, a disconnect, a timeout, a malformed message. The
                # subtitles drop but the audio pipeline is not dragged along. Never let
                # ObsProtocolError escape outward: a bare TimeoutError/KeyError piercing
                # the worker kills the process together with the TaskGroup.
                #
                # backoff/warned are not reset merely because we "got as far as identify
                # plus source validation" -- with a repeating "connects but breaks
                # instantly" case, such as SetInputSettings failing right after every
                # request, resetting at the moment of connection would pin the backoff to
                # the floor while the warning fires every time. They are reset only when
                # the session lived at least SESSION_HEALTHY_SEC, which counts as "was
                # healthy".
                if (
                    session_started is not None
                    and monotonic() - session_started >= SESSION_HEALTHY_SEC
                ):
                    backoff = INITIAL_BACKOFF_SEC
                    warned = False
                if not warned:
                    logger.warning(
                        "subtitle worker [obs] cannot reach %s (%s); "
                        "retrying in the background. Subtitles are not shown "
                        "until OBS is up.",
                        obs.url,
                        e,
                    )
                    warned = True
                else:
                    logger.debug("subtitle worker [obs] still unreachable: %s", e)
                await sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SEC)
    except CancelledError as e:
        logger.info("subtitle worker cancelled")
        raise shutdown_worker(e)
