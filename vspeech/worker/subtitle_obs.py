"""subtitle の OBS バックエンド (ADR-0040 / 0041 / 0042)。

obs-websocket のクライアントとして OBS の Text (GDI+) ソースへ字幕を push
する。OBS の構造 (シーン・input の存在・配置) には触らず、ユーザーが作った
input の設定値だけを更新する (ADR-0041)。

失敗の扱いは「観測できたものだけ即死」(ADR-0042):
  - 接続できない / 切断 / タイムアウト / 不正メッセージ -> fail-open
    (warn once + バックオフ再接続)。字幕は落ちるが音声は生き続ける。
  - 認証失敗 / ソース不在 -> fail-loud (WorkerStartupError)
繋がるまでは両者を区別できないので、繋がるまで待つ。

ADR-0038 の層B は通常 exceptions.worker_startup を使うが、ここでは使わない:
あれは except Exception で全てを WorkerStartupError に変えるため、identify 中の
タイムアウト (リトライで直る) まで fail-loud に化ける。回復不能と観測できた
2 型だけを拾って WorkerStartupError を送出する。
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


class ObsRequester(Protocol):
    async def request(
        self, request_type: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


def make_panels(config: SubtitleConfig) -> dict[str, Texts]:
    """tk バックエンドと同じ 2 パネル。

    bb_width/bb_height は tk の Canvas 実寸のための値で、OBS 側は extents で
    表現するのでレイアウトには使わない。共有 dataclass の必須項目なので
    config の窓寸法をそのまま入れておく。
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
    """両ソースが OBS に実在することを確かめる。

    存在しなければ ObsResourceNotFoundError が上がる。呼び出し側はそれを
    ObsIdentifyError と並べて名指しで捕まえ、WorkerStartupError へ変える
    (fail-loud, ADR-0042)。この 2 型だけが「繋がった上で観測できて、かつ
    リトライしても直らない」失敗であり、他の ObsProtocolError は fail-open
    の再接続に落ちる。exceptions.worker_startup は使わない — その except
    Exception はここでは広すぎ、回復可能なタイムアウトまで致命化する。
    """
    for source in (obs.text_source, obs.translated_source):
        await client.request("GetInputSettings", {"inputName": source})


async def push_styles(
    client: ObsRequester, config: SubtitleConfig, panels: dict[str, Texts]
) -> None:
    """config のスタイルを両ソースへ流し込む (ADR-0041: config が権威)。"""
    for key, ts in panels.items():
        await client.request(
            "SetInputSettings",
            {
                "inputName": _source_of(key, config.obs),
                "inputSettings": build_text_settings(ts.config, config),
                "overlay": True,
            },
        )


async def push_text(
    client: ObsRequester,
    obs: SubtitleObsConfig,
    panels: dict[str, Texts],
    ts: Texts,
) -> None:
    """パネルの現在の文字列を対応するソースへ送る。

    `Texts.texts` が区切り文字での結合と "s" アンカーの逆順を済ませているので、
    ここは送るだけ。空なら空文字を送って消す。
    """
    await client.request(
        "SetInputSettings",
        {
            "inputName": _source_of(_panel_key(panels, ts), obs),
            "inputSettings": {"text": ts.texts},
            "overlay": True,
        },
    )


async def _push_all_text(
    client: ObsRequester, obs: SubtitleObsConfig, panels: dict[str, Texts]
) -> None:
    for ts in panels.values():
        await push_text(client, obs, panels, ts)


def _refresh_panel_configs(context: SharedContext, panels: dict[str, Texts]) -> None:
    panels["n"].config = context.config.subtitle.text
    panels["n"].anchor = context.config.subtitle.text.anchor
    panels["s"].config = context.config.subtitle.translated
    panels["s"].anchor = context.config.subtitle.translated.anchor


async def _run_session(
    context: SharedContext,
    client: ObsRequester,
    in_queue: Queue[WorkerInput],
    panels: dict[str, Texts],
) -> None:
    """繋がっている間の本ループ。30fps のビジーループは持たない。

    次に消える字幕の時刻までを timeout にして待つので、何も起きていない間は
    1 回も起きない。
    """
    last_tick = monotonic()
    while True:
        if context.need_reload:
            context.reset_need_reload()
            _refresh_panel_configs(context, panels)
            await push_styles(client, context.config.subtitle, panels)
            await _push_all_text(client, context.config.subtitle.obs, panels)
        timeout = next_expiry_sec(panels)
        message: WorkerInput | None = None
        try:
            message = await wait_for(in_queue.get(), timeout=timeout)
        except TimeoutError:
            pass
        now = monotonic()
        for ts in age_panels(panels, now - last_tick):
            await push_text(client, context.config.subtitle.obs, panels, ts)
        last_tick = now
        if message is not None:
            ts = ingest_text(panels, message)
            await push_text(client, context.config.subtitle.obs, panels, ts)
        if not context.running.is_set():
            await context.running.wait()


async def subtitle_obs_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    panels = make_panels(context.config.subtitle)
    backoff = INITIAL_BACKOFF_SEC
    warned = False
    try:
        while True:
            obs = context.config.subtitle.obs
            try:
                async with connect(obs.url) as ws:
                    client = ObsWsClient(ws)
                    try:
                        await client.identify(obs.password.get_secret_value())
                        await validate_sources(client, obs)
                    except (ObsIdentifyError, ObsResourceNotFoundError) as e:
                        # 繋がった上で観測できて、かつリトライしても直らない。
                        # ここだけが fail-loud (ADR-0042)。他の ObsProtocolError
                        # (タイムアウト・不正メッセージ) は下の except へ落ちる。
                        raise WorkerStartupError("subtitle", str(e)) from e
                    logger.info("subtitle worker [obs] connected to %s", obs.url)
                    backoff = INITIAL_BACKOFF_SEC
                    warned = False
                    _refresh_panel_configs(context, panels)
                    await push_styles(client, context.config.subtitle, panels)
                    await _push_all_text(client, obs, panels)
                    await _run_session(context, client, in_queue, panels)
            except (OSError, WebSocketException, ObsProtocolError) as e:
                # OBS 未起動・切断・タイムアウト・不正メッセージ。字幕は落ちるが
                # 音声パイプラインは巻き込まない。ObsProtocolError を external に
                # 落とさないこと: 素の TimeoutError/KeyError が worker を貫通すると
                # TaskGroup ごとプロセスが死ぬ。
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
