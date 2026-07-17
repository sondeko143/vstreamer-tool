"""subtitle の OBS バックエンド (ADR-0040 / 0041 / 0042)。

obs-websocket のクライアントとして OBS の Text (GDI+) ソースへ字幕を push
する。OBS の構造 (シーン・input の存在・配置) には触らず、ユーザーが作った
input の設定値だけを更新する (ADR-0041)。

失敗の扱いは「観測できたものだけ即死」(ADR-0042):
  - 接続できない / 切断 / タイムアウト / 不正メッセージ -> fail-open
    (warn once + バックオフ再接続)。字幕は落ちるが音声は生き続ける。
  - 認証失敗 / ソース不在 -> fail-loud (WorkerStartupError)
  - 壊れた色設定 (#rrggbb でない Tk 専用色名など) -> DEGRADE (warn +
    直前のスタイルを維持して継続)。起動時点の値は preflight (層A,
    `preflight._check_subtitle`) が fail-loud に弾くが、reload はそこを
    通らないので、reload 後 (または reload 直後の再接続) にだけ壊れた値が
    残る余地があり、ここで拾う (fix pass 1, finding 1)。
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
# セッションがこの秒数以上続けば「健全だった」とみなし backoff/warn を戻す
# (fix pass 1, finding 2)。MAX_BACKOFF_SEC を流用する: どのみち次の再接続
# までは最大この秒数だけ待つ設計なので、それより長く生きたセッションは
# 「即座に壊れて再接続ループしている」ものとは質的に別物と判断できる。
SESSION_HEALTHY_SEC = MAX_BACKOFF_SEC


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


async def _push_styles_or_warn(
    client: ObsRequester, config: SubtitleConfig, panels: dict[str, Texts]
) -> None:
    """`push_styles` を試すが、色の値が壊れていてもプロセスを道連れにしない。

    preflight (層A) は起動時点の config しか見ないので、reload で入った壊れた
    `#rrggbb` (Tk 色名など) はそこで止められない -- 起点は色フィールドだけ
    TK/OBS 共有かつ preflight は起動時にしか走らないという ADR-0038/0042 の
    隙間 (fix pass 1, finding 1 (Critical))。ここに来る `ValueError` は
    `hex_color_to_obs_int` が返す「観測できて、かつ config を直せば直る」
    種類だけで、繋がっている音声パイプラインを落とす理由にはならない --
    DEGRADE (warn + 直前のスタイルを維持して継続)。`push_styles` はパネル
    ごとに `SetInputSettings` を送るので、途中で失敗しても既に送信済みの
    パネルへは反映済み・失敗したパネルは OBS 側の直前の値がそのまま残る。

    呼び出し側は 2 箇所 (初回/再接続時の接続直後、reload 時) あり、どちらも
    同じ理由で ValueError を漏らしてはいけないので 1 箇所にまとめる。
    """
    try:
        await push_styles(client, config, panels)
    except ValueError as e:
        logger.warning(
            "subtitle worker [obs] style rejected by OBS (%s); keeping the "
            "previous style and continuing.",
            e,
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


async def _apply_reload(
    context: SharedContext, client: ObsRequester, panels: dict[str, Texts]
) -> None:
    """reload で新しい config を取り込み、スタイルと現在テキストを再 push する。

    色が壊れていても (fix pass 1, finding 1) `_push_styles_or_warn` が
    ValueError を飲むので、ここは常に最後まで走り、テキストは必ず更新
    される。
    """
    context.reset_need_reload()
    _refresh_panel_configs(context, panels)
    await _push_styles_or_warn(client, context.config.subtitle, panels)
    await _push_all_text(client, context.config.subtitle.obs, panels)


def _age_across_outage(
    panels: dict[str, Texts], last_tick: list[float], now: float
) -> None:
    """接続直後の一斉 push の前に、直前のセッション終了 (または起動) からの
    経過時間ぶん字幕を老化させる。

    `last_tick` は `subtitle_obs_worker` と `_run_session` が共有する 1 要素
    のリスト。単なるローカル変数だと再接続のたびに失われ、再接続に要した
    時間 (=OBS が落ちていた時間) がエイジングに反映されず、古い字幕が
    そのまま復帰直後に再表示されてしまう (fix pass 1, finding 3)。
    `age_panels` は各パネルの `values[0]` しか老化させないが、直後の
    `_push_all_text` はパネル全部を無条件に再送するので、ここで
    `age_panels` の戻り値 (変化したパネルの一覧) を使う必要はない。
    """
    age_panels(panels, now - last_tick[0])
    last_tick[0] = now


async def _run_session(
    context: SharedContext,
    client: ObsRequester,
    in_queue: Queue[WorkerInput],
    panels: dict[str, Texts],
    last_tick: list[float],
) -> None:
    """繋がっている間の本ループ。30fps のビジーループは持たない。

    次に消える字幕の時刻までを timeout にして待つので、何も起きていない間は
    1 回も起きない。
    """
    while True:
        if context.need_reload:
            await _apply_reload(context, client, panels)
        timeout = next_expiry_sec(panels)
        message: WorkerInput | None = None
        try:
            message = await wait_for(in_queue.get(), timeout=timeout)
        except TimeoutError:
            pass
        now = monotonic()
        for ts in age_panels(panels, now - last_tick[0]):
            await push_text(client, context.config.subtitle.obs, panels, ts)
        last_tick[0] = now
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
    # セッション境界をまたいで表示時計を進め続ける (fix pass 1, finding 3)。
    # subtitle_obs_worker と _run_session が同じ 1 要素リストを共有・変更
    # する — 詳細は _age_across_outage のドキュメント参照。
    last_tick: list[float] = [monotonic()]
    try:
        while True:
            obs = context.config.subtitle.obs
            # そのセッションが「健全だった」と言える継続時間を測るための
            # 開始時刻。identify+ソース検証が終わるまでは None のままにし、
            # 「即座に壊れて再接続ループしている」セッションと区別する
            # (fix pass 1, finding 2)。
            session_started: float | None = None
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
                    session_started = monotonic()
                    _refresh_panel_configs(context, panels)
                    await _push_styles_or_warn(client, context.config.subtitle, panels)
                    _age_across_outage(panels, last_tick, monotonic())
                    await _push_all_text(client, obs, panels)
                    await _run_session(context, client, in_queue, panels, last_tick)
            except (OSError, WebSocketException, ObsProtocolError) as e:
                # OBS 未起動・切断・タイムアウト・不正メッセージ。字幕は落ちるが
                # 音声パイプラインは巻き込まない。ObsProtocolError を external に
                # 落とさないこと: 素の TimeoutError/KeyError が worker を貫通すると
                # TaskGroup ごとプロセスが死ぬ。
                #
                # backoff/warned は「identify+ソース検証まで到達した」だけでは
                # 戻さない -- SetInputSettings が毎回リクエスト直後に失敗する
                # ような「繋がるが即座に壊れる」ケースが繰り返し起きると、
                # 接続の瞬間にリセットしてしまい backoff が床に張り付いたまま
                # 警告だけが毎回出る (fix pass 1, finding 2, 実測)。セッションが
                # SESSION_HEALTHY_SEC 以上生きていた場合だけ「健全だった」と
                # みなして戻す。
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
