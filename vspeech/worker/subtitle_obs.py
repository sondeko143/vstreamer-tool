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
    残る余地があり、ここで拾う。
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
# セッションがこの秒数以上続けば「健全だった」とみなし backoff/warn を戻す。
# MAX_BACKOFF_SEC を流用する: どのみち次の再接続
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


async def _push_panel_style(
    client: ObsRequester, config: SubtitleConfig, key: str, ts: Texts
) -> None:
    """1 パネル分のスタイルを OBS へ送る (ADR-0041: config が権威)。

    呼び出し側は `_push_styles_or_warn` (guarded path) だけにすること --
    壊れた色設定 (Tk 専用色名など) は `build_text_settings` が `ValueError`
    を投げるので、直接呼ぶ側はそれを自分で処理する責任を負う。かつて存在した
    無防備な公開ラッパー `push_styles` (両パネルをまとめて呼ぶだけの薄い関数)
    は削除した -- 誰かがこの分かりやすい公開名に手を伸ばして再導入すれば、
    bare `ValueError` が `TaskGroup` ごと音声パイプラインを道連れにする
    不具合がそのまま戻る。
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
    """パネルごとに `_push_panel_style` を試すが、色の値が壊れていても
    プロセスを道連れにしない。

    preflight (層A) は起動時点の config しか見ないので、reload で入った壊れた
    `#rrggbb` (Tk 色名など) はそこで止められない -- 起点は色フィールドだけ
    TK/OBS 共有かつ preflight は起動時にしか走らないという ADR-0038/0042 の
    隙間。ここに来る `ValueError` は
    `hex_color_to_obs_int` が返す「観測できて、かつ config を直せば直る」
    種類だけで、繋がっている音声パイプラインを落とす理由にはならない --
    DEGRADE (warn + 直前のスタイルを維持して継続)。ここに来る `ValueError` は
    OBS が拒否したのではなく、`hex_color_to_obs_int` がリクエスト送信前に
    ローカルで検出した壊れた入力値なので、OBS ではなく config の値を名指し
    する。

    パネルごとに個別の try/except でガードする:
    以前は `push_styles` をまとめて 1 つの try/except で囲っていたので、
    先に壊れたパネルより後のパネルは (値が有効でも) 一切 push されずに
    直前のスタイルのまま取り残されていた。パネルごとに独立させれば、
    どちらも「壊れていなければ最新の値を反映・壊れていれば直前の値を維持」
    をそれぞれ独立に満たせる。

    `style_warned` はパネルキーごとの warn-once フラグ: 同じ壊れた値が
    push に成功する (= config が直る) まで、
    何度失敗しても 1 回しか警告しない。OBS が繋がっては切れてを繰り返す
    (フラップする) 間、再接続のたびに同じ壊れた値を送り直すと、直すまで
    毎回警告が出ていた (測定: 20 回の再接続で 20 回の警告、対して隣の
    接続断の warn-once は正しく 1 回)。push が成功したらそのパネルの
    フラグを戻す -- 直った後に別の値でまた壊れたときは、ちゃんと再度
    警告するため (二度と警告しなくなるのを防ぐ)。

    呼び出し側は 2 箇所 (初回/再接続時の接続直後、reload 時) あり、どちらも
    同じ理由で ValueError を漏らしてはいけないので 1 箇所にまとめる。
    `style_warned` は `subtitle_obs_worker` が再接続をまたいで保持する 1 つの
    辞書を両方の呼び出し site に渡す -- セッションごとに作り直すと、
    フラップのたびに warn-once がリセットされてしまう。
    """
    for key, ts in panels.items():
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
    """reload で新しい config を取り込み、スタイルと現在テキストを再 push する。

    色が壊れていても `_push_styles_or_warn` が
    ValueError を飲むので、ここは常に最後まで走り、テキストは必ず更新
    される。
    """
    context.reset_need_reload()
    _refresh_panel_configs(context, panels)
    await _push_styles_or_warn(client, context.config.subtitle, panels, style_warned)
    await _push_all_text(client, context.config.subtitle.obs, panels)


def _age_across_outage(
    panels: dict[str, Texts], last_tick: list[float], now: float
) -> None:
    """接続直後の一斉 push の前に、直前のセッション終了 (または起動) からの
    経過時間ぶん字幕を老化させる。

    `last_tick` は `subtitle_obs_worker` と `_run_session` が共有する 1 要素
    のリスト。単なるローカル変数だと再接続のたびに失われ、再接続に要した
    時間 (=OBS が落ちていた時間) がエイジングに反映されず、古い字幕が
    そのまま復帰直後に再表示されてしまう。
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
    style_warned: dict[str, bool],
) -> None:
    """繋がっている間の本ループ。30fps のビジーループは持たない。

    次に消える字幕の時刻までを timeout にして待つので、何も起きていない間は
    1 回も起きない。
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
        # old order (aging push, then ingest) meant a message already
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
        for ts in aged:
            await push_text(client, context.config.subtitle.obs, panels, ts)
        if ingested is not None:
            await push_text(client, context.config.subtitle.obs, panels, ingested)
        if not context.running.is_set():
            await context.running.wait()


async def subtitle_obs_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    panels = make_panels(context.config.subtitle)
    backoff = INITIAL_BACKOFF_SEC
    warned = False
    # セッション境界をまたいで表示時計を進め続ける。
    # subtitle_obs_worker と _run_session が同じ 1 要素リストを共有・変更
    # する — 詳細は _age_across_outage のドキュメント参照。
    last_tick: list[float] = [monotonic()]
    # パネルキーごとの style warn-once フラグ。
    # セッション (再接続) をまたいで同じ辞書を使い回す -- フラップのたびに
    # 作り直すと、同じ壊れた値でも再接続のたびにまた警告してしまう
    # (see test_style_warn_once_persists_across_reconnects_not_just_within_a_session
    # in tests/test_subtitle_obs.py, which goes RED if this dict is moved
    # inside the loop below).
    style_warned: dict[str, bool] = {}
    try:
        while True:
            obs = context.config.subtitle.obs
            # そのセッションが「健全だった」と言える継続時間を測るための
            # 開始時刻。identify+ソース検証が終わるまでは None のままにし、
            # 「即座に壊れて再接続ループしている」セッションと区別する。
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
                    await _push_styles_or_warn(
                        client, context.config.subtitle, panels, style_warned
                    )
                    _age_across_outage(panels, last_tick, monotonic())
                    await _push_all_text(client, obs, panels)
                    await _run_session(
                        context, client, in_queue, panels, last_tick, style_warned
                    )
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
                # 警告だけが毎回出る。セッションが
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
