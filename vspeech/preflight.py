"""起動時の設定 preflight (層A, ADR-0038)。

enable した各 worker の「実リソースを取得せずに安価に判定できる」設定不備
(必須フィールド・参照ファイル/ディレクトリの存在・デバイス発見可否・依存の
有無) を検査する。検出した全問題を集約して ConfigError で送出する。実ロードで
しか分からない失敗は worker 起動時に扱う (層B)。
"""

from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from vspeech.config import Config
from vspeech.config import F0ExtractorType
from vspeech.config import GcpConfig
from vspeech.config import SubtitleWorkerType
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.config import TtsWorkerType
from vspeech.config import VcConfig
from vspeech.exceptions import ConfigError
from vspeech.exceptions import ConfigProblem
from vspeech.lib.obs_text_settings import hex_color_to_obs_int
from vspeech.lib.subtitle_state import TRANSPARENT_BG_COLOR

Checker = Callable[[Config], list[ConfigProblem]]


def _check_gcp_credentials(gcp: GcpConfig, worker: str) -> list[ConfigProblem]:
    # 認証の実成立は層B。ここでは指定した key.json の存在だけ安価に見る。
    if gcp.service_account_file_path is not None:
        path = gcp.service_account_file_path.expanduser()
        if not path.is_file():
            return [
                ConfigProblem(
                    worker,
                    f"gcp.service_account_file_path '{path}' が存在しません",
                    field="gcp.service_account_file_path",
                )
            ]
    return []


def _check_vad_gate(
    cfg: TranscriptionConfig | VcConfig, worker: str
) -> list[ConfigProblem]:
    if not cfg.vad_gate:
        return []
    path = cfg.vad_model_file.expanduser()
    if not path.is_file():
        return [
            ConfigProblem(
                worker,
                f"vad_gate=true ですが vad_model_file '{path}' が存在しません",
                field=f"{worker}.vad_model_file",
            )
        ]
    return []


def _check_transcription(config: Config) -> list[ConfigProblem]:
    if not config.transcription.enable:
        return []
    w = "transcription"
    tc = config.transcription
    problems: list[ConfigProblem] = []
    if tc.worker_type == TranscriptionWorkerType.ACP:
        ami = config.ami
        required = (
            ("ami.appkey", ami.appkey.get_secret_value()),
            ("ami.engine_uri", ami.engine_uri),
            ("ami.engine_name", ami.engine_name),
            ("ami.service_id", ami.service_id),
        )
        for name, value in required:
            if not value:
                problems.append(
                    ConfigProblem(
                        w,
                        f"ACP バックエンドには {name} が必須ですが空です",
                        field=name,
                    )
                )
        if tc.transliterate_with_mozc and find_spec("mozcpy") is None:
            problems.append(
                ConfigProblem(
                    w,
                    "transliterate_with_mozc=true ですが mozcpy が未インストールです",
                    field="transcription.transliterate_with_mozc",
                )
            )
    elif tc.worker_type == TranscriptionWorkerType.GCP:
        problems.extend(_check_gcp_credentials(config.gcp, w))
    # WHISPER のモデル/GPU ロードは層B（起動時取得）。
    problems.extend(_check_vad_gate(tc, w))
    return problems


def _check_recording(config: Config) -> list[ConfigProblem]:
    if not config.recording.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_input_device
    from vspeech.shared_context import WorkerOutput

    w = "recording"
    problems: list[ConfigProblem] = []
    try:
        resolve_input_device(config.recording)
    except DeviceNotFoundError as e:
        problems.append(ConfigProblem(w, str(e), field="recording.input_device_index"))
    try:
        WorkerOutput.from_routes_list(config.recording.routes_list)
    except Exception as e:
        problems.append(
            ConfigProblem(
                w,
                f"recording.routes_list が不正です: {e}",
                field="recording.routes_list",
            )
        )
    return problems


def _check_playback(config: Config) -> list[ConfigProblem]:
    if not config.playback.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_output_device

    try:
        resolve_output_device(config.playback)
    except DeviceNotFoundError as e:
        return [ConfigProblem("playback", str(e), field="playback.output_device_index")]
    return []


def _check_translation(config: Config) -> list[ConfigProblem]:
    if not config.translation.enable:
        return []
    return _check_gcp_credentials(config.gcp, "translation")


def _check_tts(config: Config) -> list[ConfigProblem]:
    if not config.tts.enable:
        return []
    if config.tts.worker_type != TtsWorkerType.VOICEVOX:
        return []  # VR2 の実初期化は層B
    w = "tts"
    vv = config.voicevox
    problems: list[ConfigProblem] = []
    for name, path in (
        ("voicevox.openjtalk_dir", vv.openjtalk_dir),
        ("voicevox.model_dir", vv.model_dir),
    ):
        if not path.expanduser().is_dir():
            problems.append(
                ConfigProblem(w, f"{name} '{path}' が存在しません", field=name)
            )
    if (
        vv.onnxruntime_path is not None
        and not vv.onnxruntime_path.expanduser().is_file()
    ):
        problems.append(
            ConfigProblem(
                w,
                f"voicevox.onnxruntime_path '{vv.onnxruntime_path}' が存在しません",
                field="voicevox.onnxruntime_path",
            )
        )
    return problems


def _check_vc(config: Config) -> list[ConfigProblem]:
    if not config.vc.enable:
        return []
    w = "vc"
    rvc = config.rvc
    problems: list[ConfigProblem] = []
    if not rvc.model_file.expanduser().is_file():
        problems.append(
            ConfigProblem(
                w,
                f"rvc.model_file '{rvc.model_file}' が存在しません",
                field="rvc.model_file",
            )
        )
    hubert = rvc.hubert_model_file
    if hubert == Path() or not hubert.expanduser().is_dir():
        problems.append(
            ConfigProblem(
                w,
                f"rvc.hubert_model_file '{hubert}' (資産ディレクトリ) が存在しません",
                field="rvc.hubert_model_file",
            )
        )
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        if not rvc.rmvpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    w,
                    f"rvc.rmvpe_model_file '{rvc.rmvpe_model_file}' が存在しません",
                    field="rvc.rmvpe_model_file",
                )
            )
    if rvc.f0_extractor_type == F0ExtractorType.fcpe:
        if not rvc.fcpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    w,
                    f"rvc.fcpe_model_file '{rvc.fcpe_model_file}' が存在しません",
                    field="rvc.fcpe_model_file",
                )
            )
    problems.extend(_check_vad_gate(config.vc, w))
    return problems


def _check_subtitle(config: Config) -> list[ConfigProblem]:
    if not config.subtitle.enable:
        return []
    if config.subtitle.worker_type != SubtitleWorkerType.OBS:
        return []  # TK は接続先を持たない
    w = "subtitle"
    subtitle = config.subtitle
    obs = subtitle.obs
    problems: list[ConfigProblem] = []
    if not obs.url:
        problems.append(
            ConfigProblem(
                w,
                "OBS バックエンドには subtitle.obs.url が必須ですが空です",
                field="subtitle.obs.url",
            )
        )
    elif not obs.url.startswith(("ws://", "wss://")):
        problems.append(
            ConfigProblem(
                w,
                f"subtitle.obs.url '{obs.url}' は ws:// か wss:// で始まる必要があります",
                field="subtitle.obs.url",
            )
        )
    # text_source is asymmetric with translated_source, deliberately:
    # ingest_text (lib/subtitle_state.py) routes any message whose position
    # isn't a known panel key to the "n" panel, so text_source is the
    # backend's default destination -- empty means it does nothing at all.
    # translated_source has no such fallback ("s" is its own panel); an
    # empty translated_source just means this pipeline has no translation
    # step, and worker/subtitle_obs.py skips that panel and warns once if a
    # p=s message ever arrives anyway (ADR-0041/0042). Don't "fix" this back
    # to symmetric -- that is the requirement this asymmetry removes.
    if not obs.text_source:
        problems.append(
            ConfigProblem(
                w,
                "OBS バックエンドには subtitle.obs.text_source が必須ですが空です",
                field="subtitle.obs.text_source",
            )
        )
    # OBS は #rrggbb しか受け付けない (hex_color_to_obs_int) が、TK は
    # "white" のような Tk 色名も正当に使える。フィールド自体に pydantic
    # パターンバリデータを付けると動いている TK 設定を壊すので、この検査は
    # worker_type == OBS のここでしか行わない。ADR-0040 は worker_type の
    # 切り替えを「同じイベント、別バックエンド」として売っているため、これは
    # typo ではなく移行経路。
    for name, value in (
        ("subtitle.text.font_color", subtitle.text.font_color),
        ("subtitle.text.outline_color", subtitle.text.outline_color),
        ("subtitle.translated.font_color", subtitle.translated.font_color),
        ("subtitle.translated.outline_color", subtitle.translated.outline_color),
    ):
        try:
            hex_color_to_obs_int(value)
        except ValueError as e:
            problems.append(ConfigProblem(w, f"{name}: {e}", field=name))
    # bg_color だけ TRANSPARENT_BG_COLOR という番兵も正当な値として受け付ける
    # -- lib/obs_text_settings.build_text_settings の扱いをそのまま踏襲する。
    if subtitle.bg_color != TRANSPARENT_BG_COLOR:
        try:
            hex_color_to_obs_int(subtitle.bg_color)
        except ValueError as e:
            problems.append(
                ConfigProblem(w, f"subtitle.bg_color: {e}", field="subtitle.bg_color")
            )
    # 認証の成立とソースの実在は層B (接続してからでないと未起動と区別できない, ADR-0042)。
    return problems


_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_translation,
    _check_tts,
    _check_vc,
    _check_recording,
    _check_playback,
    _check_subtitle,
]


def collect_problems(config: Config) -> list[ConfigProblem]:
    """enable 済み worker の設定不備を集約して返す（送出しない）。

    GUI の起動前 readiness がこれを単一の権威として再利用する (ADR-0045)。
    「何が必須か」の判断をこの module の外に複製しないこと。
    """
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    return problems


def preflight(config: Config) -> None:
    problems = collect_problems(config)
    if problems:
        raise ConfigError(problems)
