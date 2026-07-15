"""起動時の設定 preflight (層A, ADR-0038)。

enable した各 worker の「実リソースを取得せずに安価に判定できる」設定不備
(必須フィールド・参照ファイル/ディレクトリの存在・デバイス発見可否・依存の
有無) を検査する。検出した全問題を集約して ConfigError で送出する。実ロードで
しか分からない失敗は worker 起動時に扱う (層B)。
"""

from collections.abc import Callable
from importlib.util import find_spec

from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.config import VcConfig
from vspeech.exceptions import ConfigError
from vspeech.exceptions import ConfigProblem

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
                    ConfigProblem(w, f"ACP バックエンドには {name} が必須ですが空です")
                )
        if tc.transliterate_with_mozc and find_spec("mozcpy") is None:
            problems.append(
                ConfigProblem(
                    w,
                    "transliterate_with_mozc=true ですが mozcpy が未インストールです",
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
        problems.append(ConfigProblem(w, str(e)))
    try:
        WorkerOutput.from_routes_list(config.recording.routes_list)
    except Exception as e:
        problems.append(ConfigProblem(w, f"recording.routes_list が不正です: {e}"))
    return problems


def _check_playback(config: Config) -> list[ConfigProblem]:
    if not config.playback.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_output_device

    try:
        resolve_output_device(config.playback)
    except DeviceNotFoundError as e:
        return [ConfigProblem("playback", str(e))]
    return []


_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_recording,
    _check_playback,
]


def preflight(config: Config) -> None:
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    if problems:
        raise ConfigError(problems)
