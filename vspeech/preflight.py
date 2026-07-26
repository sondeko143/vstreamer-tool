"""Startup config preflight (layer A, ADR-0038).

For every enabled worker, check the config problems that can be decided cheaply
without acquiring a real resource (required fields, existence of referenced
files/directories, device discoverability, presence of dependencies). All detected
problems are aggregated and raised as a ConfigError. Failures that only surface on a
real load are handled at worker startup (layer B).
"""

from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from vspeech.config import Config
from vspeech.config import F0ExtractorType
from vspeech.config import GcpConfig
from vspeech.config import RvcConfig
from vspeech.config import StreamVcConfig
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
    # Whether auth actually succeeds is layer B. Here we only cheaply check that the
    # configured key.json exists.
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
    cfg: TranscriptionConfig | VcConfig | StreamVcConfig, worker: str
) -> list[ConfigProblem]:
    """Require vad_model_file to exist when vad_gate=true.

    Only `.vad_gate` / `.vad_model_file` are touched, so any config carrying those two
    works as-is ([transcription] / [vc] / [stream_vc] share this).
    """
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
    # WHISPER's model/GPU load is layer B (acquired at startup).
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
        return []  # VR2's real initialization is layer B
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


def _check_rvc_assets(
    rvc: RvcConfig, worker: str, field_prefix: str
) -> list[ConfigProblem]:
    """Existence check for the RVC model assets (model/HuBERT/f0). Shared by [vc] and
    [stream_vc]."""
    problems: list[ConfigProblem] = []
    if not rvc.model_file.expanduser().is_file():
        problems.append(
            ConfigProblem(
                worker,
                f"{field_prefix}.model_file '{rvc.model_file}' が存在しません",
                field=f"{field_prefix}.model_file",
            )
        )
    hubert = rvc.hubert_model_file
    if hubert == Path() or not hubert.expanduser().is_dir():
        problems.append(
            ConfigProblem(
                worker,
                f"{field_prefix}.hubert_model_file '{hubert}' (資産ディレクトリ) が存在しません",
                field=f"{field_prefix}.hubert_model_file",
            )
        )
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        if not rvc.rmvpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    worker,
                    f"{field_prefix}.rmvpe_model_file '{rvc.rmvpe_model_file}' が存在しません",
                    field=f"{field_prefix}.rmvpe_model_file",
                )
            )
    if rvc.f0_extractor_type == F0ExtractorType.fcpe:
        if not rvc.fcpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    worker,
                    f"{field_prefix}.fcpe_model_file '{rvc.fcpe_model_file}' が存在しません",
                    field=f"{field_prefix}.fcpe_model_file",
                )
            )
    return problems


def _check_vc(config: Config) -> list[ConfigProblem]:
    if not config.vc.enable:
        return []
    w = "vc"
    problems = _check_rvc_assets(config.rvc, w, "rvc")
    problems.extend(_check_vad_gate(config.vc, w))
    return problems


def _check_subtitle(config: Config) -> list[ConfigProblem]:
    if not config.subtitle.enable:
        return []
    if config.subtitle.worker_type != SubtitleWorkerType.OBS:
        return []  # TK has no endpoint to connect to
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
    # OBS accepts only #rrggbb (hex_color_to_obs_int), whereas TK legitimately accepts
    # Tk color names such as "white". Attaching a pydantic pattern validator to the
    # field itself would break working TK configs, so this check lives here, under
    # worker_type == OBS, and nowhere else. ADR-0040 sells switching worker_type as
    # "same event, different backend", so this is a migration path, not a typo.
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
    # bg_color alone also accepts the TRANSPARENT_BG_COLOR sentinel as a valid value
    # -- this mirrors how lib/obs_text_settings.build_text_settings treats it.
    if subtitle.bg_color != TRANSPARENT_BG_COLOR:
        try:
            hex_color_to_obs_int(subtitle.bg_color)
        except ValueError as e:
            problems.append(
                ConfigProblem(w, f"subtitle.bg_color: {e}", field="subtitle.bg_color")
            )
    # Auth success and source existence are layer B (until we connect they cannot be
    # told apart from "OBS is not running yet", ADR-0042).
    return problems


def _check_stream_vc(config: Config) -> list[ConfigProblem]:
    if not config.stream_vc.enable:
        return []
    from vspeech.config import StreamVcRole
    from vspeech.config import TransportType
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_stream_vc_input_device
    from vspeech.lib.audio import resolve_stream_vc_output_device
    from vspeech.stream_vc.capture import ms_to_samples

    w = "stream_vc"
    sv = config.stream_vc
    role = sv.role
    # local (M2, single process) still does all of capture+vc+playback. producer and
    # consumer (M3, split across two machines, ADR-0055) each own only half the job, so
    # demanding the assets/devices of the half they do not own would force an
    # unsatisfiable config on a consumer machine with no GPU.
    does_vc = role in (StreamVcRole.local, StreamVcRole.producer)
    does_play = role in (StreamVcRole.local, StreamVcRole.consumer)
    problems: list[ConfigProblem] = []

    if does_vc:
        problems += _check_rvc_assets(sv.rvc, w, "stream_vc.rvc")
        # StreamingVc's guard decides on the lengths after the ms->sample rounding.
        # Compare in that same sample domain here so there is no boundary where
        # sub-ms rounding passes preflight but __init__ raises ValueError.
        cf = ms_to_samples(sv.crossfade_ms)
        blk = ms_to_samples(sv.block_ms)
        ctx = ms_to_samples(sv.context_ms)
        if cf >= blk:
            problems.append(
                ConfigProblem(
                    w,
                    f"crossfade_ms ({sv.crossfade_ms}) は block_ms ({sv.block_ms}) 未満が必須です",
                    field="stream_vc.crossfade_ms",
                )
            )
        if cf > ctx:
            problems.append(
                ConfigProblem(
                    w,
                    f"crossfade_ms ({sv.crossfade_ms}) は context_ms ({sv.context_ms}) 以下が必須です",
                    field="stream_vc.crossfade_ms",
                )
            )
        # field becomes "stream_vc.vad_model_file" (the worker name is the prefix).
        problems.extend(_check_vad_gate(sv, w))
        try:
            resolve_stream_vc_input_device(sv)
        except DeviceNotFoundError as e:
            problems.append(
                ConfigProblem(w, str(e), field="stream_vc.input_device_index")
            )
    if does_play:
        try:
            resolve_stream_vc_output_device(sv)
        except DeviceNotFoundError as e:
            problems.append(
                ConfigProblem(w, str(e), field="stream_vc.output_device_index")
            )

    # role != local needs a network transport. Left on in_process nobody receives what
    # vc sends and every block is dropped silently (a silent misconfig), so reject it
    # fail-loud.
    if role is not StreamVcRole.local and sv.transport_type is not TransportType.udp:
        problems.append(
            ConfigProblem(
                w,
                "role=producer/consumer は transport_type=udp が必須です",
                field="stream_vc.transport_type",
            )
        )
    # The other direction: role=local is self-contained on one machine (in_process), so
    # a udp setting is ignored. Point it out fail-loud instead of discarding it
    # silently (catches a mistyped two-machine split early).
    if role is StreamVcRole.local and sv.transport_type is TransportType.udp:
        problems.append(
            ConfigProblem(
                w,
                "role=local は単一マシン (in_process) で動作し transport_type=udp は無視されます。"
                "2 マシンに分けるなら role を producer/consumer にしてください",
                field="stream_vc.transport_type",
            )
        )
    # With UDP each role needs an address. in_process (local) needs none.
    if sv.transport_type is TransportType.udp:
        if role is StreamVcRole.producer and not (sv.peer_host and sv.peer_port):
            problems.append(
                ConfigProblem(
                    w,
                    "role=producer は peer_host/peer_port(送信先)が必須です",
                    field="stream_vc.peer_port",
                )
            )
        if role is StreamVcRole.consumer and not sv.bind_port:
            problems.append(
                ConfigProblem(
                    w,
                    "role=consumer は bind_port(待受ポート)が必須です",
                    field="stream_vc.bind_port",
                )
            )
    return problems


_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_translation,
    _check_tts,
    _check_vc,
    _check_recording,
    _check_playback,
    _check_subtitle,
    _check_stream_vc,
]


def collect_problems(config: Config) -> list[ConfigProblem]:
    """Aggregate and return the config problems of the enabled workers (never raises).

    This module is the single authority on "what is required", and the startup
    fail-loud (ADR-0038) is its only reader. Do not duplicate that judgement outside
    this module -- the GUI once reused it as a pre-launch readiness check precisely so
    it could call instead of duplicate (ADR-0045; removed along with the GUI in
    ADR-0061).
    """
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    return problems


def preflight(config: Config) -> None:
    problems = collect_problems(config)
    if problems:
        raise ConfigError(problems)
