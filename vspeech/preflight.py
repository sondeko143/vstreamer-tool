"""Startup config preflight (layer A, ADR-0038).

For every enabled worker, check the config problems that can be decided cheaply
without acquiring a real resource (required fields, existence of referenced
files/directories, device discoverability, presence of dependencies). All detected
problems are aggregated and raised as a ConfigError. Failures that only surface on a
real load are handled at worker startup (layer B).

One check is a deliberate exception to "without acquiring a real resource":
`_check_device_rate` actually opens (and immediately closes) the audio device at its
resolved rate, because whether PortAudio accepts that rate can only be known by trying
(ADR-0076). It is the one place in this module that touches real hardware.
"""

from collections.abc import Callable
from importlib.util import find_spec
from json import JSONDecodeError
from pathlib import Path
from typing import IO
from typing import TYPE_CHECKING

from pydantic import ValidationError
from toml import TomlDecodeError

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
from vspeech.logger import logger

if TYPE_CHECKING:
    # Type-only: importing vspeech.lib.audio for real would pull in sounddevice at
    # module load, making preflight.py (imported unconditionally by main.py) require
    # the `audio` extra even for a transcription-only pipeline. Every runtime use below
    # imports it lazily, inside the enable-gated checker that actually needs it.
    from vspeech.lib.audio import DeviceInfo

Checker = Callable[[Config], list[ConfigProblem]]

# The standard audio rate family ([0075](docs/adr/0075-wire-sample-rate-validation.md)):
# 8000 Hz (telephony) through 192000 Hz (hi-res), stepped by 25 Hz -- the family's own
# gcd. ADR-0075 measured that every pair drawn from this family builds at most 261k taps
# (84ms), well under MAX_PROTOTYPE_TAPS, so testing a resolved device rate against every
# member here cannot reject a legitimate combination. Used as the proxy pipeline rate for
# stream_vc's output boundary (ADR-0076): unlike recording/stream_vc-input, whose
# pipeline rate is a config constant, stream_vc output's target_sample_rate comes from
# the RVC model's own metadata, which preflight cannot cheaply read (that would mean
# building a GPU session here). The RVC model rates this boundary actually carries
# (32000/40000/48000) all sit inside this family.
_STANDARD_SAMPLE_RATES = (
    8000,
    11025,
    16000,
    22050,
    24000,
    32000,
    40000,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
)


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


def _check_device_rate(
    *,
    device: DeviceInfo,
    override: int | None,
    input: bool,
    config_key: str,
    worker: str,
    channels: int,
    dtype: str,
    ratio_targets: tuple[int, ...] = (),
) -> list[ConfigProblem]:
    """Verify the rate `device` would be opened at (ADR-0076). Shared by all four device
    boundaries ([recording]/[playback]/[stream_vc] in/out); only `resolve_input_device` /
    `resolve_output_device` / `resolve_stream_vc_*_device` differ per boundary, and those
    already ran by the time this is called (device resolution and rate resolution are
    two separate steps, each with its own try/except, so DeviceRateUnresolvedError being
    a DeviceNotFoundError subclass cannot be caught by the wrong handler here).

    Three independent things can be wrong once a device is found, checked in order,
    each cheaper than the next, so a failure at one stops before paying for the next:

    1. The rate cannot be decided at all (DeviceRateUnresolvedError, ADR-0074).
    2. The decided rate produces a pathological resample ratio against `ratio_targets`
       (empty for a boundary whose counterpart rate is not known at preflight time --
       see worker/playback.py's per-utterance warning instead, ADR-0075). Pure
       arithmetic, no hardware touched -- a device already known to be unusable for
       this boundary is not worth opening just to report the same conclusion twice.
    3. PortAudio refuses to open the device at that rate at all -- can only be found by
       trying, so this is the one preflight check that acquires a real device: opened,
       started and immediately closed, never read from or written to. The stream is
       always closed, success or failure -- sounddevice's stream objects have no
       `__del__` and `close()` is the only path to `Pa_CloseStream`, so a `start()`
       failure that skipped `close()` would leak the native handle for the rest of the
       process (this bit preflight itself in review; the shared `open_device_stream`
       had the same gap and is fixed alongside this, lib/audio.py).
    """
    from vspeech.exceptions import DeviceRateUnresolvedError
    from vspeech.lib.audio import resolve_device_rate
    from vspeech.lib.resample import make_resampler

    try:
        rate, _how = resolve_device_rate(
            device, override, input=input, config_key=config_key
        )
    except DeviceRateUnresolvedError as e:
        return [ConfigProblem(worker, str(e), field=config_key)]

    for target in ratio_targets:
        try:
            make_resampler(rate, target)
        except ValueError as e:
            return [
                ConfigProblem(
                    worker,
                    f"{config_key} は {rate}Hz に解決されましたが、"
                    f"{target}Hz への変換比が病的です: {e}",
                    field=config_key,
                )
            ]  # one pathological target already condemns the device rate; the loop
            # need not (and, per the docstring above, must not) touch hardware too.

    import sounddevice as sd

    stream = None
    try:
        stream = (
            sd.RawInputStream(
                samplerate=rate, device=device.index, channels=channels, dtype=dtype
            )
            if input
            else sd.RawOutputStream(
                samplerate=rate, device=device.index, channels=channels, dtype=dtype
            )
        )
        stream.start()
    except (OSError, sd.PortAudioError) as e:
        kind = "入力" if input else "出力"
        return [
            ConfigProblem(
                worker,
                f"{kind}デバイス '{device.name}' を {rate}Hz "
                f"(channels={channels}, dtype={dtype}) で開けません: {e}",
                field=config_key,
            )
        ]
    else:
        # Parity with the worker's own open path (lib/audio.open_device_stream step 4):
        # a rate PortAudio silently substitutes is not a config problem by itself (the
        # worker still converts at the requested rate), so this stays a log line, not a
        # ConfigProblem -- just visible at preflight time instead of only at real
        # startup.
        reported = float(stream.samplerate)
        if abs(reported - rate) > 0.5:
            logger.warning(
                "%s デバイスが要求した %dHz とは異なる %.4fHz を報告しています; "
                "変換は要求したレートのまま行います",
                config_key,
                rate,
                reported,
            )
    finally:
        if stream is not None:
            stream.close()
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
    from vspeech.lib.audio import get_sd_dtype
    from vspeech.lib.audio import resolve_input_device
    from vspeech.shared_context import WorkerOutput

    w = "recording"
    rec = config.recording
    problems: list[ConfigProblem] = []
    try:
        device = resolve_input_device(rec)
    except DeviceNotFoundError as e:
        problems.append(ConfigProblem(w, str(e), field="recording.input_device_index"))
    else:
        # config.rate is the one fixed pipeline rate this boundary ever converts to
        # (ADR-0076), so the ratio check tests it exactly instead of a proxy set.
        problems.extend(
            _check_device_rate(
                device=device,
                override=rec.input_device_rate,
                input=True,
                config_key="recording.input_device_rate",
                worker=w,
                channels=rec.channels,
                dtype=get_sd_dtype(rec.format),
                ratio_targets=(rec.rate,),
            )
        )
    try:
        WorkerOutput.from_routes_list(rec.routes_list)
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

    w = "playback"
    pb = config.playback
    try:
        device = resolve_output_device(pb)
    except DeviceNotFoundError as e:
        return [ConfigProblem(w, str(e), field="playback.output_device_index")]
    # No ratio_targets: the source rate (whatever TTS/VC/remote worker produced the
    # utterance) is not known until a real utterance arrives, so there is no fixed
    # pipeline rate to test against here -- worker/playback.py already warns and moves
    # on to the next utterance for a pathological one (ADR-0075/0076). This is still the
    # first time the device itself (rate resolution + actually opening it) can be
    # validated at startup at all -- before ADR-0073/0074 fixed the device to its own
    # native rate, this boundary's rate was only known once the first utterance arrived.
    return _check_device_rate(
        device=device,
        override=pb.output_device_rate,
        input=False,
        config_key="playback.output_device_rate",
        worker=w,
        channels=1,
        dtype="int16",
    )


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
    from vspeech.stream_vc.capture import CAPTURE_RATE
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
            input_device = resolve_stream_vc_input_device(sv)
        except DeviceNotFoundError as e:
            problems.append(
                ConfigProblem(w, str(e), field="stream_vc.input_device_index")
            )
        else:
            # CAPTURE_RATE is the one fixed pipeline rate this boundary ever converts
            # to (ADR-0076), so the ratio check tests it exactly instead of a proxy set.
            problems.extend(
                _check_device_rate(
                    device=input_device,
                    override=sv.input_device_rate,
                    input=True,
                    config_key="stream_vc.input_device_rate",
                    worker=w,
                    channels=1,
                    dtype="int16",
                    ratio_targets=(CAPTURE_RATE,),
                )
            )
    if does_play:
        try:
            output_device = resolve_stream_vc_output_device(sv)
        except DeviceNotFoundError as e:
            problems.append(
                ConfigProblem(w, str(e), field="stream_vc.output_device_index")
            )
        else:
            # The counterpart rate here is whatever RVC model is loaded (its
            # target_sample_rate, read from ONNX metadata at worker startup, layer B --
            # preflight cannot cheaply know it without building a GPU session). Test
            # against the whole standard rate family as a proxy instead (ADR-0076).
            problems.extend(
                _check_device_rate(
                    device=output_device,
                    override=sv.output_device_rate,
                    input=False,
                    config_key="stream_vc.output_device_rate",
                    worker=w,
                    channels=1,
                    dtype="int16",
                    ratio_targets=_STANDARD_SAMPLE_RATES,
                )
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
    this module -- call it.
    """
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    return problems


def preflight(config: Config) -> None:
    problems = collect_problems(config)
    if problems:
        raise ConfigError(problems)


def _dotted(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc)


def load_config(file: IO[bytes]) -> Config:
    """Read the --config file, reporting a malformed one as ConfigError (ADR-0068).

    The file has to be parsed and validated before `preflight()` can be run on it,
    so these failures escaped the aggregation above and reached the user as a raw
    pydantic/TOML traceback. ADR-0038 makes this module the one place config problems
    surface; that is only true if the ones found while reading arrive the same way.

    A pydantic ValidationError carries one entry per offending setting, so it maps
    onto ConfigProblem one-to-one. A decode error is a single problem about the file
    as a whole, and no setting can be named for it.
    """
    try:
        return Config.read_config_from_file(file)
    except ValidationError as e:
        raise ConfigError(
            [
                ConfigProblem(
                    "config",
                    f"{_dotted(err['loc']) or '(トップレベル)'}: {err['msg']}",
                    field=_dotted(err["loc"]) or None,
                )
                for err in e.errors()
            ]
        ) from e
    except (TomlDecodeError, JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(
            [
                ConfigProblem(
                    "config", f"設定ファイルとして読めません ({file.name}): {e}"
                )
            ]
        ) from e
