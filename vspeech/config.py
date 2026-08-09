import json
import logging
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import IO
from typing import Any
from typing import Literal

import toml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_serializer
from pydantic import model_validator
from toml.encoder import TomlArraySeparatorEncoder

from vspeech.exceptions import ReplaceFilterParseError

type Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]


class SampleFormat(IntEnum):
    UINT8 = 1
    INT8 = 2
    INT16 = 4
    INT24 = 8
    FLOAT32 = 16
    INVALID = 0


def get_sample_size(format: SampleFormat) -> int:
    if format == SampleFormat.UINT8:
        return 1
    if format == SampleFormat.INT8:
        return 1
    if format == SampleFormat.INT16:
        return 2
    if format == SampleFormat.INT24:
        return 3
    if format == SampleFormat.FLOAT32:
        return 4

    raise ValueError(f"Invalid format: {format}")


class EventType(Enum):
    tts = "tts"
    vc = "vc"
    subtitle = "subtitle"
    transcription = "transcription"
    translation = "translation"
    recording = "recording"
    playback = "playback"
    pause = "pause"
    resume = "resume"
    reload = "reload"
    set_filters = "set_filters"
    ping = "ping"
    forward = "forward"

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls(name)
        except ValueError as e:
            if name in ["sub"]:
                return EventType.subtitle
            if name in ["transc"]:
                return EventType.transcription
            if name in ["transl"]:
                return EventType.translation
            if name in ["rec"]:
                return EventType.recording
            if name in ["play"]:
                return EventType.playback
            if name in ["fwd"]:
                return EventType.forward
            else:
                raise e


class TranscriptionWorkerType(Enum):
    ACP = "ACP"
    GCP = "GCP"
    WHISPER = "WHISPER"


class TtsWorkerType(Enum):
    VR2 = "VR2"
    VOICEVOX = "VOICEVOX"


class SubtitleWorkerType(Enum):
    TK = "TK"
    OBS = "OBS"


class VR2Param(BaseModel):
    volume: float = 1.0
    speed: float = 1.0
    pitch: float = 1.0
    emphasis: float = 1.0
    pause_middle: float = 150
    pause_long: float = 370
    pause_sentence: float = 800
    master_volume: float = 1.0


class VoicevoxParam(BaseModel):
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    intonation_scale: float = 1.0
    volume_scale: float = 1.0
    pre_phoneme_length: float = 0.1
    post_phoneme_length: float = 0.1


class ReplaceFilter(BaseModel):
    pattern: str
    replaced: str

    def __str__(self) -> str:
        return self.pattern + "=" + self.replaced

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_str(value: str) -> ReplaceFilter:
        try:
            pattern, replaced = value.split("=")
        except ValueError as e:
            raise ReplaceFilterParseError(e)
        return ReplaceFilter(pattern=pattern, replaced=replaced)


type RoutesList = list[list[str]]


class RecordingConfig(BaseModel):
    enable: bool = False
    routes_list: RoutesList = Field(default_factory=lambda: [["transcription"]])
    format: SampleFormat = SampleFormat.INT16
    channels: int = Field(default=1, gt=0, description="recording channels")
    rate: int = Field(default=16000, gt=0, description="recording rate")
    chunk: int = Field(default=1024, gt=0, description="recording block size")
    interval_sec: float = Field(default=0.1, description="recording interval sec.")
    silence_threshold: int = Field(
        default=-40,
        description="voice detection volume percentage (approx)",
    )
    max_recording_sec: float = Field(
        default=5,
        description="max wav file length to process",
    )
    gradually_stopping_interval: int = Field(default=3)
    last_interval_frames_buffer_size: int = Field(default=5)
    input_host_api_name: str | None = Field(
        default=None, description="PortAudio host api name to select an input device"
    )
    input_device_name: str | None = Field(
        default=None, description="PortAudio device name to select an input device"
    )
    input_device_index: int | None = Field(
        default=None, description="use this device as recording input if supplied"
    )


class TranscriptionConfig(BaseModel):
    enable: bool = False
    worker_type: TranscriptionWorkerType = TranscriptionWorkerType.GCP
    transliterate_with_mozc: bool = False
    recording_log: bool = False
    recording_log_dir: Path = Path("./rec")
    # Silero VAD skip gate (opt-in). Independent of vc.vad_* (ADR-0037).
    # Drops chunks whose speech ratio is below the threshold before recognition.
    # No output ducking.
    vad_gate: bool = False
    vad_model_file: Path = Field(default=Path())
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_speech_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class TtsConfig(BaseModel):
    enable: bool = False
    worker_type: TtsWorkerType = TtsWorkerType.VR2


class PlaybackConfig(BaseModel):
    enable: bool = False
    volume: int = 100
    output_host_api_name: str | None = Field(
        default=None,
        description="PortAudio host api name to select an output device",
    )
    output_device_name: str | None = Field(
        default=None,
        description="PortAudio device name to select an output device",
    )
    output_device_index: int | None = Field(
        default=None,
        description="use this device as voiceroid2 output if supplied",
    )


class SubtitleTextConfig(BaseModel):
    anchor: Anchor = "center"
    display_sec_per_letter: float = 0.5
    min_display_sec: float = 2.5
    max_text_len: int = 30
    font_size: int = 20
    font_family: str = "Meiryo"
    font_style: str = "bold"
    font_color: str = "#ffffff"
    outline_color: str = "#000000"
    max_histories: int = 10
    delimiter: str = " "
    margin: int = 4


class SubtitleObsConfig(BaseModel):
    url: str = Field(
        default="ws://127.0.0.1:4455",
        description="obs-websocket server URL (Tools -> obs-websocket Settings)",
    )
    password: SecretStr = Field(
        default=SecretStr(""), description="obs-websocket server password"
    )
    text_source: str = Field(
        default="",
        description="name of the OBS Text (GDI+) source that shows transcription",
    )
    translated_source: str = Field(
        default="",
        description="name of the OBS Text (GDI+) source that shows translation",
    )

    @field_serializer("password", when_used="json")
    def serialize_password(self, v: SecretStr) -> str:
        return v.get_secret_value()


class SubtitleConfig(BaseModel):
    enable: bool = False
    worker_type: SubtitleWorkerType = SubtitleWorkerType.TK
    window_width: int = 1600
    window_height: int = 120
    bg_color: str = "#00ff00"
    text: SubtitleTextConfig = Field(
        default_factory=lambda: SubtitleTextConfig(anchor="s")
    )
    translated: SubtitleTextConfig = Field(
        default_factory=lambda: SubtitleTextConfig(anchor="n")
    )
    obs: SubtitleObsConfig = Field(default_factory=SubtitleObsConfig)


class TranslationConfig(BaseModel):
    enable: bool = False
    sec_await_next_text: int = 5
    max_sec_await_total: int = 10
    max_n_chunk_await_total: int = 10


class VcConfig(BaseModel):
    enable: bool = False
    adjust_output_vol_to_input_voice: bool = True
    envelope_strength: float = 1.0
    min_gain: float = 0.1
    max_gain: float = 1.0
    volume_adjust_window_ms: float = 25.0
    vad_gate: bool = False
    vad_model_file: Path = Field(default=Path())
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_speech_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    vad_speech_pad_ms: float = 100.0
    vad_min_gain: float = 0.0


class AmiConfig(BaseModel):
    appkey: SecretStr = Field(
        default=SecretStr(""), description="Amivoice Cloud Platform API APPKEY"
    )
    engine_name: str = Field(
        default="", description="AmiVoice Cloud Platform API engine name"
    )
    engine_uri: str = Field(
        default="",
        description="AmiVoice Cloud Platform API engine uri (sync http only)",
    )
    service_id: str = Field(
        default="", description="AmiVoice Cloud Platform API service id"
    )
    request_timeout: float = 3.0
    extra_parameters: str = "keepFillerToken=1"

    @field_serializer("appkey", when_used="json")
    def serialize_appkey(self, v: SecretStr) -> str:
        return v.get_secret_value()


type ServiceAccountInfo = dict[str, SecretStr]


class GcpConfig(BaseModel):
    service_account_file_path: Path | None = Field(
        default=None,
        description="Google Cloud Platform API credentials file (key.json)",
    )
    service_account_info: ServiceAccountInfo = Field(
        default_factory=dict,
        description="Google Cloud Platform API service account info",
    )
    use_ce_credentials: bool = Field(
        default=False,
        description="Whether to use Compute Engine credentials",
    )
    request_timeout: float = 3.0
    max_retry_count: int = 5
    retry_delay_sec: float = 0.5

    @field_serializer("service_account_info", when_used="json")
    def serialize_service_account_info(self, v: ServiceAccountInfo) -> dict[str, str]:
        return {k: secret.get_secret_value() for k, secret in v.items()}


class Vr2Config(BaseModel):
    params: VR2Param = Field(default_factory=VR2Param)
    voice_name: str | None = None


class WhisperConfig(BaseModel):
    model: str = "large-v3"
    no_speech_prob_threshold: float = 0.6
    logprob_threshold: float = -1.0
    gpu_id: int | None = None
    gpu_name: str = ""


class VoicevoxConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    speaker_id: int = 1
    params: VoicevoxParam = Field(default_factory=VoicevoxParam)
    openjtalk_dir: Path = Path("./voicevox/dict/open_jtalk_dic_utf_8-1.11")
    model_dir: Path = Path("./voicevox/models/vvms")
    onnxruntime_path: Path | None = Field(
        default=None,
        description="voicevox_onnxruntime ライブラリの実パス (onnxruntime-gpu とは別物)",
    )


class RvcQuality(IntEnum):
    zero = 0
    one = 1


class F0ExtractorType(Enum):
    dio = "dio"
    harvest = "harvest"
    rmvpe = "rmvpe"
    fcpe = "fcpe"


class RvcConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_file: Path = Field(default=Path())
    hubert_model_file: Path = Field(
        default=Path(),
        description="scripts/convert_hubert.py が出力した変換済み ContentVec 資産ディレクトリ",
    )
    f0_up_key: int = Field(default=0)
    window: int = Field(default=160)
    quality: RvcQuality = Field(default=RvcQuality.zero)
    gpu_id: int | None = Field(default=None)
    gpu_name: str = Field(default="")
    f0_extractor_type: F0ExtractorType = Field(default=F0ExtractorType.rmvpe)
    input_boost: float = Field(default=1.0)
    rmvpe_model_file: Path = Field(default=Path())
    fcpe_model_file: Path = Field(default=Path())


class TransportType(Enum):
    in_process = "in_process"
    udp = "udp"


class StreamVcRole(Enum):
    local = "local"  # M2: capture+vc+playback in one process (default, unchanged)
    producer = "producer"  # capture + vc + UDP send (GPU host)
    consumer = "consumer"  # UDP recv + jitter buffer + playback (no torch/GPU)


class StreamVcConfig(BaseModel):
    enable: bool = False
    # Model settings independent of the utterance path [vc]/[rvc] (ADR-0054). Shared
    # asset paths are propagated explicitly to each path (ADR-0046). Only the f0
    # extractor overrides RvcConfig's default (rmvpe) with fcpe: streaming pulls f0 on
    # every block, so the lighter per-inference fcpe fits, and the on-hardware ear
    # check was done in that configuration (ADR-0053). [stream_vc] is disabled by
    # default, so existing behaviour does not change.
    #
    # default_factory alone cannot express this override: it only fires when
    # [stream_vc.rvc] is **entirely absent**, while a realistic config containing
    # model_file and friends makes pydantic validate the RvcConfig table, where an
    # omitted f0_extractor_type falls back to RvcConfig's own default (rmvpe). Hence
    # the before-validator below injects fcpe only when "the raw dict has no
    # f0_extractor_type" (an explicit value is honoured, even rmvpe).
    rvc: RvcConfig = Field(
        default_factory=lambda: RvcConfig(f0_extractor_type=F0ExtractorType.fcpe),
        description="ストリーミング専用の RVC 設定。f0_extractor_type を省略すると "
        "[rvc] の既定 (rmvpe) ではなく fcpe になる(streaming の実機耳確認済み構成)。"
        "明示した値は尊重する",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_stream_rvc_f0_to_fcpe(cls, data):
        """Inject fcpe when the [stream_vc.rvc] table has no f0_extractor_type.

        default_factory only fires when the table is missing altogether, so a realistic
        table carrying model_file and friends would let f0 fall back to RvcConfig's
        default rmvpe. Detect "not specified" while the data is still a raw dict and
        supply fcpe (an explicit rmvpe/fcpe is left alone). When rvc is not a dict
        (unspecified -> default_factory, or an RvcConfig instance passed directly),
        leave it untouched.
        """
        if isinstance(data, dict):
            rvc = data.get("rvc")
            if isinstance(rvc, dict) and "f0_extractor_type" not in rvc:
                data = {
                    **data,
                    "rvc": {**rvc, "f0_extractor_type": F0ExtractorType.fcpe.value},
                }
        return data

    @model_validator(mode="after")
    def _check_envelope_gain_bounds(self):
        if self.envelope_min_gain > self.envelope_max_gain:
            raise ValueError(
                f"envelope_min_gain ({self.envelope_min_gain}) must be <= "
                f"envelope_max_gain ({self.envelope_max_gain})"
            )
        return self

    block_ms: float = Field(
        default=160.0,
        gt=0,
        description="固定ブロック(hop)長 ms。160ms が実機耳確認で clean、"
        "80ms は速い代わりに seam のプチプチが残る",
    )
    context_ms: float = Field(
        default=500.0,
        ge=0,
        description="rolling 左文脈 ms。実機耳確認で 500ms 未満はガタつき、"
        "500ms 超にしても改善しない",
    )
    crossfade_ms: float = Field(
        default=25.0,
        ge=0,
        description="クロスフェード帯 ms (< block, <= context)。フェード則は SOLA の "
        "有無で変わる(on=振幅保存/off=等電力)。25ms が実機耳確認済みの値",
    )
    sola_search_ms: float = Field(
        default=5.0,
        ge=0,
        description="SOLA 位相合わせの探索半幅 ms (0 で無効)。実測 ±5ms で十分",
    )
    # Silero VAD noise gate (opt-in). Independent of the utterance path vc.vad_*
    # (ADR-0053). Decided on the input block, applied to the output block (inference
    # is never skipped = context and crossfade stay continuous).
    vad_gate: bool = Field(
        default=False,
        description="ストリーミング経路の VAD ノイズゲート。off だと無音中も"
        "部屋のノイズフロアが変換されて鳴り続ける",
    )
    vad_model_file: Path = Field(
        default=Path(),
        description="silero_vad.onnx (v6.2.1)。[vc] と同じファイルで良い",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="speech と判定する窓確率の閾値。判定も適用も 32ms 窓ごと。"
        "VAD の再帰状態を持ち越すので発話窓はほぼ 1.0 に張り付く = 下げる必要は無い",
    )
    vad_hangover_ms: float = Field(
        default=300.0,
        ge=0,
        description="最後に speech と判定した窓からゲートを開けたまま保つ時間 ms "
        "(後方 dilation のみ)。語間の短い無音でゲートがバタつくのを防ぐ。前方へは "
        "広げない — 語頭直前のブレスを開けてしまうため",
    )
    vad_min_gain: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ゲートが閉じたときの出力ゲイン (0.0 = 完全ミュート)",
    )
    # Input envelope following (opt-in, ADR-0057). Ducks the output volume to follow
    # the input's relative loudness envelope, bringing attack/decay closer to batch
    # conversion. Off by default = bit-identical. The reference is a rolling EMA of the
    # mean input RMS (envelope_ema_ms). Applied before the VAD gate.
    envelope_follow: bool = Field(
        default=False,
        description="出力音量を入力の相対ラウドネス包絡へ追従させる (アタック/"
        "ディケイを滑らかに)。off だと RVC 生出力のままで立ち上がりが急峻",
    )
    # [Open, deferred 2026-08-06] The default stays 1.0 even though the measurement says
    # 0.3 is where the shaping actually happens (min rail 36.6% -> 0%, shaping range
    # 14.7% -> 46.7%). Moving a tuned default wants an ear check to back it, and the
    # comparison on the rig was inconclusive -- 0.3 and 1.0 were not reliably tellable
    # apart. Revisit if a later listening session separates them.
    envelope_strength: float = Field(
        default=1.0,
        ge=0,
        description="包絡形状の指数。0 で無効相当、>1 で追従を強調。ただし 1.0 でも "
        "shape が min/max の窓を外れっぱなしになりやすく、ゲインが上下限へ張り付いて "
        "整形が二値的になる。<1 にすると中間帯が広がって実際に整形が効く"
        "(実測: 0.3 で下限張り付きが消える)。1.0 との聞き分けは耳では未確定",
    )
    envelope_min_gain: float = Field(
        default=0.1, ge=0.0, le=1.0, description="duck の下限ゲイン (静音部の残し量)"
    )
    envelope_max_gain: float = Field(
        default=1.0,
        ge=0.0,
        description="ゲイン上限。既定 1.0 = duck のみ (クリップしない)。>1 は "
        "loud 部を int16 域外へ持ち上げてハードクリップするのでヘッドルームがある時のみ",
    )
    envelope_window_ms: float = Field(
        default=25.0, gt=0, description="入力 RMS のフレーム窓 ms"
    )
    envelope_ema_ms: float = Field(
        default=2000.0,
        gt=0,
        description="参照レベル (入力平均 RMS の rolling EMA) の時定数 ms。"
        "短いと loud onset で参照が跳ねて過敏、長いとレベル変化に鈍い。実測で調整",
    )
    input_host_api_name: str | None = Field(default=None)
    input_device_name: str | None = Field(default=None)
    input_device_index: int | None = Field(default=None)
    output_host_api_name: str | None = Field(default=None)
    output_device_name: str | None = Field(default=None)
    output_device_index: int | None = Field(default=None)
    transport_type: TransportType = Field(default=TransportType.in_process)
    max_queued_blocks: int = Field(
        default=8, gt=0, description="capture/transport の上限。満杯で最古を drop"
    )
    role: StreamVcRole = Field(
        default=StreamVcRole.local,
        description="local=M2 単一プロセス(既定)。producer=capture+vc+送信。"
        "consumer=受信+jitter buffer+再生(GPU/torch 不要)。ADR-0055",
    )
    # producer: where to send. consumer: where to listen. Unused when role=local.
    peer_host: str | None = Field(
        default=None, description="producer の送信先ホスト(consumer の bind と一致)"
    )
    peer_port: int | None = Field(
        default=None, gt=0, le=65535, description="producer の送信先ポート"
    )
    bind_host: str = Field(
        default="0.0.0.0",  # nosec B104 - LAN receiver binds all interfaces by design; override to restrict
        description="consumer の待受ホスト(既定 全 IF)",
    )
    bind_port: int | None = Field(
        default=None, gt=0, le=65535, description="consumer の待受ポート"
    )
    jitter_buffer_ms: float = Field(
        default=0.0,
        ge=0,
        description="consumer のジッタバッファ深さ ms。深さ=付加遅延なので既定は"
        "浅く保ち、実測ジッタから最小に詰める(ADR-0056)。round(jitter_buffer_ms/"
        "block_ms) ブロックを prebuffer してから再生を始める",
    )


class TelemetryConfig(BaseModel):
    enable: bool = True
    max_samples: int = 5000
    log_raw_e2e: bool = True
    skew_warn_threshold: float = 10.0
    skew_hard_ceiling_sec: float = 60.0
    jsonl_path: str = ""


class CustomTomlEncoder(TomlArraySeparatorEncoder):
    def dump_value(self, v: Any) -> str:
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, Enum):
            v = v.value
        return super().dump_value(v)


class Config(BaseModel):
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    playback: PlaybackConfig = Field(default_factory=PlaybackConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    vc: VcConfig = Field(default_factory=VcConfig)
    ami: AmiConfig = Field(default_factory=AmiConfig)
    gcp: GcpConfig = Field(default_factory=GcpConfig)
    vr2: Vr2Config = Field(default_factory=Vr2Config)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    voicevox: VoicevoxConfig = Field(default_factory=VoicevoxConfig)
    rvc: RvcConfig = Field(default_factory=RvcConfig)
    stream_vc: StreamVcConfig = Field(default_factory=StreamVcConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    listen_address: str = "[::]"
    listen_port: int = 8080
    filters: list[ReplaceFilter] = []
    log_file: str = "./voice_%%Y_%%m_%%d.log"
    log_level: int | str = logging.INFO

    # `extra="forbid"` was inherited implicitly from SettingsConfigDict before
    # ADR-0066. Restate it here or a typo'd key in config.toml starts being
    # swallowed: plain BaseModel defaults to extra="ignore".
    #
    # [Open, deferred 2026-08-09] This only guards the top level. The nested
    # section models (RecordingConfig and friends) are plain BaseModel, so
    # `[recording] enabel = true` is still ignored silently. Tightening them
    # was left out of ADR-0066 to avoid stacking a second breaking change onto
    # the top-level one, not because the stricter behaviour is undesirable — it
    # is the same fail-loud the top level already gets.
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def is_file_json(file_path: str | Path):
        file_name = str(file_path)
        return file_name.endswith(".json")

    @staticmethod
    def read_config_from_file(file: IO[bytes]):
        file_name = file.name
        if Config.is_file_json(file_name):
            config_obj = json.loads(file.read())
        else:
            config_obj = toml.loads(file.read().decode("utf-8"))
        return Config.model_validate(config_obj)

    def export_to_toml(self):
        encoded = self.model_dump()
        # Every SecretStr field in Config must be hand-unwrapped below, or its
        # raw value leaks as a masked/repr string; guarded by
        # tests/test_config_secret.py::test_every_secret_str_field_survives_export_to_toml
        conf_dict = {
            **encoded,
            "ami": {**encoded["ami"], "appkey": self.ami.appkey.get_secret_value()},
            "gcp": {
                **encoded["gcp"],
                "service_account_info": {
                    k: v.get_secret_value()
                    for k, v in self.gcp.service_account_info.items()
                },
            },
            "subtitle": {
                **encoded["subtitle"],
                "obs": {
                    **encoded["subtitle"]["obs"],
                    "password": self.subtitle.obs.password.get_secret_value(),
                },
            },
        }
        return toml.dumps(conf_dict, encoder=CustomTomlEncoder(dict, separator="\n"))

    def get_nested_value(self, name: str):
        *attributes, child = name.split(".")
        nest = self
        for attribute in attributes:
            nest = getattr(nest, attribute)
        return getattr(nest, child)
