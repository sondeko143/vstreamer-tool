import json
import logging
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import IO
from typing import Any
from typing import Literal

import toml
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_serializer
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
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
    # Silero VAD スキップゲート (opt-in)。vc.vad_* とは独立 (ADR-0037)。
    # 音声比率が閾値未満のチャンクを音声認識前に落とす。出力ダックは無し。
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


class StreamVcConfig(BaseModel):
    enable: bool = False
    # 発話系 [vc]/[rvc] とは独立したモデル設定(ADR-0054)。共有素材パスは
    # 各系統へ明示 propagate する方針(ADR-0046)。f0 抽出器だけは RvcConfig の
    # 既定 (rmvpe) を上書きして fcpe にする: streaming はブロックごとに毎回 f0 を
    # 引くので 1 推論あたりの軽い fcpe が向き、実機耳確認もその構成で行った
    # (ADR-0053)。[stream_vc] は既定 disable なので既存挙動は変わらない。
    rvc: RvcConfig = Field(
        default_factory=lambda: RvcConfig(f0_extractor_type=F0ExtractorType.fcpe),
        description="ストリーミング専用の RVC 設定。f0_extractor_type だけ "
        "[rvc] と既定が異なり fcpe(streaming の実機耳確認済み構成)",
    )
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
        description="等電力クロスフェード帯 ms (< block, <= context)。"
        "25ms が実機耳確認済みの値",
    )
    sola_search_ms: float = Field(
        default=5.0,
        ge=0,
        description="SOLA 位相合わせの探索半幅 ms (0 で無効)。実測 ±5ms で十分",
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


class Config(BaseSettings):
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
    listen_port: int = Field(
        default=8080, validation_alias=AliasChoices("listen_port", "PORT")
    )
    template_texts: list[str] = Field(default_factory=lambda: [""])
    text_send_operations: RoutesList = Field(
        default_factory=lambda: [["tts", "playback"]]
    )
    filters: list[ReplaceFilter] = []
    log_file: str = "./voice_%%Y_%%m_%%d.log"
    log_level: int | str = logging.INFO

    model_config = SettingsConfigDict(
        env_prefix="vspeech_",
        env_nested_delimiter="__",
    )

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
