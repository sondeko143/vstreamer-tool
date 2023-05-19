import json
import logging
from enum import Enum
from enum import IntEnum
from pathlib import Path
from typing import IO
from typing import Any
from typing import Optional
from typing import TypeAlias
from typing import Union

import toml
from pydantic import BaseModel
from pydantic import BaseSettings
from pydantic import Field
from pydantic import SecretStr
from toml.encoder import TomlArraySeparatorEncoder

from vspeech.exceptions import ReplaceFilterParseError


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
    subtitle_translated = "subtitle_translated"
    transcription = "transcription"
    translation = "translation"
    recording = "recording"
    playback = "playback"
    pause = "pause"
    resume = "resume"
    reload = "reload"
    set_filters = "set_filters"
    ping = "ping"

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls(name)
        except ValueError as e:
            if name in ["sub"]:
                return EventType.subtitle
            if name in ["sub_t"]:
                return EventType.subtitle_translated
            if name in ["transc"]:
                return EventType.transcription
            if name in ["transl"]:
                return EventType.translation
            if name in ["rec"]:
                return EventType.recording
            if name in ["play"]:
                return EventType.playback
            else:
                raise e


class TranscriptionWorkerType(Enum):
    ACP = "ACP"
    GCP = "GCP"
    WHISPER = "WHISPER"


class TtsWorkerType(Enum):
    VR2 = "VR2"
    VOICEVOX = "VOICEVOX"


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
    def from_str(value: str) -> "ReplaceFilter":
        try:
            pattern, replaced = value.split("=")
        except ValueError as e:
            raise ReplaceFilterParseError(e)
        return ReplaceFilter(pattern=pattern, replaced=replaced)


RoutesList: TypeAlias = list[list[str]]


class RecordingConfig(BaseModel):
    enable: bool = True
    routes_list: RoutesList = Field(default_factory=lambda: [["transcription"]])
    format: SampleFormat = SampleFormat.INT16
    channels: int = Field(
        default=1, cli=("-c", "--channels"), description="recording channels"
    )
    rate: int = Field(default=16000, description="recording rate")
    chunk: int = Field(default=1024, description="recording block size")
    interval_sec: float = Field(
        default=0.5, cli=("-s", "--interval_sec"), description="recording interval sec."
    )
    silence_threshold: int = Field(
        default=-40,
        cli=("-t", "--silence_threshold"),
        description="voice detection volume percentage (approx)",
    )
    max_recording_sec: float = Field(
        default=5,
        cli=("-m", "--max_recording_sec"),
        description="max wav file length to process",
    )
    gradually_stopping_interval: int = Field(default=3)
    input_host_api_name: Optional[str] = Field(
        default=None, description="PortAudio host api name to select an input device"
    )
    input_device_name: Optional[str] = Field(
        default=None, description="PortAudio device name to select an input device"
    )
    input_device_index: Optional[int] = Field(
        default=None, description="use this device as recording input if supplied"
    )


class TranscriptionConfig(BaseModel):
    enable: bool = True
    worker_type: TranscriptionWorkerType = TranscriptionWorkerType.ACP
    transliterate_with_mozc: bool = False
    recording_log: bool = False
    recording_log_dir: Path = Path("./rec")


class TtsConfig(BaseModel):
    enable: bool = True
    worker_type: TtsWorkerType = TtsWorkerType.VR2


class PlaybackConfig(BaseModel):
    enable: bool = True
    volume: int = 100
    output_host_api_name: Optional[str] = Field(
        default=None,
        description="PortAudio host api name to select an output device",
    )
    output_device_name: Optional[str] = Field(
        default=None,
        description="PortAudio device name to select an output device",
    )
    output_device_index: Optional[int] = Field(
        default=None,
        description="use this device as voiceroid2 output if supplied",
    )


class SubtitleTextConfig(BaseModel):
    display_sec_per_letter: float = 0.5
    min_display_sec: float = 2.5
    max_text_len: int = 30
    font_size: int = 20
    font_family: str = "Meiryo"
    font_style: str = "bold"
    font_color: str = "#ffffff"
    outline_color: str = "#000000"


class SubtitleConfig(BaseModel):
    enable: bool = True
    window_width: int = 1600
    window_height: int = 120
    bg_color: str = "#00ff00"
    text: SubtitleTextConfig = Field(default_factory=SubtitleTextConfig)
    translated: SubtitleTextConfig = Field(default_factory=SubtitleTextConfig)


class TranslationConfig(BaseModel):
    enable: bool = True
    source_language_code: Optional[str] = "ja"
    target_language_code: str = "en"


class VcConfig(BaseModel):
    enable: bool = True


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


ServiceAccountInfo: TypeAlias = dict[str, SecretStr]


class GcpConfig(BaseModel):
    service_account_file_path: Path = Field(
        default="", description="Google Cloud Platform API credentials file (key.json)"
    )
    service_account_info: ServiceAccountInfo = Field(
        default_factory=dict,
        description="Google Cloud Platform API service account info",
    )
    request_timeout: float = 3.0
    max_retry_count: int = 5
    retry_delay_sec: float = 0.5


class Vr2Config(BaseModel):
    params: VR2Param = Field(default_factory=VR2Param)
    voice_name: Optional[str] = None


class WhisperConfig(BaseModel):
    model: str = "base"
    no_speech_prob_threshold: float = 0.6
    logprob_threshold: float = -1.0


class VoicevoxConfig(BaseModel):
    speaker_id: int = 1
    params: VoicevoxParam = Field(default_factory=VoicevoxParam)
    openjtalk_dir: Path = Path("./voicevox_core/open_jtalk_dic_utf_8-1.11")


class RvcQuality(IntEnum):
    zero = 0
    one = 1


class F0ExtractorType(Enum):
    dio = "dio"
    harvest = "harvest"


class RvcConfig(BaseModel):
    model_file: Path = Field(default="")
    hubert_model_file: Path = Field(default="")
    f0_up_key: int = Field(default=0)
    window: int = Field(default=160)
    quality: RvcQuality = Field(default=RvcQuality.zero)
    gpu_id: int = Field(default=0)
    f0_extractor_type: F0ExtractorType = Field(default=F0ExtractorType.harvest)


class CustomTomlEncoder(TomlArraySeparatorEncoder):  # type: ignore
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

    listen_address: str = "[::]"
    listen_port: int = Field(default=8080, env="PORT")
    template_texts: list[str] = Field(default_factory=lambda: [""])
    text_send_operations: list[list[str]] = Field(
        default_factory=lambda: [["tts", "playback"]]
    )
    filters: list[ReplaceFilter] = []
    log_file: str = "./voice_%%Y_%%m_%%d.log"
    log_level: Union[int, str] = logging.INFO

    class Config(BaseSettings.Config):
        env_prefix = "vspeech_"
        env_nested_delimiter = "__"

        @staticmethod
        def encode_secret(v: SecretStr):
            return v.get_secret_value()

        json_encoders = {
            SecretStr: encode_secret,
        }

    @staticmethod
    def is_file_json(file_path: Union[str, Path]):
        file_name = str(file_path)
        return file_name.endswith(".json")

    @staticmethod
    def read_config_from_file(file: IO[bytes]):
        file_name = file.name
        if Config.is_file_json(file_name):
            config_obj = json.loads(file.read())
        else:
            config_obj = toml.loads(file.read().decode("utf-8"))
        return Config.parse_obj(config_obj)

    def export_to_toml(self):
        encoded = self.dict()
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
        }
        return toml.dumps(conf_dict, encoder=CustomTomlEncoder(dict, separator="\n"))
