import logging
from enum import Enum
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import pyaudio
from pydantic import BaseModel
from pydantic import Field
from pydantic_cli import DefaultConfig


class TranscriptionWorkerType(Enum):
    ACP = "ACP"
    GCP = "GCP"
    WHISPER = "WHISPER"


class SpeechWorkerType(Enum):
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


class Config(BaseModel):
    class Config(DefaultConfig):
        CLI_JSON_ENABLE = True
        CLI_JSON_CONFIG_PATH = "./config.json"
        CLI_JSON_VALIDATE_PATH = False

    format: int = pyaudio.paInt16  # data type formate
    channels: int = Field(1, cli=("-c", "--channels"), description="recording channels")
    rate: int = Field(16000, description="recording rate")
    chunk: int = Field(1024, description="recording block size")
    record_interval_sec: float = Field(
        0.5, cli=("-s", "--interval_sec"), description="recording interval sec."
    )
    silence_threshold: float = Field(
        10,
        cli=("-t", "--silence_threshold"),
        description="voice detection volume percentage (approx)",
    )
    max_recording_sec: float = Field(
        5,
        cli=("-m", "--max_recording_sec"),
        description="max wav file length to process",
    )
    transcription_worker_type: TranscriptionWorkerType = TranscriptionWorkerType.ACP
    ami_appkey: str = Field("", description="Amivoice Cloud Platform API APPKEY")
    ami_engine_name: str = Field(
        "", description="AmiVoice Cloud Platform API engine name"
    )
    ami_engine_uri: str = Field(
        "", description="AmiVoice Cloud Platform API engine uri (sync http only)"
    )
    ami_service_id: str = Field(
        "", description="AmiVoice Cloud Platform API service id"
    )
    ami_request_timeout: float = 3.0
    gcp_project_id: str = Field("", description="Google cloud platform project ID")
    gcp_credentials_file_path: str = Field(
        "", description="Google Cloud Platform API credentials file (key.json)"
    )
    gcp_request_timeout: float = 3.0
    min_subtitle_display_sec: float = 2.5
    max_subtitle_text_len: int = 30
    max_subtitle_translated_len: int = 50
    subtitle_font_size: int = 20
    subtitle_font_family: str = "Meiryo"
    subtitle_font_style: str = "bold"
    input_host_api_name: Optional[str] = Field(
        None, description="PortAudio host api name to select an input device"
    )
    input_device_name: Optional[str] = Field(
        None, description="PortAudio device name to select an input device"
    )
    input_device_index: Optional[int] = Field(
        None, description="use this device as recording input if supplied"
    )
    output_host_api_name: Optional[str] = Field(
        None, description="PortAudio host api name to select an output device"
    )
    output_device_name: Optional[str] = Field(
        None, description="PortAudio device name to select an output device"
    )
    output_device_index: Optional[int] = Field(
        None, description="use this device as voiceroid2 output if supplied"
    )
    speech_worker_type: SpeechWorkerType = SpeechWorkerType.VR2
    vr2_params: VR2Param = Field(default_factory=VR2Param)
    vr2_voice_name: Optional[str] = None
    whisper_model: str = "base"
    voicevox_core_dir: Path = Path("./voicevox_core")
    voicevox_speaker_id: int = 1
    openjtalk_dir: Path = Path("./voicevox_core/open_jtalk_dic_utf_8-1.11")
    enable_translation: bool = True
    port: int = 19827
    template_texts: List[str] = Field(default_factory=lambda: [""])
    log_file: str = "./voice_%%Y_%%m_%%d.log"
    log_level: Union[int, str] = logging.INFO
