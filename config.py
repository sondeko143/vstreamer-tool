import pyaudio
from pydantic import BaseModel


class Config(BaseModel):
    format: int = pyaudio.paInt16  # data type formate
    channels: int = 2  # Adjust to your number of channels
    rate: int = 44100  # Sample Rate
    chunk: int = 1024  # Block Size
    record_seconds: float = 0.5  # Record chunk
    silence_threshold: float = 0.01  # For silence detection
    max_recording_seconds: float = 10
    log_file: str = "./voice_%Y_%m_%d.log"
    ami_appkey: str
    ami_engine_name: str
    ami_engine_uri: str
    yukari_net_port: int
