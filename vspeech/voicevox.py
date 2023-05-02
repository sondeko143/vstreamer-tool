from pathlib import Path

from voicevox_core import AccelerationMode
from voicevox_core import VoicevoxCore
from vspeech.config import VoicevoxParam


class Voicevox:
    core: VoicevoxCore

    def __init__(self, open_jtalk_dict_dir: Path) -> None:
        self.core = VoicevoxCore(
            acceleration_mode=AccelerationMode.AUTO,
            open_jtalk_dict_dir=open_jtalk_dict_dir.expanduser(),
        )

    # ラッパー関数
    def load_model(self, speaker_id: int):
        self.core.load_model(speaker_id)

    def is_model_loaded(self, speaker_id: int) -> bool:
        return self.core.is_model_loaded(speaker_id)

    def voicevox_tts(self, text: str, speaker_id: int, params: VoicevoxParam) -> bytes:
        audio_query = self.core.audio_query(text, speaker_id=speaker_id)
        for key, value in params:
            setattr(audio_query, key, value)
        wav = self.core.synthesis(audio_query, speaker_id=speaker_id)
        return wav
