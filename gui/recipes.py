from collections.abc import Callable
from dataclasses import dataclass

from vspeech.config import Config


@dataclass
class Recipe:
    key: str
    label: str
    apply: Callable[[Config], Config]


def _mic_loopback(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["playback"]]
    return config


def _mic_transcribe_tts(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.transcription.enable = True
    config.tts.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["transcription", "tts", "playback"]]
    return config


def _mic_vc(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.vc.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["vc", "playback"]]
    return config


def _text_tts(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = False
    config.tts.enable = True
    config.playback.enable = True
    config.text_send_operations = [["tts", "playback"]]
    return config


def _blank(base: Config) -> Config:
    return base.model_copy(deep=True)


RECIPES: list[Recipe] = [
    Recipe("mic_loopback", "マイク→再生 (モニター)", _mic_loopback),
    Recipe(
        "mic_transcribe_tts", "マイク→文字起こし→読み上げ→再生", _mic_transcribe_tts
    ),
    Recipe("mic_vc", "マイク→ボイチェン→再生", _mic_vc),
    Recipe("text_tts", "テキスト→読み上げ→再生", _text_tts),
    Recipe("blank", "空 (default のまま)", _blank),
]

RECIPES_BY_KEY: dict[str, Recipe] = {recipe.key: recipe for recipe in RECIPES}
