from gui.recipes import RECIPES_BY_KEY
from vspeech.config import Config


def test_mic_loopback():
    out = RECIPES_BY_KEY["mic_loopback"].apply(Config())
    assert out.recording.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["playback"]]
    assert out.transcription.enable is False


def test_mic_transcribe_tts():
    out = RECIPES_BY_KEY["mic_transcribe_tts"].apply(Config())
    assert out.recording.enable is True
    assert out.transcription.enable is True
    assert out.tts.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["transcription", "tts", "playback"]]


def test_mic_vc():
    out = RECIPES_BY_KEY["mic_vc"].apply(Config())
    assert out.recording.enable is True
    assert out.vc.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["vc", "playback"]]


def test_text_tts():
    out = RECIPES_BY_KEY["text_tts"].apply(Config())
    assert out.recording.enable is False
    assert out.tts.enable is True
    assert out.playback.enable is True
    assert out.text_send_operations == [["tts", "playback"]]


def test_blank_leaves_disabled():
    out = RECIPES_BY_KEY["blank"].apply(Config())
    assert out.recording.enable is False
    assert out.tts.enable is False


def test_apply_does_not_mutate_input():
    base = Config()
    RECIPES_BY_KEY["mic_loopback"].apply(base)
    assert base.recording.enable is False
