import logging

import pytest

# vspeech.lib.voiceroid imports pyvcroid2 at module import time; skip cleanly
# where the vroid2 extra (Windows-only) isn't installed.
pytest.importorskip("pyvcroid2")

from vspeech.config import VR2Param  # noqa: E402
from vspeech.lib.voiceroid import VR2  # noqa: E402


class FakeParam:
    """Stand-in for pyvcroid2's Param: stores whatever camelCase attr is set."""


class FakeVcRoid2:
    def __init__(self):
        self.param = FakeParam()
        self.loaded: list[str] = []

    def loadVoice(self, voice_name):  # noqa: N802 - mirrors pyvcroid2's API
        self.loaded.append(voice_name)


def _vr2_with_fake() -> tuple[VR2, FakeVcRoid2]:
    fake = FakeVcRoid2()
    vr2 = VR2(vc_roid2=fake)  # ty: ignore[invalid-argument-type]
    return vr2, fake


def test_load_voice_applies_all_params_to_engine():
    """Every VR2Param field must be pushed onto vc_roid2.param (camelCased)."""
    vr2, fake = _vr2_with_fake()
    params = VR2Param(
        volume=1.5,
        speed=1.2,
        pitch=0.9,
        emphasis=1.1,
        pause_middle=120,
        pause_long=400,
        pause_sentence=900,
        master_volume=1.3,
    )

    vr2.load_voice("akari_44", params)

    assert fake.loaded == ["akari_44"]
    # keys are the camelCase names load_voice pushes onto the engine's param
    expected = {
        "volume": 1.5,
        "speed": 1.2,
        "pitch": 0.9,
        "emphasis": 1.1,
        "pauseMiddle": 120,
        "pauseLong": 400,
        "pauseSentence": 900,
        "masterVolume": 1.3,
    }
    for attr, want in expected.items():
        assert getattr(fake.param, attr) == want, attr


def test_load_voice_logs_applied_params_at_debug(caplog):
    """The DEBUG log must report the value actually read back from the engine."""
    vr2, _ = _vr2_with_fake()
    # vspeech.logger.logger is the root logger (colorlog.getLogger() with no name).
    with caplog.at_level(logging.DEBUG):
        vr2.load_voice("akari_44", VR2Param(volume=1.5))

    applied = [r.getMessage() for r in caplog.records if "applied" in r.getMessage()]
    assert any("volume" in m and "1.5" in m for m in applied)
