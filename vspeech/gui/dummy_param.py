from dataclasses import dataclass
from typing import List

from pyvcroid2.pyvcroid2 import Param


@dataclass
class DummySpeaker:
    voiceName: str


@dataclass
class DummyTTtsParam:
    speaker: List[DummySpeaker]
    numSpeakers: int
    voiceName: str


def create_dummy_param() -> Param:
    speaker = DummySpeaker(voiceName="dummy")
    ttts_param = DummyTTtsParam(speaker=[speaker], numSpeakers=1, voiceName="dummy")
    return Param(default_parameter=ttts_param, parameter=ttts_param)
