from uuid import uuid4

import pytest

from vspeech.config import EventType
from vspeech.config import SubtitleTextConfig
from vspeech.shared_context import EventAddress
from vspeech.shared_context import Params
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.subtitle import Text
from vspeech.worker.subtitle import Texts
from vspeech.worker.subtitle import ingest_text


def make_message(text: str, position=None) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(
            EventType.subtitle, params=Params(position=position)
        ),
        following_events=[],
        text=text,
        sound=SoundInput.invalid(),
        file_path="",
        filters=[],
    )


def make_panels() -> dict[str, Texts]:
    return {
        "n": Texts(
            tag="text",
            anchor="s",
            config=SubtitleTextConfig(anchor="s"),
            bb_width=300,
            bb_height=200,
        ),
        "s": Texts(
            tag="translated",
            anchor="n",
            config=SubtitleTextConfig(anchor="n"),
            bb_width=300,
            bb_height=200,
        ),
    }


def test_ingest_routes_to_position_panel():
    panels = make_panels()
    returned = ingest_text(panels, make_message("hi", position="s"))
    assert returned is panels["s"]
    assert [t.value for t in panels["s"].values] == ["hi"]
    assert list(panels["n"].values) == []


def test_ingest_falls_back_to_n_panel_when_position_is_none():
    panels = make_panels()
    returned = ingest_text(panels, make_message("hi", position=None))
    assert returned is panels["n"]
    assert [t.value for t in panels["n"].values] == ["hi"]


def test_ingest_appends_message_text_as_newest():
    panels = make_panels()
    panels["n"].values.append(Text(value="old"))
    ingest_text(panels, make_message("new", position="n"))
    assert [t.value for t in panels["n"].values] == ["old", "new"]


def test_ingest_sets_display_remain_sec_to_min_for_short_text():
    panels = make_panels()
    ingest_text(panels, make_message("hi", position="n"))
    # "hi" (2 letters * 0.5) is below the 2.5s floor, so it gets min_display_sec.
    assert panels["n"].values[-1].display_remain_sec == pytest.approx(2.5)


def test_ingest_pops_oldest_when_total_exceeds_max_text_len():
    panels = make_panels()
    # max_text_len defaults to 30; three 20-char entries overflow it.
    panels["n"].values.append(Text(value="A" * 20))
    panels["n"].values.append(Text(value="B" * 20))
    ingest_text(panels, make_message("C" * 20, position="n"))
    assert [t.value for t in panels["n"].values] == ["B" * 20, "C" * 20]
