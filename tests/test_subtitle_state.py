import pytest

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.lib.subtitle_state import Text
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import age_panels
from vspeech.lib.subtitle_state import anchor_to_justify
from vspeech.lib.subtitle_state import next_expiry_sec


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


def test_age_panels_only_ages_the_head_entry():
    # tk バックエンドの 30fps ループは values[0] だけを減らす。壁時計でも同じ意味を保つ。
    panels = make_panels()
    panels["n"].values.append(Text(value="first", display_remain_sec=1.0))
    panels["n"].values.append(Text(value="second", display_remain_sec=5.0))
    age_panels(panels, 0.5)
    assert panels["n"].values[0].display_remain_sec == 0.5
    assert panels["n"].values[1].display_remain_sec == 5.0


def test_age_panels_pops_expired_head_and_returns_the_panel():
    panels = make_panels()
    panels["n"].values.append(Text(value="gone", display_remain_sec=0.25))
    panels["n"].values.append(Text(value="stays", display_remain_sec=5.0))
    changed = age_panels(panels, 0.25)
    assert changed == [panels["n"]]
    assert [t.value for t in panels["n"].values] == ["stays"]


def test_age_panels_never_goes_negative():
    panels = make_panels()
    panels["n"].values.append(Text(value="gone", display_remain_sec=0.1))
    age_panels(panels, 99.0)
    assert list(panels["n"].values) == []


def test_age_panels_returns_nothing_when_no_head_expires():
    panels = make_panels()
    panels["n"].values.append(Text(value="stays", display_remain_sec=5.0))
    assert age_panels(panels, 0.5) == []


def test_age_panels_ignores_empty_panels():
    panels = make_panels()
    assert age_panels(panels, 1.0) == []


def test_age_panels_handles_each_panel_independently():
    panels = make_panels()
    panels["n"].values.append(Text(value="n-gone", display_remain_sec=0.1))
    panels["s"].values.append(Text(value="s-stays", display_remain_sec=9.0))
    changed = age_panels(panels, 0.1)
    assert changed == [panels["n"]]
    assert [t.value for t in panels["s"].values] == ["s-stays"]


def test_next_expiry_sec_is_none_when_nothing_is_displayed():
    assert next_expiry_sec(make_panels()) is None


def test_next_expiry_sec_is_the_soonest_head_across_panels():
    panels = make_panels()
    panels["n"].values.append(Text(value="n", display_remain_sec=3.0))
    panels["s"].values.append(Text(value="s", display_remain_sec=1.5))
    assert next_expiry_sec(panels) == 1.5


def test_next_expiry_sec_ignores_non_head_entries():
    panels = make_panels()
    panels["n"].values.append(Text(value="head", display_remain_sec=4.0))
    panels["n"].values.append(Text(value="behind", display_remain_sec=0.1))
    assert next_expiry_sec(panels) == 4.0


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", "left"),
        ("n", "center"),
        ("ne", "right"),
        ("w", "left"),
        # "center" contains "e" as a substring, so an unguarded `"e" in
        # anchor` test would mis-fire as "right" -- this is the case that
        # regressed silently until this function existed as the one shared
        # copy the TK and OBS backends both call.
        ("center", "center"),
        ("e", "right"),
        ("sw", "left"),
        ("s", "center"),
        ("se", "right"),
    ],
)
def test_anchor_to_justify_covers_every_anchor(anchor: Anchor, expected: str):
    assert anchor_to_justify(anchor) == expected


# Characterisation tests for Texts.coord_x / coord_y, pinning TODAY's behaviour
# for all 9 Anchor members. bb_width/bb_height/margin are all distinct so a
# swapped axis (x for y) or a margin/centre mix-up (e.g. returning margin
# instead of bb_dim - margin) is visible as a wrong number, not a coincidental
# pass. These must keep passing unchanged across the coord_x/coord_y ->
# anchor_to_justify/anchor_to_vertical refactor -- if a refactor requires
# editing one of these expected values, the refactor changed behaviour.
_COORD_BB_WIDTH = 300
_COORD_BB_HEIGHT = 200
_COORD_MARGIN = 7


def make_texts_for_coords(anchor: Anchor) -> Texts:
    return Texts(
        tag="text",
        anchor=anchor,
        config=SubtitleTextConfig(anchor=anchor, margin=_COORD_MARGIN),
        bb_width=_COORD_BB_WIDTH,
        bb_height=_COORD_BB_HEIGHT,
    )


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", _COORD_MARGIN),
        ("n", _COORD_BB_WIDTH // 2),
        ("ne", _COORD_BB_WIDTH - _COORD_MARGIN),
        ("w", _COORD_MARGIN),
        ("center", _COORD_BB_WIDTH // 2),
        ("e", _COORD_BB_WIDTH - _COORD_MARGIN),
        ("sw", _COORD_MARGIN),
        ("s", _COORD_BB_WIDTH // 2),
        ("se", _COORD_BB_WIDTH - _COORD_MARGIN),
    ],
)
def test_coord_x_covers_every_anchor(anchor: Anchor, expected: int):
    assert make_texts_for_coords(anchor).coord_x == expected


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", _COORD_MARGIN),
        ("n", _COORD_MARGIN),
        ("ne", _COORD_MARGIN),
        ("w", _COORD_BB_HEIGHT // 2),
        ("center", _COORD_BB_HEIGHT // 2),
        ("e", _COORD_BB_HEIGHT // 2),
        ("sw", _COORD_BB_HEIGHT - _COORD_MARGIN),
        ("s", _COORD_BB_HEIGHT - _COORD_MARGIN),
        ("se", _COORD_BB_HEIGHT - _COORD_MARGIN),
    ],
)
def test_coord_y_covers_every_anchor(anchor: Anchor, expected: int):
    assert make_texts_for_coords(anchor).coord_y == expected
