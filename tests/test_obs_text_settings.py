import pytest

from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleTextConfig
from vspeech.lib.obs_text_settings import anchor_to_align
from vspeech.lib.obs_text_settings import anchor_to_valign
from vspeech.lib.obs_text_settings import build_text_settings
from vspeech.lib.obs_text_settings import font_size_to_obs_lfheight
from vspeech.lib.obs_text_settings import hex_color_to_obs_int


def test_hex_color_to_obs_int_reverses_rgb_to_bgr():
    # OBS は 0x00BBGGRR で保存する (実機 32.1.2 で GetSourceScreenshot により測定)。
    # #ffffff / #000000 は回文なので取り違えても素通りする -- この 1 本と次の 1 本
    # だけがバイト順を守っている (ADR-0041)。
    assert hex_color_to_obs_int("#ff8000") == 0x0080FF


def test_hex_color_to_obs_int_reversal_is_not_incidental():
    assert hex_color_to_obs_int("#0080ff") == 0xFF8000


def test_hex_color_to_obs_int_accepts_shipped_defaults():
    assert hex_color_to_obs_int("#ffffff") == 0xFFFFFF
    assert hex_color_to_obs_int("#000000") == 0x000000


def test_hex_color_to_obs_int_is_case_insensitive_and_hash_optional():
    # "FF8000" / "#Ff8000" is the same colour as "#ff8000" above, so it must
    # reverse to the same BGR int (0x0080FF), not the un-reversed 0xFF8000 --
    # asserting the un-reversed value here would silently contradict the
    # reversal this module exists to get right.
    assert hex_color_to_obs_int("FF8000") == 0x0080FF
    assert hex_color_to_obs_int("#Ff8000") == 0x0080FF


@pytest.mark.parametrize("bad", ["", "#fff", "#gggggg", "#ff80000", "nope"])
def test_hex_color_to_obs_int_rejects_junk(bad: str):
    with pytest.raises(ValueError):
        hex_color_to_obs_int(bad)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", "left"),
        ("w", "left"),
        ("sw", "left"),
        ("ne", "right"),
        ("e", "right"),
        ("se", "right"),
        ("n", "center"),
        ("s", "center"),
        # "center" contains "e" as a substring, so an unguarded test would
        # mis-fire as "right"; anchor_to_justify guards `anchor == "center"`
        # first (see lib/subtitle_state.anchor_to_justify).
        ("center", "center"),
    ],
)
def test_anchor_to_align_matches_the_tk_justify_rule(anchor, expected):
    # tk の draw_text_with_outline と同じ規則 (lib/subtitle_state
    # .anchor_to_justify 経由): e -> right, w -> left, else center
    assert anchor_to_align(anchor) == expected


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("nw", "top"),
        ("n", "top"),
        ("ne", "top"),
        ("sw", "bottom"),
        ("s", "bottom"),
        ("se", "bottom"),
        ("w", "center"),
        ("e", "center"),
        ("center", "center"),
    ],
)
def test_anchor_to_valign(anchor, expected):
    assert anchor_to_valign(anchor) == expected


@pytest.mark.parametrize(
    ("font_size", "expected_lfheight"),
    [
        # points -> 96 DPI pixels, negated to hit LOGFONT.lfHeight's
        # negative = em-height convention (ADR-0044). Checked against the
        # ADR's measured table: round(24*96/72) == 32, round(22*96/72) == 29,
        # round(56*96/72) == 75.
        (24, -32),
        (22, -29),
        (56, -75),
    ],
)
def test_font_size_to_obs_lfheight_converts_points_to_negative_pixels(
    font_size, expected_lfheight
):
    assert font_size_to_obs_lfheight(font_size) == expected_lfheight


def test_font_size_to_obs_lfheight_passes_negative_through_unchanged():
    # A negative font_size already means pixels/em-height in Tk's own
    # convention (identical to LOGFONT.lfHeight's). Routing it through the
    # points->pixels formula would turn -32 into +43, which OBS reads as a
    # 43px *cell* height -- wrong, and silently so (ADR-0044). This test is
    # the one that stops that branch from being "simplified" away.
    assert font_size_to_obs_lfheight(-32) == -32


def test_font_size_to_obs_lfheight_zero_passes_through():
    # Tk treats font_size=0 as "use the platform default size" -- there is
    # no point value to convert. Passing 0 through unchanged also happens to
    # be what the conversion formula would produce anyway
    # (-round(0 * 96/72) == -0 == 0), so this is not a special case in the
    # arithmetic, only in the reasoning for why 0 is not converted like a
    # positive size (ADR-0044).
    assert font_size_to_obs_lfheight(0) == 0


def test_build_text_settings_maps_every_tk_key():
    subtitle = SubtitleConfig(window_width=1920, window_height=120)
    text = SubtitleTextConfig(
        anchor="s",
        font_family="Meiryo UI",
        font_size=24,
        font_style="bold",
        font_color="#ff8000",
        outline_color="#0000ff",
        margin=4,
    )
    got = build_text_settings(text, subtitle)
    # font_size=24 (points) -> lfHeight -32 (ADR-0044), not a pass-through 24.
    assert got["font"] == {"face": "Meiryo UI", "size": -32, "flags": 1}
    # BGR: #ff8000 -> 0x0080FF, #0000ff -> 0xFF0000 (実機で測定, ADR-0041)
    assert got["color"] == 0x0080FF
    assert got["opacity"] == 100
    assert got["outline"] is True
    assert got["outline_size"] == 1
    assert got["outline_color"] == 0xFF0000
    assert got["outline_opacity"] == 100
    assert got["align"] == "center"
    assert got["valign"] == "bottom"
    assert got["extents"] is True
    assert got["extents_cx"] == 1920 - 4 * 2
    assert got["extents_cy"] == 120 - 4 * 2
    assert got["extents_wrap"] is True


def test_build_text_settings_non_bold_clears_the_flag():
    text = SubtitleTextConfig(font_style="normal")
    assert build_text_settings(text, SubtitleConfig())["font"]["flags"] == 0


def test_build_text_settings_bold_is_case_insensitive_like_tk():
    # tk: "bold" if config.font_style.lower() == "bold" else "normal"
    text = SubtitleTextConfig(font_style="BOLD")
    assert build_text_settings(text, SubtitleConfig())["font"]["flags"] == 1


def test_build_text_settings_wires_the_lfheight_conversion_for_pixel_configs():
    # Integration check that build_text_settings actually calls through
    # font_size_to_obs_lfheight rather than a second, separate pass-through
    # of its own -- a config that already says font_size=-32 (Tk pixels)
    # must reach OBS as -32, not get re-converted (ADR-0044).
    text = SubtitleTextConfig(font_size=-32)
    got = build_text_settings(text, SubtitleConfig())
    assert got["font"]["size"] == -32


def test_build_text_settings_transparent_bg_becomes_zero_opacity():
    subtitle = SubtitleConfig(bg_color="systemTransparent")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_opacity"] == 0


def test_build_text_settings_opaque_bg_is_honoured_like_tk():
    # tk で bg_color="#00ff00" なら緑の背景になる。OBS でも同じにする。
    # (緑は回文なのでバイト順は守れない -- 下の非対称ケースがそれを見る)
    subtitle = SubtitleConfig(bg_color="#00ff00")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_color"] == 0x00FF00
    assert got["bk_opacity"] == 100


def test_build_text_settings_bg_colour_is_also_bgr():
    subtitle = SubtitleConfig(bg_color="#ff8000")
    got = build_text_settings(SubtitleTextConfig(), subtitle)
    assert got["bk_color"] == 0x0080FF
    assert got["bk_opacity"] == 100


def test_build_text_settings_never_sets_a_negative_extent():
    # margin が窓より大きい病的な config でも OBS に負値を送らない。
    subtitle = SubtitleConfig(window_width=4, window_height=4)
    got = build_text_settings(SubtitleTextConfig(margin=100), subtitle)
    assert got["extents_cx"] >= 1
    assert got["extents_cy"] >= 1


def test_build_text_settings_does_not_set_text():
    # テキストは別経路 (毎回変わる) で push する。スタイルと混ぜない。
    assert "text" not in build_text_settings(SubtitleTextConfig(), SubtitleConfig())
