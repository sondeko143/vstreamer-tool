"""Mapping from config to the OBS `text_gdiplus` settings dict (ADR-0041).

The config is the authority on display style, so every settings key of the tk version
is mapped onto its OBS counterpart. Formats that depend on the OBS version (the byte
order of the color int, the font-flag bits) are confined to this file.
"""

import re
from typing import Any
from typing import Literal

from vspeech.config import Anchor
from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleTextConfig
from vspeech.lib.subtitle_state import TRANSPARENT_BG_COLOR
from vspeech.lib.subtitle_state import anchor_to_justify
from vspeech.lib.subtitle_state import anchor_to_vertical
from vspeech.lib.subtitle_state import font_style_is_bold

# enum obs_font_style from obs-properties.h.
OBS_FONT_BOLD = 1

_HEX_COLOR = re.compile(r"\A#?([0-9a-fA-F]{6})\Z")


def hex_color_to_obs_int(hex_color: str) -> int:
    """Convert `#rrggbb` into the int OBS keeps in its settings.

    OBS stores colors as **0x00BBGGRR (BGR)**. Measured on real hardware (OBS 32.1.2 /
    obs-websocket 5.7.3): writing `0xFF8000` renders as rgb(0,128,255), and writing
    `0x0080FF` renders as rgb(255,128,0).

    The shipped defaults #ffffff / #000000 are palindromes, so getting this order wrong
    still passes with them. Only a test using an asymmetric color protects the order.
    """
    m = _HEX_COLOR.match(hex_color)
    if not m:
        raise ValueError(f"'{hex_color}' は #rrggbb 形式の色ではありません")
    digits = m.group(1)
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def font_size_to_obs_lfheight(font_size: int) -> int:
    """Pass Tk's `font_size` sign convention through to OBS's `font.size`
    (=`LOGFONT.lfHeight`) unchanged (ADR-0044).

    Only positive values (Tk's point spec) are converted to pixels at 96 DPI and
    returned negated -- `LOGFONT.lfHeight` uses the convention that a negative value is
    the em height and a positive one is the cell height (including internal leading),
    which is exactly Tk's convention. Negative values (already a pixel/em-height spec)
    pass through untouched.

    Running everything through `-round(size * 96/72)` without branching on the sign
    would turn an existing negative setting (e.g. -32) into +43, which OBS would misread
    as "cell height 43px" and diverge from Tk's look (em 32px) -- the branch exists to
    protect exactly that
    (test_font_size_to_obs_lfheight_passes_negative_through_unchanged).

    0 is the value Tk itself treats as "the platform default size", so there is no point
    count to convert. Pass it through -- although `-round(0 * 96/72)` also reaches 0 via
    `-0.0`, so this is a matter of reasoning rather than an arithmetic special case.
    """
    if font_size > 0:
        return -round(font_size * 96 / 72)
    return font_size


def anchor_to_align(anchor: Anchor) -> Literal["left", "center", "right"]:
    """OBS's `align` follows the same rule as tk's `justify`
    (lib/subtitle_state.anchor_to_justify). Not reimplemented here, to avoid keeping the
    rule in two places.

    `"center"` contains `"e"` as a substring, so a naive `"e" in anchor` test would
    wrongly yield `"right"` -- that is why `anchor_to_justify` carries the
    `anchor == "center"` guard.
    """
    return anchor_to_justify(anchor)


def anchor_to_valign(anchor: Anchor) -> Literal["top", "center", "bottom"]:
    """OBS's `valign` follows `lib/subtitle_state.anchor_to_vertical` (the same vertical
    placement rule as `Texts.coord_y`). Not reimplemented here, to avoid keeping the
    rule in two places.

    `"center"` contains `"n"` as a substring, so a naive `"n" in anchor` test would
    wrongly yield `"top"` -- that is why `anchor_to_vertical` carries the
    `anchor == "center"` guard.
    """
    return anchor_to_vertical(anchor)


def build_text_settings(
    text_config: SubtitleTextConfig, subtitle_config: SubtitleConfig
) -> dict[str, Any]:
    """Build the style settings for one panel. `text` is not included (it is pushed
    through a separate path)."""
    bg = subtitle_config.bg_color
    transparent = bg == TRANSPARENT_BG_COLOR
    margin = text_config.margin
    return {
        "font": {
            "face": text_config.font_family,
            "size": font_size_to_obs_lfheight(text_config.font_size),
            # `flags`' bold bit follows the shared rule in lib/subtitle_state
            # (font_style_is_bold) so this doesn't drift into a second
            # hand-synced copy from subtitle_tk.draw_text_with_outline's Tk
            # `weight` (ADR-0041).
            "flags": OBS_FONT_BOLD if font_style_is_bold(text_config.font_style) else 0,
        },
        "color": hex_color_to_obs_int(text_config.font_color),
        "opacity": 100,
        "outline": True,
        # tk draws its outline as four copies offset by 1px. This is the equivalent
        # thickness.
        "outline_size": 1,
        "outline_color": hex_color_to_obs_int(text_config.outline_color),
        "outline_opacity": 100,
        "align": anchor_to_align(text_config.anchor),
        "valign": anchor_to_valign(text_config.anchor),
        "bk_color": 0x000000 if transparent else hex_color_to_obs_int(bg),
        "bk_opacity": 0 if transparent else 100,
        "extents": True,
        "extents_cx": max(subtitle_config.window_width - margin * 2, 1),
        "extents_cy": max(subtitle_config.window_height - margin * 2, 1),
        "extents_wrap": True,
    }
