"""config -> OBS `text_gdiplus` 設定 dict のマッピング (ADR-0041)。

config を表示スタイルの権威とするため、tk 版の設定キーをすべて OBS 側の
設定へ写す。OBS のバージョンに依存する形式 (色の int の並び、font flags の
ビット) はこのファイルだけに閉じ込める。
"""

import re
from typing import Any

from vspeech.config import Anchor
from vspeech.config import SubtitleConfig
from vspeech.config import SubtitleTextConfig

# obs-properties.h の enum obs_font_style。
OBS_FONT_BOLD = 1

# tk の TRANSPARENT_BG_COLOR と同じ番兵。tk では win32 の -transparentcolor に
# 化けるが、OBS では背景の不透明度 0 に写す (カラーキー自体が不要になる)。
TRANSPARENT_BG_COLOR = "systemTransparent"

_HEX_COLOR = re.compile(r"\A#?([0-9a-fA-F]{6})\Z")


def hex_color_to_obs_int(hex_color: str) -> int:
    """`#rrggbb` を OBS が設定に持つ int へ変換する。

    OBS は色を **0x00BBGGRR (BGR)** で保存する。実機 (OBS 32.1.2 /
    obs-websocket 5.7.3) で測定した: `0xFF8000` を書き込むと rgb(0,128,255)
    にレンダリングされ、`0x0080FF` を書き込むと rgb(255,128,0) になる。

    出荷デフォルトの #ffffff / #000000 は回文なので、この並びを取り違えても
    素通りする。並びを守れるのは非対称な色のテストだけ。
    """
    m = _HEX_COLOR.match(hex_color)
    if not m:
        raise ValueError(f"'{hex_color}' は #rrggbb 形式の色ではありません")
    digits = m.group(1)
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def anchor_to_align(anchor: Anchor) -> str:
    """tk の `draw_text_with_outline` の justify 規則をそのまま写す。

    `subtitle_tk.draw_text_with_outline` の `justify_val` はここと同じ部分
    文字列判定で、`anchor == "center"` を先読みするガードを持たない。その
    ため `"center"` は `"e"` を含む文字列として素通りし `"right"` になる
    -- これは tk 側の既存の挙動で、ADR-0040 によりこのブランチでは変えない
    ので、OBS 側もそのまま写して見た目を揃える。
    """
    if "e" in anchor:
        return "right"
    if "w" in anchor:
        return "left"
    return "center"


def anchor_to_valign(anchor: Anchor) -> str:
    """tk の `coord_y` が anchor の n/s 成分で決めているのと同じ区分。

    `subtitle_state.Texts.coord_y` は部分文字列判定の前に
    `anchor == "center"` を明示的にガードしている (`"center"` は `"n"` も
    含むので、ガードが無いと誤って top 側に倒れる)。実際の縦位置を決めるのは
    `coord_y` の側なので、ここでも同じガードを写す。
    """
    if anchor == "center":
        return "center"
    if "n" in anchor:
        return "top"
    if "s" in anchor:
        return "bottom"
    return "center"


def build_text_settings(
    text_config: SubtitleTextConfig, subtitle_config: SubtitleConfig
) -> dict[str, Any]:
    """1 パネル分のスタイル設定を組む。`text` は含めない (別経路で push する)。"""
    bg = subtitle_config.bg_color
    transparent = bg == TRANSPARENT_BG_COLOR
    margin = text_config.margin
    return {
        "font": {
            "face": text_config.font_family,
            "size": text_config.font_size,
            "flags": OBS_FONT_BOLD if text_config.font_style.lower() == "bold" else 0,
        },
        "color": hex_color_to_obs_int(text_config.font_color),
        "opacity": 100,
        "outline": True,
        # tk は 1px オフセットの 4 隅コピーで輪郭を描く。それに相当する太さ。
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
