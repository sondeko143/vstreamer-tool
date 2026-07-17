"""config -> OBS `text_gdiplus` 設定 dict のマッピング (ADR-0041)。

config を表示スタイルの権威とするため、tk 版の設定キーをすべて OBS 側の
設定へ写す。OBS のバージョンに依存する形式 (色の int の並び、font flags の
ビット) はこのファイルだけに閉じ込める。
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

# obs-properties.h の enum obs_font_style。
OBS_FONT_BOLD = 1

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


def font_size_to_obs_lfheight(font_size: int) -> int:
    """Tk の `font_size` 符号規約のまま OBS の `font.size` (=`LOGFONT.lfHeight`)
    へ渡す (ADR-0044)。

    正 (Tk のポイント指定) だけ 96 DPI でピクセルへ換算し、符号を負にして
    返す -- `LOGFONT.lfHeight` は負が em 高、正がセル高（内部レディング込み）
    という規約を持ち、Tk の符号規約はこれと同一。負 (=既にピクセル/em 高の
    指定) はそのまま素通しする。

    ここを符号で分岐せず一律 `-round(size * 96/72)` に通すと、既存の負値
    設定 (例: -32) が +43 になり、OBS が「セル高 43px」と誤読して Tk の
    見た目 (em 32px) からずれる -- 分岐はここを守るためにある
    (test_font_size_to_obs_lfheight_passes_negative_through_unchanged)。

    0 は Tk 自身が「プラットフォーム既定サイズ」として扱う値で、変換対象の
    ポイント数が無い。素通しする -- もっとも `-round(0 * 96/72)` も `-0.0`
    経由で 0 になるので、これは算術上の特別扱いではなく理由づけの話でしかない。
    """
    if font_size > 0:
        return -round(font_size * 96 / 72)
    return font_size


def anchor_to_align(anchor: Anchor) -> Literal["left", "center", "right"]:
    """OBS の `align` は tk の `justify` と同じ規則 (lib/subtitle_state
    .anchor_to_justify) を使う。二重管理を避けるためここでは再実装しない。

    `"center"` は `"e"` を部分文字列として含むため、素の `"e" in anchor`
    判定だと誤って `"right"` になる -- `anchor_to_justify` 側がその
    `anchor == "center"` ガードを持つ理由。
    """
    return anchor_to_justify(anchor)


def anchor_to_valign(anchor: Anchor) -> Literal["top", "center", "bottom"]:
    """OBS の `valign` は `lib/subtitle_state.anchor_to_vertical`
    (`Texts.coord_y` と同じ縦位置の判定) を使う。二重管理を避けるためここで
    は再実装しない。

    `"center"` は `"n"` を部分文字列として含むため、素の `"n" in anchor`
    判定だと誤って `"top"` になる -- `anchor_to_vertical` 側がその
    `anchor == "center"` ガードを持つ理由。
    """
    return anchor_to_vertical(anchor)


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
            "size": font_size_to_obs_lfheight(text_config.font_size),
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
