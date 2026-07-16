"""subtitle の純粋な状態機械 (ADR-0040)。

TK / OBS 両バックエンドがこれを共有するので、履歴・トリム・区切り文字・
表示時間の意味はバックエンドによらず同一になる。tkinter に依存しない。
"""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.shared_context import WorkerInput


def anchor_to_justify(anchor: Anchor) -> Literal["left", "center", "right"]:
    """The shared TK `justify` / OBS `align` rule -- one copy, not two.

    `"center"` contains both `"e"` and `"n"` as substrings (c-e-n-t-e-r), so
    it must be guarded before any bare `"e" in anchor` / `"n" in anchor`
    test, or it mis-fires (e.g. `anchor="center"` would justify "right").
    Mirrors the same guard `coord_x`/`coord_y` below already use for
    position. Used by `subtitle_tk.draw_text_with_outline` (as `justify`)
    and `lib.obs_text_settings.anchor_to_align` (as `align`) so the two
    backends can't drift into hand-synced copies of this rule again.
    """
    if anchor == "center":
        return "center"
    if "e" in anchor:
        return "right"
    if "w" in anchor:
        return "left"
    return "center"


def update_display_sec(
    current_sec: float, current_text: str, add_text: str, config: SubtitleTextConfig
) -> float:
    min_sec = config.min_display_sec
    sec_per_letter = config.display_sec_per_letter
    max_text_len = config.max_text_len
    add_sec = max(
        len(add_text) * sec_per_letter,
        min_sec,
    )
    if len(current_text) + len(add_text) > max_text_len:
        return add_sec
    return current_sec + add_sec


@dataclass
class Text:
    value: str = ""
    display_remain_sec: float = 0


@dataclass
class Texts:
    tag: str
    anchor: Anchor
    config: SubtitleTextConfig
    bb_width: int
    bb_height: int
    display_remain_sec: float = 0
    values: deque[Text] = field(init=False)

    def __post_init__(self):
        self.values = deque([], maxlen=self.config.max_histories)

    @property
    def coord_x(self):
        if self.anchor == "center":
            return self.bb_width // 2
        elif "e" in self.anchor:
            return self.bb_width - self.config.margin
        elif "w" in self.anchor:
            return self.config.margin
        else:
            return self.bb_width // 2

    @property
    def coord_y(self):
        if self.anchor == "center":
            return self.bb_height // 2
        elif "s" in self.anchor:
            return self.bb_height - self.config.margin
        elif "n" in self.anchor:
            return self.config.margin
        else:
            return self.bb_height // 2

    @property
    def texts(self):
        if "s" in self.anchor:
            return self.config.delimiter.join(t.value for t in reversed(self.values))
        else:
            return self.config.delimiter.join(t.value for t in self.values)


def how_many_should_we_pop(texts: deque[Text], max_length: int):
    total_length = 0
    for idx, text in enumerate(reversed(texts)):
        total_length += len(text.value)
        if total_length > max_length:
            return len(texts) - (idx + 1)
    return 0


def ingest_text(texts: dict[str, Texts], message: WorkerInput) -> Texts:
    """Route an inbound message to its panel, append it, and trim overflow.

    Picks the panel named by the message's `position` param, falling back to the
    "n" panel when it is unset/unknown. Appends the text as a new entry whose
    display duration comes from `update_display_sec`, then drops the oldest
    entries that overflow `max_text_len`. Returns the panel that needs redrawing.
    """
    position = message.current_event.params.position
    ts = texts[position] if position in texts else texts["n"]
    t = Text()
    t.display_remain_sec = update_display_sec(
        current_sec=t.display_remain_sec,
        current_text=t.value,
        add_text=message.text,
        config=ts.config,
    )
    t.value = message.text
    ts.values.append(t)
    n_pop = how_many_should_we_pop(ts.values, max_length=ts.config.max_text_len)
    for _ in range(n_pop):
        ts.values.popleft()
    return ts


def age_panels(panels: dict[str, Texts], elapsed_sec: float) -> list[Texts]:
    """Age each panel's head entry by `elapsed_sec` and drop it once it expires.

    Mirrors the TK loop's aging: only `values[0]` counts down, so entries queue
    up behind the one on screen. TK drives this off a fixed 1/30s frame count;
    callers here pass real elapsed time. Returns the panels whose contents
    changed and therefore need re-rendering.
    """
    changed: list[Texts] = []
    for ts in panels.values():
        if not ts.values:
            continue
        t = ts.values[0]
        t.display_remain_sec = max(t.display_remain_sec - elapsed_sec, 0)
        if t.display_remain_sec <= 0:
            ts.values.popleft()
            changed.append(ts)
    return changed


def next_expiry_sec(panels: dict[str, Texts]) -> float | None:
    """Seconds until the soonest head entry expires, or None if nothing is shown."""
    remains = [ts.values[0].display_remain_sec for ts in panels.values() if ts.values]
    return min(remains) if remains else None
