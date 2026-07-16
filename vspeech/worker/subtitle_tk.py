"""subtitle の TK バックエンド (ADR-0040)。

ADR-0040 の非ゴールにより、ここのロジックは OBS バックエンド追加の前後で
変えない。tkinter への依存はこのファイルだけに閉じ込める。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import current_task
from asyncio import sleep
from collections.abc import Callable
from functools import partial
from sys import platform
from tkinter import Canvas
from tkinter import Tk
from tkinter.font import Font
from typing import Any
from typing import Literal
from typing import Protocol

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.exceptions import shutdown_worker
from vspeech.lib.subtitle_state import Texts
from vspeech.lib.subtitle_state import age_panels
from vspeech.lib.subtitle_state import ingest_text
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


class _SubtitleCanvas(Protocol):
    """Structural surface `draw_text_with_outline`/`redraw_panel` need from a Tk
    `Canvas`. Narrower than `tkinter.Canvas` (whose `delete`/`pack`/`create_text`
    are part of a large, overload-heavy widget hierarchy) so a display-free test
    fake can satisfy it structurally without subclassing the real widget, which
    needs a live Tk root and whose `delete`/`pack` overrides would otherwise
    violate Liskov substitution. Every member here, including `create_text`'s
    keyword-only parameters, is typed to what typeshed's `tkinter.Canvas` stub
    actually accepts (or a subset of it, per Protocol contravariance), so a real
    `Canvas` still satisfies this structurally and `ty` verifies it.
    """

    def delete(self, tag: str, /) -> object: ...
    def pack(self) -> object: ...
    def create_text(
        self,
        x: float,
        y: float,
        /,
        *,
        text: str,
        font: Font,
        fill: str,
        anchor: Anchor,
        justify: Literal["left", "center", "right"],
        tags: str,
    ) -> object: ...


def wrap_text_to_width(text: str, measure: Callable[[str], int], max_width: int) -> str:
    """Hard-wrap `text` so no line measures wider than `max_width`.

    `measure` maps a string to its rendered pixel width (e.g. `Font.measure`);
    injecting it keeps this function pure and Tk-free. `max_width <= 0` disables
    wrapping and returns `text` unchanged. Existing newlines are preserved and
    each source line is wrapped independently; a single character that alone
    exceeds `max_width` is still kept (never dropped) on its own line.

    Costs one `measure` call per character once a line does not fit, and each
    call is a Tcl round-trip: measured at ~0.44ms, so a wrapping redraw blocks
    the event loop for 26-87ms (1920px / Meiryo UI 24 / 100-200 chars), against
    0.53ms when the line fits and the early return above takes over. Only
    `max_text_len`, `font_size` and the window width decide which side of that
    cliff you land on. A binary search over the prefix length would give the
    same result in ~7 calls: `measure` is monotonic in prefix length, so the
    longest prefix that fits is found exactly. Deferred as out of scope for the
    OBS backend branch, which does not use this path at all (the OBS backend
    lets `extents_wrap` break lines inside OBS).
    """
    if max_width <= 0:
        return text
    wrapped_lines: list[str] = []
    for line in text.split("\n"):
        if not line:
            wrapped_lines.append("")
            continue
        if measure(line) <= max_width:
            wrapped_lines.append(line)
            continue
        current_line = ""
        for char in line:
            if not current_line:
                current_line = char
                continue
            test_line = current_line + char
            if measure(test_line) <= max_width:
                current_line = test_line
            else:
                wrapped_lines.append(current_line)
                current_line = char
        if current_line:
            wrapped_lines.append(current_line)
    return "\n".join(wrapped_lines)


def draw_text_with_outline(
    canvas: _SubtitleCanvas,
    text_coord_x: float,
    text_coord_y: float,
    texts: str,
    text_tag: str,
    anchor: Anchor,
    config: SubtitleTextConfig,
    max_width: int = 0,
):
    text_color = config.font_color
    outline_color = config.outline_color

    font_tuple = Font(
        family=config.font_family,
        size=config.font_size,
        weight="bold" if config.font_style.lower() == "bold" else "normal",
    )

    texts = wrap_text_to_width(texts, font_tuple.measure, max_width)

    justify_val: Literal["left", "center", "right"] = "center"
    if "e" in anchor:
        justify_val = "right"
    elif "w" in anchor:
        justify_val = "left"

    offset = 1
    for i in range(0, 4):
        x = text_coord_x - offset if i % 2 == 0 else text_coord_x + offset
        y = text_coord_y + offset if i < 2 else text_coord_y - offset
        canvas.create_text(
            x,
            y,
            text=texts,
            font=font_tuple,
            fill=outline_color,
            anchor=anchor,
            justify=justify_val,
            tags=text_tag,
        )
    canvas.create_text(
        text_coord_x,
        text_coord_y,
        text=texts,
        font=font_tuple,
        fill=text_color,
        anchor=anchor,
        justify=justify_val,
        tags=text_tag,
    )


TRANSPARENT_BG_COLOR = "systemTransparent"
WIN32_TRANSPARENT_COLOR = "#000001"


def set_bg_color(canvas: Canvas, bg_color: str):
    if bg_color == TRANSPARENT_BG_COLOR and platform == "win32":
        canvas.configure(bg=WIN32_TRANSPARENT_COLOR)
    else:
        canvas.configure(bg=bg_color)


def redraw_panel(canvas: _SubtitleCanvas, ts: Texts):
    """Clear the panel's previous text for its tag and draw its current state.

    Deletes before drawing so successive frames don't stack, and packs so the
    canvas re-lays out. `max_width` reserves a `margin`-wide gutter each side.
    """
    canvas.delete(ts.tag)
    draw_text_with_outline(
        canvas=canvas,
        texts=ts.texts,
        text_coord_x=ts.coord_x,
        text_coord_y=ts.coord_y,
        text_tag=ts.tag,
        anchor=ts.anchor,
        config=ts.config,
        max_width=ts.bb_width - ts.config.margin * 2,
    )
    canvas.pack()


def on_closing(gui_task: Any):
    if not gui_task.cancelled() and not gui_task.done():
        gui_task.cancel()


async def subtitle_tk_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    tk_root = Tk()
    address = f"{context.config.listen_address}:{context.config.listen_port}"
    tk_root.title(f"vspeech:subtitle {address}")
    tk_root.protocol("WM_DELETE_WINDOW", partial(on_closing, current_task()))
    try:
        initial_width = context.config.subtitle.window_width
        initial_height = context.config.subtitle.window_height
        tk_root.geometry(f"{initial_width}x{initial_height}")
        tk_root.configure(borderwidth=0, highlightthickness=0)
        canvas = Canvas(
            tk_root,
            width=initial_width,
            height=initial_height,
            highlightthickness=0,
        )
        if (
            context.config.subtitle.bg_color == TRANSPARENT_BG_COLOR
            and platform == "win32"
        ):
            tk_root.wm_attributes("-transparentcolor", WIN32_TRANSPARENT_COLOR)
            tk_root.configure(bg=WIN32_TRANSPARENT_COLOR)
        else:
            tk_root.configure(bg=context.config.subtitle.bg_color)
        set_bg_color(canvas, bg_color=context.config.subtitle.bg_color)
        texts = {
            "n": Texts(
                tag="text",
                anchor=context.config.subtitle.text.anchor,
                config=context.config.subtitle.text,
                bb_height=tk_root.winfo_height(),
                bb_width=tk_root.winfo_width(),
            ),
            "s": Texts(
                tag="translated",
                anchor=context.config.subtitle.translated.anchor,
                config=context.config.subtitle.translated,
                bb_height=tk_root.winfo_height(),
                bb_width=tk_root.winfo_width(),
            ),
        }
        interval_sec = 1.0 / 30.0
        while True:
            set_bg_color(canvas, bg_color=context.config.subtitle.bg_color)
            for p in texts:
                if p == "n":
                    texts[p].config = context.config.subtitle.text
                elif p == "s":
                    texts[p].config = context.config.subtitle.translated
                texts[p].bb_width = tk_root.winfo_width()
                texts[p].bb_height = tk_root.winfo_height()
            for ts in age_panels(texts, interval_sec):
                redraw_panel(canvas, ts)
            tk_root.update()
            await sleep(interval_sec)
            try:
                message = in_queue.get_nowait()
                redraw_panel(canvas, ingest_text(texts, message))
            except QueueEmpty:
                pass
    except Exception as e:
        logger.exception(e)
        raise e
    except CancelledError as e:
        logger.info("subtitle worker cancelled")
        tk_root.destroy()
        raise shutdown_worker(e)
