from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import Task
from asyncio import TaskGroup
from asyncio import sleep
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from sys import platform
from tkinter import Canvas
from tkinter import Tk
from tkinter.font import Font
from typing import Any

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.exceptions import shutdown_worker
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


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


def draw_text_with_outline(
    canvas: Canvas,
    text_coord_x: float,
    text_coord_y: float,
    texts: str,
    text_tag: str,
    anchor: Anchor,
    config: SubtitleTextConfig,
):
    text_color = config.font_color
    outline_color = config.outline_color

    font_tuple = Font(
        family=config.font_family,
        size=config.font_size,
        weight="bold" if config.font_style.lower() == "bold" else "normal",
    )
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
            tags=text_tag,
        )
    canvas.create_text(
        text_coord_x,
        text_coord_y,
        text=texts,
        font=font_tuple,
        fill=text_color,
        anchor=anchor,
        tags=text_tag,
    )


TRANSPARENT_BG_COLOR = "systemTransparent"
WIN32_TRANSPARENT_COLOR = "#000001"


def set_bg_color(canvas: Canvas, bg_color: str):
    if bg_color == TRANSPARENT_BG_COLOR and platform == "win32":
        canvas.configure(bg=WIN32_TRANSPARENT_COLOR)
    else:
        canvas.configure(bg=bg_color)


@dataclass
class Text:
    value: str = ""
    display_remain_sec: float = 0


@dataclass
class Texts:
    tag: str
    anchor: Anchor
    config: SubtitleTextConfig
    display_remain_sec: float = 0
    values: deque[Text] = field(init=False)

    def __post_init__(self):
        self.values = deque([], maxlen=self.config.max_histories)


def how_many_should_we_pop(texts: deque[Text], max_length: int):
    total_length = 0
    for idx, text in enumerate(reversed(texts)):
        total_length += len(text.value)
        if total_length > max_length:
            return len(texts) - (idx + 1)
    return 0


async def subtitle_worker(
    context: SharedContext,
    tk_root: Tk,
    in_queue: Queue[WorkerInput],
):
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
            ),
            "s": Texts(
                tag="translated",
                anchor=context.config.subtitle.translated.anchor,
                config=context.config.subtitle.translated,
            ),
        }
        interval_sec = 1.0 / 30.0
        while True:
            set_bg_color(canvas, bg_color=context.config.subtitle.bg_color)
            text_coord_x = tk_root.winfo_width() / 2
            text_coord_y = tk_root.winfo_height() / 2
            for p in texts:
                if p == "n":
                    texts[p].config = context.config.subtitle.text
                elif p == "s":
                    texts[p].config = context.config.subtitle.translated
                if not texts[p].values:
                    continue
                t = texts[p].values[0]
                t.display_remain_sec = max(t.display_remain_sec - interval_sec, 0)
                if t.display_remain_sec <= 0:
                    texts[p].values.popleft()
                    canvas.delete(texts[p].tag)
                    draw_text_with_outline(
                        canvas=canvas,
                        texts=texts[p].config.delimiter.join(
                            t.value for t in texts[p].values
                        ),
                        text_coord_x=text_coord_x,
                        text_coord_y=text_coord_y,
                        text_tag=texts[p].tag,
                        anchor=texts[p].anchor,
                        config=texts[p].config,
                    )
                    canvas.pack()
            tk_root.update()
            await sleep(interval_sec)
            try:
                unprocessed_text = in_queue.get_nowait()
                p = unprocessed_text.current_event.params.position
                if p in texts:
                    ts = texts[p]
                else:
                    ts = texts["n"]
                t = Text()
                t.display_remain_sec = update_display_sec(
                    current_sec=t.display_remain_sec,
                    current_text=t.value,
                    add_text=unprocessed_text.text,
                    config=ts.config,
                )
                t.value = unprocessed_text.text
                ts.values.append(t)
                n_pop = how_many_should_we_pop(
                    ts.values, max_length=ts.config.max_text_len
                )
                for _ in range(n_pop):
                    ts.values.popleft()
                canvas.delete(ts.tag)
                draw_text_with_outline(
                    canvas=canvas,
                    texts=ts.config.delimiter.join(t.value for t in ts.values),
                    text_coord_x=text_coord_x,
                    text_coord_y=text_coord_y,
                    text_tag=ts.tag,
                    anchor=ts.anchor,
                    config=ts.config,
                )
                canvas.pack()
            except QueueEmpty:
                pass
    except Exception as e:
        logger.exception(e)
        raise e
    except CancelledError as e:
        logger.info("subtitle worker cancelled")
        tk_root.destroy()
        raise shutdown_worker(e)


def on_closing(gui_task: Task[Any]):
    if not gui_task.cancelled() and not gui_task.done():
        gui_task.cancel()


def create_subtitle_task(
    tg: TaskGroup,
    context: SharedContext,
):
    tk_root = Tk()
    address = f"{context.config.listen_address}:{context.config.listen_port}"
    tk_root.title(f"vspeech:subtitle {address}")
    worker = context.add_worker(
        event=EventType.subtitle, configs_depends_on=["subtitle"]
    )
    gui_task = tg.create_task(
        subtitle_worker(
            context,
            tk_root,
            in_queue=worker.in_queue,
        ),
        name=worker.event.name,
    )
    tk_root.protocol("WM_DELETE_WINDOW", partial(on_closing, gui_task))
    return gui_task
