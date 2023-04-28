from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import Task
from asyncio import sleep
from functools import partial
from sys import platform
from tkinter import Canvas
from tkinter import Tk
from typing import Any
from typing import Literal
from typing import Tuple
from typing import TypeAlias

from vspeech.config import SubtitleTextConfig
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


def update_text(current_text: str, add_text: str, max_text_len: int) -> str:
    if len(current_text) + len(add_text) > max_text_len:
        return add_text
    return current_text + " " + add_text


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


Anchor: TypeAlias = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]


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
    font_tuple: Tuple[str, str, str] = (
        config.font_family,
        f"{config.font_size}",
        config.font_style,
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


async def subtitle_worker(
    context: SharedContext,
    tk_root: Tk,
    in_queue: Queue[WorkerInput],
    translation_in_queue: Queue[WorkerInput],
):
    try:
        initial_width = context.config.subtitle.subtitle_window_width
        initial_height = context.config.subtitle.subtitle_window_height
        tk_root.geometry(f"{initial_width}x{initial_height}")
        tk_root.configure(borderwidth=0, highlightthickness=0)
        canvas = Canvas(
            tk_root,
            width=initial_width,
            height=initial_height,
            highlightthickness=0,
        )
        if (
            context.config.subtitle.subtitle_bg_color == TRANSPARENT_BG_COLOR
            and platform == "win32"
        ):
            tk_root.wm_attributes("-transparentcolor", WIN32_TRANSPARENT_COLOR)
            tk_root.configure(bg=WIN32_TRANSPARENT_COLOR)
        else:
            tk_root.configure(bg=context.config.subtitle.subtitle_bg_color)
        set_bg_color(canvas, bg_color=context.config.subtitle.subtitle_bg_color)
        texts = ""
        translations = ""
        text_display_remain_sec = 0
        translation_display_remain_sec = 0
        text_tag = "text"
        translation_tag = "translation"
        interval_sec = 1.0 / 30.0
        while True:
            config = context.config.subtitle
            set_bg_color(canvas, bg_color=context.config.subtitle.subtitle_bg_color)
            text_display_remain_sec = max(text_display_remain_sec - interval_sec, 0)
            if text_display_remain_sec <= 0:
                canvas.delete(text_tag)
                texts = ""
            translation_display_remain_sec = max(
                translation_display_remain_sec - interval_sec, 0
            )
            if translation_display_remain_sec <= 0:
                canvas.delete(translation_tag)
                translations = ""
            text_coord_x = tk_root.winfo_width() / 2
            text_coord_y = tk_root.winfo_height() / 2
            tk_root.update()
            await sleep(interval_sec)
            try:
                unprocessed_text = in_queue.get_nowait()
                text_display_remain_sec = update_display_sec(
                    current_sec=text_display_remain_sec,
                    current_text=texts,
                    add_text=unprocessed_text.text,
                    config=config.text,
                )
                texts = update_text(
                    current_text=texts,
                    add_text=unprocessed_text.text,
                    max_text_len=config.text.max_text_len,
                )
                canvas.delete(text_tag)
                draw_text_with_outline(
                    canvas=canvas,
                    texts=texts,
                    text_coord_x=text_coord_x,
                    text_coord_y=text_coord_y,
                    text_tag=text_tag,
                    anchor="s",
                    config=config.text,
                )
                canvas.pack()
            except QueueEmpty:
                pass
            try:
                unprocessed_translation = translation_in_queue.get_nowait()
                translation_display_remain_sec = update_display_sec(
                    current_sec=text_display_remain_sec,
                    current_text=translations,
                    add_text=unprocessed_translation.text,
                    config=config.translated,
                )
                translations = update_text(
                    current_text=translations,
                    add_text=unprocessed_translation.text,
                    max_text_len=config.translated.max_text_len,
                )
                canvas.delete(translation_tag)
                draw_text_with_outline(
                    canvas=canvas,
                    texts=translations,
                    text_coord_x=text_coord_x,
                    text_coord_y=text_coord_y,
                    text_tag=translation_tag,
                    anchor="n",
                    config=config.translated,
                )
                canvas.pack()
            except QueueEmpty:
                pass
    except Exception as e:
        logger.exception(e)
        raise e
    except CancelledError:
        logger.info("gui worker cancelled")
        tk_root.destroy()
        raise


def on_closing(gui_task: Task[Any]):
    if not gui_task.cancelled() and not gui_task.done():
        gui_task.cancel()


def create_subtitle_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    tk_root = Tk()
    tk_root.title(f"vspeech:subtitle {context.config.listen_address}")
    in_queue = Queue[WorkerInput]()
    event = EventType.subtitle
    context.input_queues[event] = in_queue
    translated_event = EventType.subtitle_translated
    translated_in_queue = Queue[WorkerInput]()
    context.input_queues[translated_event] = translated_in_queue
    gui_task = loop.create_task(
        subtitle_worker(
            context,
            tk_root,
            in_queue=in_queue,
            translation_in_queue=translated_in_queue,
        ),
        name=event.name,
    )
    tk_root.protocol("WM_DELETE_WINDOW", partial(on_closing, gui_task))
    return gui_task
