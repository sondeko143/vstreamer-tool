from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import Task
from asyncio import sleep
from functools import partial
from tkinter import Canvas
from tkinter import Tk
from typing import Any

from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.transcription import Transcription
from vspeech.translation import Translation


async def subtitle_worker(
    context: SharedContext,
    tk_root: Tk,
    in_queue: Queue[Transcription],
    translation_in_queue: Queue[Translation],
):
    try:
        width = 1600
        height = 120
        interval_sec = context.config.record_interval_sec / 2
        tk_root.geometry(f"{width}x{height}")
        tk_root.configure(bg="#00FF00", borderwidth=0)
        canvas = Canvas(
            tk_root, width=width, height=height, bg="#00FF00", highlightthickness=0
        )
        text_coord_x = canvas.winfo_reqwidth() / 2
        text_coord_y = canvas.winfo_reqheight() / 2
        font_tuple = (
            context.config.subtitle_font_family,
            f"{context.config.subtitle_font_size}",
            context.config.subtitle_font_style,
        )
        texts = ""
        translations = ""
        text_display_remain_sec = 0
        translation_display_remain_sec = 0
        max_text_len = context.config.max_subtitle_text_len
        max_translated_len = context.config.max_subtitle_translated_len
        text_tag = "text"
        translation_tag = "translation"
        while True:
            tk_root.update()
            await sleep(interval_sec)
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
            try:
                unprocessed_text = in_queue.get_nowait()
                if len(texts) + len(unprocessed_text.text) > max_text_len:
                    texts = ""
                    text_display_remain_sec = 0
                text_display_remain_sec += max(
                    len(unprocessed_text.text) / 3,
                    context.config.min_subtitle_display_sec,
                )
                canvas.delete(text_tag)
                texts += unprocessed_text.text
                original_text_id = canvas.create_text(
                    text_coord_x,
                    text_coord_y,
                    text=texts,
                    font=font_tuple,
                    fill="white",
                    anchor="s",
                    tags=text_tag,
                )
                box_id = canvas.create_rectangle(
                    canvas.bbox(original_text_id), fill="#000000", tags=text_tag
                )
                canvas.tag_lower(box_id)
                canvas.pack()
            except QueueEmpty:
                pass
            try:
                unprocessed_translation = translation_in_queue.get_nowait()
                if (
                    len(translations) + len(unprocessed_translation.translated)
                    > max_translated_len
                ):
                    translations = ""
                    translation_display_remain_sec = 0
                translation_display_remain_sec += max(
                    len(unprocessed_translation.translated) / 9,
                    context.config.min_subtitle_display_sec,
                )
                canvas.delete(translation_tag)
                if translations:
                    translations += " "
                translations += unprocessed_translation.translated
                translated_text_id = canvas.create_text(
                    text_coord_x,
                    text_coord_y,
                    text=translations,
                    font=font_tuple,
                    fill="white",
                    anchor="n",
                    tags=translation_tag,
                )
                box_id = canvas.create_rectangle(
                    canvas.bbox(translated_text_id),
                    fill="#000000",
                    tags=translation_tag,
                )
                canvas.tag_lower(box_id)
                canvas.pack()
            except QueueEmpty:
                pass
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
    tk_root.title("vspeech:subtitle")
    in_queue = Queue[Transcription]()
    event = EventType.subtitle
    context.input_queues[event] = in_queue
    translated_event = EventType.subtitle_translated
    translated_in_queue = Queue[Translation]()
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
