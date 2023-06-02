import re
from typing import List
from typing import Optional

from vspeech.config import Config
from vspeech.config import ReplaceFilter
from vspeech.exceptions import ReplaceFilterParseError
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import InputQueues
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput
from vspeech.shared_context import is_text_event


def text_filter(filters: List[ReplaceFilter], text: Optional[str]):
    if not text:
        return ""
    replaced_text = f"{text}"
    for replace_filter in filters:
        replaced_text = re.sub(
            replace_filter.pattern, replace_filter.replaced, replaced_text
        )
    return replaced_text


def transform_content(command: WorkerInput, filters: List[ReplaceFilter]):
    if is_text_event(command.current_event.event):
        command.text = text_filter(filters=filters, text=command.text)


def process_command(context: SharedContext, request: WorkerInput):
    transform_content(filters=context.config.filters, command=request)
    current = request.current_event.event
    if EventType.transcription == current:
        context.put_queue(dest_event=EventType.transcription, request=request)
    if EventType.translation == current:
        context.put_queue(dest_event=EventType.translation, request=request)
    if EventType.subtitle == current:
        context.put_queue(dest_event=EventType.subtitle, request=request)
    if EventType.tts == current:
        context.put_queue(dest_event=EventType.tts, request=request)
    if EventType.vc == current:
        context.put_queue(dest_event=EventType.vc, request=request)
    if EventType.playback == current:
        context.put_queue(dest_event=EventType.playback, request=request)
    if EventType.pause == current:
        logger.debug("pause")
        context.running.clear()
    if EventType.resume == current:
        logger.debug("resume")
        context.running.set()
    if EventType.reload == current:
        already_stopped = not context.running.is_set()
        context.running.clear()
        file_path = request.file_path
        logger.debug("reload: %s", file_path)
        with open(file_path, "rb") as f:
            new_config = Config.read_config_from_file(f)
        workers = context.workers.values()
        logger.debug("workers: %s", workers)
        for worker in workers:
            worker.need_reload = False
            for config_name in worker.configs_depends_on:
                new_config_value = new_config.get_nested_value(config_name)
                old_config_value = context.config.get_nested_value(config_name)
                logger.debug(
                    "worker[%s] old [%s], new[%s].",
                    worker.event.name,
                    old_config_value,
                    new_config_value,
                )
                if new_config_value != old_config_value:
                    logger.info("worker[%s] will be reloaded.", worker.event.name)
                    worker.need_reload = True
                    break
        context.config = new_config
        if not already_stopped:
            context.running.set()
    if EventType.set_filters == current:
        context.config.filters.clear()
        filters = request.filters
        for filter in filters:
            if not filter:
                continue
            try:
                context.config.filters.append(ReplaceFilter.from_str(filter))
            except ReplaceFilterParseError:
                logger.warning("ignore invalid filter string %s", filter)
    if EventType.ping == current:
        logger.info("ping.")
    if EventType.forward == current:
        logger.info("forward.")
        worker_output = WorkerOutput.from_input(request)
        worker_output.text = request.text
        if worker_output.followings:
            context.sender_queue.put_nowait(worker_output)
