import logging
from asyncio.tasks import current_task
from datetime import datetime
from sys import stdout

from vspeech.config import Config

logger = logging.getLogger()


class TaskStreamHandler(logging.StreamHandler):  # type: ignore
    def emit(self, record: logging.LogRecord) -> None:
        try:
            task = current_task()
            if task:
                record.__setattr__("task", f"{task.get_name()}")
            else:
                record.__setattr__("task", "main")
        except RuntimeError:
            record.__setattr__("task", "main")
        super().emit(record)


class TaskFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            task = current_task()
            if task:
                record.__setattr__("task", f"{task.get_name()}")
            else:
                record.__setattr__("task", "main")
        except RuntimeError:
            record.__setattr__("task", "main")
        super().emit(record)


def configure_logger(config: Config):
    log_format = logging.Formatter("%(asctime)s %(thread)s[%(task)s] : %(message)s")
    now = datetime.now()
    filename = now.strftime(config.log_file.replace("%%", "%"))
    file_handler = TaskFileHandler(filename, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(config.log_level)
    stdout_handler = TaskStreamHandler(stdout)
    stdout_handler.setLevel(config.log_level)
    stdout_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(config.log_level)
