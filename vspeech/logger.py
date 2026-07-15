import logging
from asyncio.tasks import current_task
from datetime import datetime
from pathlib import Path
from sys import stderr
from sys import stdout

import colorlog
from colorlog.formatter import ColoredFormatter

from vspeech.config import Config

logger = colorlog.getLogger()


class TaskStreamHandler(colorlog.StreamHandler):
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
    # stdout/stderr がコンソール以外 (GUI サブプロセスの pipe, リダイレクト) だと
    # encoding が cp1252 等になり、日本語ログで UnicodeEncodeError を投げて emit が
    # 落ちる → preflight の失敗理由が stdout に出ない (ADR-0038 Goal 1 を潰す)。
    # UTF-8 + backslashreplace で「読める UTF-8 を出す」かつ「絶対に落ちない」を両立。
    for _stream in (stdout, stderr):
        try:
            _stream.reconfigure(  # ty: ignore[unresolved-attribute]
                encoding="utf-8", errors="backslashreplace"
            )
        except AttributeError, ValueError:
            pass
    log_file_format = logging.Formatter(
        "%(asctime)s %(thread)s[%(task)s] %(levelname)s : %(message)s"
    )
    log_sout_format = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname).4s%(reset)s %(thread)s[%(task)s]  : %(message)s"
    )
    now = datetime.now()
    filename = now.strftime(config.log_file.replace("%%", "%"))
    if filename:
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            file_handler = TaskFileHandler(filename, encoding="utf-8")
            file_handler.setFormatter(log_file_format)
            file_handler.setLevel(config.log_level)
            logger.addHandler(file_handler)
        except OSError as e:
            print(
                f"log file disabled (cannot open {filename}): {e}",
                file=stderr,
            )
    stdout_handler = TaskStreamHandler(stdout)
    stdout_handler.setLevel(config.log_level)
    stdout_handler.setFormatter(log_sout_format)
    logger.addHandler(stdout_handler)
    logger.setLevel(config.log_level)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
