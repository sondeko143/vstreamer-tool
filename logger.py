from datetime import datetime
import logging
from sys import stdout

from config import Config

logger = logging.getLogger()


def configure_logger(config: Config):
    now = datetime.now()
    filename = now.strftime(config.log_file)
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)
