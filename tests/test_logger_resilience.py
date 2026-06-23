from vspeech.config import Config
from vspeech.logger import configure_logger
from vspeech.logger import logger


def test_configure_logger_survives_bad_log_file(tmp_path):
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    bad = str(afile / "sub" / "voice.log")  # parent is a file → mkdir fails
    cfg = Config()
    cfg.log_file = bad
    before = list(logger.handlers)
    try:
        configure_logger(cfg)  # must not raise
        # stdout handler still attached (at least one handler present)
        assert len(logger.handlers) >= 1
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
