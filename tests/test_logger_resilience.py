from vspeech.config import Config
from vspeech.logger import configure_logger
from vspeech.logger import logger


def test_stdout_handler_emits_utf8_under_narrow_encoding(monkeypatch):
    import io

    import vspeech.logger as lg

    raw = io.BytesIO()
    # Simulate a pipe/redirect whose encoding cannot represent Japanese.
    narrow = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(lg, "stdout", narrow)
    cfg = Config()
    cfg.log_file = ""  # no file handler; isolate the stdout handler
    before = list(logger.handlers)
    try:
        configure_logger(cfg)  # must reconfigure `narrow` to utf-8
        logger.error("設定不備 テスト")  # Japanese
        narrow.flush()
        # Without the fix, cp1252 can't encode this and nothing lands.
        assert "設定不備".encode() in raw.getvalue()
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)


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
