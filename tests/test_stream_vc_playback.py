from vspeech.stream_vc.playback import UNDERFLOW_LOG_EVERY
from vspeech.stream_vc.playback import detect_gap
from vspeech.stream_vc.playback import should_log_underflow


def test_detect_gap_none_prev():
    assert detect_gap(None, 0) == 0


def test_detect_gap_contiguous():
    assert detect_gap(4, 5) == 0


def test_detect_gap_missing():
    assert detect_gap(4, 7) == 2  # 5, 6 missing


def test_detect_gap_reorder_or_dup_is_zero():
    assert detect_gap(7, 5) == 0  # out-of-order/dup -> not a forward gap


def test_underflow_logs_first_occurrence():
    assert should_log_underflow(1)


def test_underflow_log_is_rate_limited():
    # 持続 underflow (block_ms=160 なら ~6 回/秒) でログを溢れさせない。
    assert not should_log_underflow(2)
    assert not should_log_underflow(UNDERFLOW_LOG_EVERY - 1)


def test_underflow_logs_every_nth():
    assert should_log_underflow(UNDERFLOW_LOG_EVERY)
    assert should_log_underflow(UNDERFLOW_LOG_EVERY * 3)
