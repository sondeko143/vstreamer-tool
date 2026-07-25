from vspeech.stream_vc.playback import detect_gap


def test_detect_gap_none_prev():
    assert detect_gap(None, 0) == 0


def test_detect_gap_contiguous():
    assert detect_gap(4, 5) == 0


def test_detect_gap_missing():
    assert detect_gap(4, 7) == 2  # 5, 6 missing


def test_detect_gap_reorder_or_dup_is_zero():
    assert detect_gap(7, 5) == 0  # out-of-order/dup -> not a forward gap
