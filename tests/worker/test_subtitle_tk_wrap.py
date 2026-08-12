from vspeech.worker.subtitle_tk import wrap_text_to_width


def fixed_width(px_per_char: int):
    """A Tk-free stand-in for Font.measure: every char is `px_per_char` wide."""
    return lambda s: len(s) * px_per_char


def test_line_that_fits_is_returned_unchanged():
    measure = fixed_width(10)
    # "abc" measures 30 <= 100, so no wrapping happens.
    assert wrap_text_to_width("abc", measure, max_width=100) == "abc"


def test_non_positive_max_width_disables_wrapping():
    measure = fixed_width(10)
    # Historically the caller gated the whole block on `max_width > 0`.
    assert wrap_text_to_width("abcdef", measure, max_width=0) == "abcdef"
    assert wrap_text_to_width("abcdef", measure, max_width=-5) == "abcdef"


def test_long_line_wraps_at_char_boundary():
    measure = fixed_width(10)
    # max_width=25 fits 2 chars (20<=25) but not 3 (30>25).
    assert wrap_text_to_width("ABCDE", measure, max_width=25) == "AB\nCD\nE"


def test_char_wider_than_max_width_is_kept_alone():
    measure = fixed_width(10)
    # Even though each char (10) exceeds max_width (5), no char is dropped:
    # the first char of each wrapped line is placed unconditionally.
    assert wrap_text_to_width("ABCD", measure, max_width=5) == "A\nB\nC\nD"


def test_existing_newlines_are_preserved_and_wrapped_independently():
    measure = fixed_width(10)
    # First line fits (20<=25); second line wraps.
    assert wrap_text_to_width("AB\nCDEF", measure, max_width=25) == "AB\nCD\nEF"


def test_empty_lines_are_preserved():
    measure = fixed_width(10)
    assert wrap_text_to_width("AB\n\nCD", measure, max_width=100) == "AB\n\nCD"
