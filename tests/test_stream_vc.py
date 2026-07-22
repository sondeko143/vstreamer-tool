import numpy as np

from vspeech.lib.stream_vc import next_context
from vspeech.lib.stream_vc import slice_block_output


def test_next_context_returns_tail():
    seq = np.arange(5)
    assert list(next_context(seq, 2)) == [3, 4]


def test_next_context_zero_is_empty():
    seq = np.arange(3)
    assert len(next_context(seq, 0)) == 0


def test_slice_block_output_takes_block_tail():
    out = np.arange(10)
    # block is last block_len/seq_len = 2/10 of the sequence -> last 2 output samples
    assert list(slice_block_output(out, block_len=2, seq_len=10)) == [8, 9]


def test_slice_block_output_rounds_proportionally():
    out = np.arange(100)
    # 40 / (200+40) of 100 -> round(16.67) = 17
    assert len(slice_block_output(out, block_len=40, seq_len=240)) == 17
