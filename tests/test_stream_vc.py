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


def test_next_context_clamps_when_longer_than_seq():
    # context_len > len(seq): return the whole buffer (not a negative-index slice)
    seq = np.arange(3)
    assert list(next_context(seq, 5)) == [0, 1, 2]


def test_slice_block_output_block_len_zero_returns_all():
    out = np.arange(10)
    assert list(slice_block_output(out, block_len=0, seq_len=10)) == list(range(10))


def test_slice_block_output_clamps_when_block_exceeds_seq():
    # block_len > seq_len: clamp to the whole output, not a negative-index slice
    out = np.arange(100)
    assert list(slice_block_output(out, block_len=300, seq_len=240)) == list(range(100))


def test_helpers_work_on_torch_tensors():
    # docstring claims numpy/torch agnosticism; verify the torch path.
    import pytest

    torch = pytest.importorskip("torch")
    seq = torch.arange(5)
    assert next_context(seq, 2).tolist() == [3, 4]
    assert next_context(seq, 0).numel() == 0
    out = torch.arange(10)
    assert slice_block_output(out, block_len=2, seq_len=10).tolist() == [8, 9]
