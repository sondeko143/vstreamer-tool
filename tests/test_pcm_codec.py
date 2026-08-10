"""Shared PCM decode/encode used at every device boundary (ADR-0070)."""

import numpy as np
import pytest

from vspeech.config import SampleFormat
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm

FORMATS = [
    SampleFormat.UINT8,
    SampleFormat.INT8,
    SampleFormat.INT16,
    SampleFormat.INT24,
    SampleFormat.FLOAT32,
]


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_preserves_the_signal(fmt: SampleFormat) -> None:
    x = (np.sin(np.linspace(0, 20, 500)) * 0.8).astype(np.float32)
    got = decode_pcm(encode_pcm(x, fmt), fmt, channels=1)
    # 8-bit formats quantise coarsely; everything else is far finer. The encode/decode
    # scale is asymmetric (encode by N-1, decode by N), so round-trip error grows with
    # |x|: bound ~= 0.5/scale_dec + |x|/scale_dec. At this test's amplitude (0.8) that
    # is ~0.0102 for 8-bit and ~0.00004 for 16-bit, so the tolerances below carry
    # margin above those analytic worst cases rather than the coarser round numbers.
    tol = 1 / 80.0 if fmt in (SampleFormat.UINT8, SampleFormat.INT8) else 1 / 20000.0
    assert got.shape == x.shape
    assert np.max(np.abs(got - x)) < tol


def test_uint8_silence_is_the_128_bias_not_full_scale_dc() -> None:
    """unsigned 8-bit PCM centres on 128. Decoding it as signed turns silence into
    full-scale DC."""
    silence = bytes([128] * 64)
    assert np.max(np.abs(decode_pcm(silence, SampleFormat.UINT8, channels=1))) == 0.0


def test_int24_is_sign_extended() -> None:
    # -1 in 3-byte little-endian two's complement.
    data = b"\xff\xff\xff" * 8
    got = decode_pcm(data, SampleFormat.INT24, channels=1)
    assert np.all(got < 0.0)
    assert np.allclose(got, -1.0 / (1 << 23), atol=1e-9)


def test_multichannel_is_deinterleaved_not_downmixed() -> None:
    interleaved = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    encoded = encode_pcm(interleaved, SampleFormat.FLOAT32)
    got = decode_pcm(encoded, SampleFormat.FLOAT32, channels=2)
    assert got.shape == (2, 2)
    assert np.allclose(got[:, 0], [0.5, 0.25])
    assert np.allclose(got[:, 1], [-0.5, -0.25])


@pytest.mark.parametrize("fmt", FORMATS)
def test_encode_saturates_instead_of_wrapping(fmt: SampleFormat) -> None:
    """Resampling overshoots past the original peak (Gibbs). A wrapping cast turns
    that overshoot into a full-scale sign flip, i.e. a click."""
    over = np.array([1.9, -1.9, 3.0, -3.0], dtype=np.float32)
    got = decode_pcm(encode_pcm(over, fmt), fmt, channels=1)
    assert np.all(got[[0, 2]] > 0.9), f"{fmt}: positive overshoot wrapped"
    assert np.all(got[[1, 3]] < -0.9), f"{fmt}: negative overshoot wrapped"


def test_unsupported_format_raises() -> None:
    with pytest.raises(ValueError):
        decode_pcm(b"\x00\x00", SampleFormat.INVALID, channels=1)
    with pytest.raises(ValueError):
        encode_pcm(np.zeros(2, dtype=np.float32), SampleFormat.INVALID)
