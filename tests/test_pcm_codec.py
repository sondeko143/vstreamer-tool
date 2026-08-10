"""Shared PCM decode/encode used at every device boundary (ADR-0070)."""

import warnings

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

# decode_pcm's divisor for each integer format (encode uses divisor - 1, which is
# why round-trip error is amplitude-dependent -- see test_round_trip_preserves_the_signal).
_DECODE_SCALE = {
    SampleFormat.UINT8: 128.0,
    SampleFormat.INT8: 128.0,
    SampleFormat.INT16: 32768.0,
    SampleFormat.INT24: float(1 << 23),
}


def _round_trip_bound(fmt: SampleFormat, peak: float) -> float:
    """Analytic worst-case round-trip error for the asymmetric (encode by N-1,
    decode by N) quantisation scale: bound = (peak + 0.5) / N.

    A 1.5x margin absorbs float32's own rounding noise on top of the quantisation
    rounding -- most visible for INT24, whose 2**-23 step sits at the float32
    epsilon, so the two noise sources are comparable in size there.
    """
    return (peak + 0.5) / _DECODE_SCALE[fmt] * 1.5


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_preserves_the_signal(fmt: SampleFormat) -> None:
    x = (np.sin(np.linspace(0, 20, 500)) * 0.8).astype(np.float32)
    got = decode_pcm(encode_pcm(x, fmt), fmt, channels=1)
    assert got.shape == x.shape
    if fmt == SampleFormat.FLOAT32:
        # No quantisation step at all -- encode/decode is a bit-exact round trip.
        assert np.array_equal(got, x)
        return
    bound = _round_trip_bound(fmt, float(np.max(np.abs(x))))
    assert np.max(np.abs(got - x)) < bound


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


@pytest.mark.parametrize("fmt", FORMATS)
def test_multichannel_is_deinterleaved_not_downmixed(fmt: SampleFormat) -> None:
    """A channel-reshape bug specific to INT24 (which goes through an intermediate
    (-1, 3) byte reshape before the channel reshape) would not be caught by testing
    FLOAT32 alone, so this covers every format."""
    interleaved = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    encoded = encode_pcm(interleaved, fmt)
    got = decode_pcm(encoded, fmt, channels=2)
    assert got.shape == (2, 2)
    tol = 1e-9 if fmt == SampleFormat.FLOAT32 else _round_trip_bound(fmt, 0.5)
    assert np.allclose(got[:, 0], [0.5, 0.25], atol=tol)
    assert np.allclose(got[:, 1], [-0.5, -0.25], atol=tol)


@pytest.mark.parametrize("fmt", FORMATS)
def test_encode_saturates_instead_of_wrapping(fmt: SampleFormat) -> None:
    """Resampling overshoots past the original peak (Gibbs). A wrapping cast turns
    that overshoot into a full-scale sign flip, i.e. a click."""
    over = np.array([1.9, -1.9, 3.0, -3.0], dtype=np.float32)
    got = decode_pcm(encode_pcm(over, fmt), fmt, channels=1)
    assert np.all(got[[0, 2]] > 0.9), f"{fmt}: positive overshoot wrapped"
    assert np.all(got[[1, 3]] < -0.9), f"{fmt}: negative overshoot wrapped"
    if fmt == SampleFormat.FLOAT32:
        # FLOAT32 has no quantisation step of its own, so pin the clipped value
        # exactly -- otherwise a version with no FLOAT32 clip at all (returning the
        # unclipped overshoot unchanged) would pass this test identically.
        assert np.allclose(got, [1.0, -1.0, 1.0, -1.0])


@pytest.mark.parametrize("fmt", FORMATS)
def test_encode_nan_is_silence_not_full_scale(fmt: SampleFormat) -> None:
    """NaN survives np.clip unchanged, and casting NaN to an integer dtype is
    undefined. Measured against the unguarded implementation: UINT8 lands the byte
    on 0x00, which decode_pcm reads back as full-scale -1.0 DC -- exactly the
    failure mode this module exists to prevent. Silence is the only safe fallback
    at a device boundary.

    Two assertions are needed to discriminate every format against the unguarded
    implementation: on x86, an undefined NaN->int cast happens to land on 0 for
    INT8/INT16/INT24, which already decodes to silence -- only UINT8's cast lands
    on a byte that decodes to full-scale DC. So the decoded-value check alone only
    fails pre-fix for UINT8 (and, separately, for FLOAT32, which has no int cast to
    misbehave but does propagate the bare NaN). The unguarded cast still raises
    `RuntimeWarning: invalid value encountered in cast` for all three of
    INT8/INT16/INT24, though, so promoting that warning to an error here makes all
    five parametrizations fail against the unguarded implementation for a real
    reason, landing on 0 by undefined-behaviour luck rather than being a contract.
    """
    x = np.array([float("nan"), 0.3, float("nan"), -0.3], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = decode_pcm(encode_pcm(x, fmt), fmt, channels=1)
    assert got[0] == 0.0
    assert got[2] == 0.0


def test_channels_must_be_positive() -> None:
    data = encode_pcm(np.zeros(6, dtype=np.float32), SampleFormat.INT16)
    with pytest.raises(ValueError):
        decode_pcm(data, SampleFormat.INT16, channels=0)
    with pytest.raises(ValueError):
        decode_pcm(data, SampleFormat.INT16, channels=-1)


def test_unsupported_format_raises() -> None:
    with pytest.raises(ValueError):
        decode_pcm(b"\x00\x00", SampleFormat.INVALID, channels=1)
    with pytest.raises(ValueError):
        encode_pcm(np.zeros(2, dtype=np.float32), SampleFormat.INVALID)
