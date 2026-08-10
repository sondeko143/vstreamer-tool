"""Format-aware PCM decode/encode shared by every device boundary (ADR-0070).

Dispatch is keyed on SampleFormat, NOT byte width: UINT8 and INT8 share a width but
differ in sign and bias, so a width-keyed table would decode unsigned-8 as signed and
skip its 128 offset (silence -> full-scale DC).

This started as transcription.py's private `_pcm_to_float32_mono`. It is shared now
because recording and playback need the same decode, and they must keep the channel
layout rather than downmix.
"""

import numpy as np
from numpy.typing import NDArray

from vspeech.config import SampleFormat

_INT24_SIGN_BIT = 0x800000
_INT24_SCALE = float(1 << 23)


def decode_pcm(data: bytes, format: SampleFormat, channels: int) -> NDArray[np.float32]:
    """Decode interleaved PCM bytes into float32.

    Integer formats (UINT8/INT8/INT16/INT24) land in [-1, 1]. FLOAT32 input is
    passed through unbounded -- clamping here would alter the signal before it
    reaches the resampler; encode_pcm is where the [-1, 1] bound is enforced, on
    the way back out.

    Returns `(frames,)` for mono and `(frames, channels)` otherwise -- the shape
    PolyphaseResampler.process expects.
    """
    if channels < 1:
        raise ValueError(f"channels は 1 以上を指定してください: {channels!r}")
    if format == SampleFormat.FLOAT32:
        samples = np.frombuffer(data, dtype=np.float32).astype(np.float32)
    elif format == SampleFormat.UINT8:
        # unsigned 8-bit PCM is biased by 128 (128 == silence).
        samples = (
            np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif format == SampleFormat.INT8:
        samples = np.frombuffer(data, dtype=np.int8).astype(np.float32) / 128.0
    elif format == SampleFormat.INT16:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif format == SampleFormat.INT24:
        # 3-byte little-endian signed PCM -> sign-extended int32 -> [-1, 1).
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        as32 = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        as32 = (as32 ^ _INT24_SIGN_BIT) - _INT24_SIGN_BIT
        samples = as32.astype(np.float32) / _INT24_SCALE
    else:
        raise ValueError(f"未対応の PCM フォーマットです: {format!r}")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return np.ascontiguousarray(samples, dtype=np.float32)


def encode_pcm(x: NDArray[np.float32], format: SampleFormat) -> bytes:
    """Encode float32 (mono `(n,)` or interleavable `(n, channels)`) back to PCM.

    Always saturates at full scale. Resampling overshoots past the original peak
    (Gibbs), and a wrapping cast turns that overshoot into a sign flip -- an audible
    click. Never replace this with a bare `.astype(np.int16)`.

    NaN input encodes as silence, not as full scale. NaN survives np.clip unchanged,
    and casting NaN to an integer dtype is undefined -- measured: an unguarded UINT8
    cast lands the byte on 0x00, which decode_pcm reads back as full-scale -1.0 DC,
    exactly the failure mode this module exists to avoid.
    """
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    # Coerce NaN to 0.0 before clipping, for every format including FLOAT32. +-inf
    # is already handled correctly by np.clip below, so NaN is the only case left.
    flat = np.nan_to_num(flat, nan=0.0)
    if format == SampleFormat.FLOAT32:
        # float32 output still gets clipped: PortAudio would clip it anyway, and
        # leaving it unbounded makes the boundary behave differently per format.
        return np.clip(flat, -1.0, 1.0).astype(np.float32).tobytes()
    clipped = np.clip(flat, -1.0, 1.0)
    if format == SampleFormat.UINT8:
        return (np.rint(clipped * 127.0) + 128.0).astype(np.uint8).tobytes()
    if format == SampleFormat.INT8:
        return np.rint(clipped * 127.0).astype(np.int8).tobytes()
    if format == SampleFormat.INT16:
        return np.rint(clipped * 32767.0).astype(np.int16).tobytes()
    if format == SampleFormat.INT24:
        as32 = np.rint(clipped * (_INT24_SCALE - 1.0)).astype(np.int32)
        packed = np.empty((as32.size, 3), dtype=np.uint8)
        packed[:, 0] = as32 & 0xFF
        packed[:, 1] = (as32 >> 8) & 0xFF
        packed[:, 2] = (as32 >> 16) & 0xFF
        return packed.tobytes()
    raise ValueError(f"未対応の PCM フォーマットです: {format!r}")
