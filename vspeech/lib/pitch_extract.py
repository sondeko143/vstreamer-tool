from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from scipy import signal
from torch import Tensor

from vspeech.config import F0ExtractorType


class PitchExtractor:
    pass


RMVPE_THRESHOLD = 0.3


def median_filter_f0(
    f0: NDArray[np.floating[Any]], radius: int
) -> NDArray[np.floating[Any]]:
    """Median-filter f0 inside each voiced run (RVC's `filter_radius`, ADR-0070).

    Kills the isolated single-frame octave errors rmvpe/fcpe emit. Extraction is
    block-wise with no continuity constraint across blocks, so one bad frame otherwise
    reaches the NSF unopposed and rings as a short artefact.

    Unvoiced frames (0) never enter a window and are returned untouched. A window
    spanning the 0 boundary would drag the run's first voiced frame toward min(),
    blunting voiced onsets more audibly than the artefact being removed -- so each
    maximal run of f0 > 0 is filtered on its own, with run borders padded by edge
    replication rather than zeros.

    Edge replication also makes the array's final frame an identity (the replicated
    copies are a strict majority of the window), so this filter needs no lookahead and
    adds no latency. In the streaming path the emitted region ends about 3 frames before
    the array end, so every emitted frame still gets a genuine two-sided window at
    radius <= 3; beyond that the tail degrades to unfiltered rather than wrong.
    """
    if radius <= 0:
        return f0
    kernel = 2 * radius + 1
    out = f0.copy()
    # Run boundaries from the transitions of the voiced mask, bracketed by False so a run
    # touching either end is closed. np.diff on a bool array is XOR, so the flat indices
    # come out as alternating (start, stop) pairs.
    voiced = f0 > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([False], voiced, [False]))))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        padded = np.pad(f0[start:stop], radius, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        out[start:stop] = np.median(windows, axis=1)
    return out


def _pyworld():
    """Import pyworld lazily. It was dropped from the runtime dependencies because it
    has no cp314 wheel, so it is only needed when dio/harvest is selected. The default
    rmvpe never loads it."""
    try:
        import pyworld  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "f0_extractor_type 'dio'/'harvest' には optional な 'pyworld' が必要です "
            "(cp314 wheel 無し; 手動導入してください: `uv pip install pyworld`)。"
            "既定の 'rmvpe' はこれを必要としません。"
        ) from e
    return pyworld


def pitch_extract_harvest(
    audio: NDArray[np.float32],
    f0_max: int,
    sr: int,
) -> NDArray[np.double]:
    pyworld = _pyworld()
    f0_, t = pyworld.harvest(
        audio.astype(np.double),
        fs=sr,
        f0_ceil=f0_max,
        frame_period=10,
    )
    f0 = cast(
        NDArray[np.double],
        pyworld.stonemask(audio.astype(np.double), f0_, t, sr),
    )
    return signal.medfilt(f0, 3)


def pitch_extract_dio(
    audio: NDArray[np.float32],
    f0_max: int,
    f0_min: int,
    sr: int,
):
    pyworld = _pyworld()
    f0_, t = pyworld.dio(
        audio.astype(np.double),
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        channels_in_octave=2,
        frame_period=10,
    )
    return cast(
        NDArray[np.double],
        pyworld.stonemask(audio.astype(np.double), f0_, t, sr),
    )


def pitch_extract_rmvpe(
    audio: Tensor,
    session: InferenceSession,
    threshold: float = RMVPE_THRESHOLD,
) -> NDArray[np.double]:
    """Extract f0 with an rmvpe.onnx model (VCClient-style export).

    The ONNX graph bundles mel extraction, the E2E network, and threshold-based
    voicing/decoding, so it consumes the raw 16kHz mono waveform (batched as
    ``(1, N)``) plus a voicing ``threshold`` and emits f0 in Hz with unvoiced
    frames zeroed as its first output.

    The f0 output name differs across exports (``f0`` for the yxlllc/RMVPE
    ``export.py``, ``pitchf`` for the w-okada re-export), so we request all
    outputs and read index 0 instead of hard-coding a name. The yxlllc export's
    second ``uv`` output is unused — index 0 is already threshold-masked.
    """
    audio_num = audio.detach().cpu().numpy().astype(np.float32)
    audio_num = np.expand_dims(audio_num, axis=0)
    onnx_f0 = cast(
        NDArray[np.float32],
        session.run(
            None,
            {
                "waveform": audio_num,
                "threshold": np.array([threshold], dtype=np.float32),
            },
        )[0],
    )
    # For a single frame (T=1) squeeze collapses to 0-d and the caller's f0[:p_len]
    # raises IndexError. atleast_1d guarantees 1-D.
    return cast(NDArray[np.double], np.atleast_1d(onnx_f0.squeeze()))


# The baked-in reflect-pad (432) requires N>=433 (about 27ms @16kHz). Left-zero-pad up
# to this minimum length to avoid an ONNXRuntime crash on short input. Same value as
# FLOOR in scripts/export_fcpe_onnx.py (both derive from the win_size-hop pad). Revisit
# both when re-exporting with a different mel config.
FCPE_MIN_SAMPLES = 433


def pitch_extract_fcpe(
    audio: Tensor,
    session: InferenceSession,
) -> NDArray[np.double]:
    """Extract f0 from the FCPE onnx.

    threshold / sample_rate / decoder_mode are baked in at export time, so the only
    runtime input is a 16kHz mono waveform (batched ``(1, N)``) and the output is f0
    (Hz). FCPE's threshold decode (threshold=0.006) zeroes unvoiced frames.

    `.infer()`'s f0_min/f0_max post-processing is deliberately not baked in, which
    aligns this with rmvpe.onnx's "mel -> net -> threshold voicing -> f0" contract
    (rmvpe also returns raw f0 from pitch_extract_rmvpe). That symmetry is what makes
    forward-only safe here. Re-export when you want to change the threshold or the
    post-processing.

    N < FCPE_MIN_SAMPLES is below the minimum length the baked-in reflect-pad requires
    and would crash onnx, so pad it up with left zeros (a defence that normally never
    fires: the real vc path is long enough thanks to _quality_padding).
    """
    audio_np = audio.detach().cpu().numpy().astype(np.float32)
    if audio_np.shape[-1] < FCPE_MIN_SAMPLES:
        audio_np = np.pad(audio_np, (FCPE_MIN_SAMPLES - audio_np.shape[-1], 0))
    audio_num = np.expand_dims(audio_np, axis=0)
    onnx_f0 = cast(
        NDArray[np.float32],
        session.run(None, {"waveform": audio_num})[0],
    )
    # FCPE's decode can produce NaN (0/0) on a fully unvoiced frame. Newer exports
    # collapse that to 0 inside the graph, but collapse it at runtime too so a NaN from
    # an older/foreign fcpe.onnx never leaks into RVC's NSF (pitchf) (unvoiced=0, the
    # same contract as rmvpe).
    f0 = np.nan_to_num(
        np.atleast_1d(onnx_f0.squeeze()), nan=0.0, posinf=0.0, neginf=0.0
    )
    return cast(NDArray[np.double], f0)


def pitch_extract(
    audio: Tensor,
    f0_up_key: int,
    sr: int,
    window: int,
    f0_extractor: F0ExtractorType,
    f0_session: InferenceSession | None,
    silence_front: int = 0,
) -> tuple[NDArray[Any], NDArray[np.floating[Any]]]:
    start_frame = int(silence_front * sr / window)
    real_silence_front = start_frame * window / sr

    silence_front_offset = int(np.round(real_silence_front * sr))
    audio = audio[silence_front_offset:]

    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    if f0_extractor == F0ExtractorType.dio:
        f0 = pitch_extract_dio(
            audio=audio.detach().cpu().numpy(), f0_max=f0_max, f0_min=f0_min, sr=sr
        )
    elif f0_extractor == F0ExtractorType.harvest:
        f0 = pitch_extract_harvest(
            audio=audio.detach().cpu().numpy(), f0_max=f0_max, sr=sr
        )
    elif f0_extractor == F0ExtractorType.rmvpe:
        if not f0_session:
            raise ValueError("RMVPE onnx session is not provided.")
        f0 = pitch_extract_rmvpe(audio, session=f0_session)
    elif f0_extractor == F0ExtractorType.fcpe:
        if not f0_session:
            raise ValueError("FCPE onnx session is not provided.")
        f0 = pitch_extract_fcpe(audio, session=f0_session)
    else:
        raise ValueError("unknown f0 extractor type")

    f0 *= pow(2, f0_up_key / 12)
    # f0 is returned raw (f0bak); the caller (_select_pitch) truncates it to
    # p_len and aligns it to the feature length. rmvpe/harvest/dio all return
    # >= p_len frames.
    f0bak = f0.copy()
    f0_mel = 1127.0 * np.log(1.0 + f0bak / 700.0)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
    )
    f0_coarse = np.rint(f0_mel).astype(int)

    return f0_coarse, f0bak
