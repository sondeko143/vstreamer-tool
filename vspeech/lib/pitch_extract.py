from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pyworld
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from scipy import signal
from torch import Tensor
from torch import cuda

from vspeech.config import F0ExtractorType


class PitchExtractor:
    pass


RMVPE_THRESHOLD = 0.3


def create_rmvpe_session(model_file: Path, gpu_id: int):
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(
            0,
            {
                "device_id": gpu_id,
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        )
    return InferenceSession(
        str(model_file.expanduser()),
        sess_options=sess_options,
        providers=providers,
        provider_options=providers_options,
    )


def pitch_extract_harvest(
    audio: NDArray[np.float32],
    f0_max: int,
    sr: int,
) -> NDArray[np.double]:
    f0_, t = pyworld.harvest(  # type: ignore
        audio.astype(np.double),
        fs=sr,
        f0_ceil=f0_max,
        frame_period=10,
    )
    f0 = cast(
        NDArray[np.double],
        pyworld.stonemask(audio.astype(np.double), f0_, t, sr),  # type: ignore
    )
    return signal.medfilt(f0, 3)


def pitch_extract_dio(
    audio: NDArray[np.float32],
    f0_max: int,
    f0_min: int,
    sr: int,
):
    f0_, t = pyworld.dio(  # type: ignore
        audio.astype(np.double),
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        channels_in_octave=2,
        frame_period=10,
    )
    return cast(
        NDArray[np.double],
        pyworld.stonemask(audio.astype(np.double), f0_, t, sr),  # type: ignore
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
    return cast(NDArray[np.double], onnx_f0.squeeze())


def pitch_extract(
    audio: Tensor,
    f0_up_key: int,
    sr: int,
    window: int,
    f0_extractor: F0ExtractorType,
    rmvpe_session: InferenceSession | None,
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
        if not rmvpe_session:
            raise ValueError("RMVPE onnx session is not provided.")
        f0 = pitch_extract_rmvpe(audio, session=rmvpe_session)

    else:
        raise ValueError("unknown f0 extractor type")

    f0 *= pow(2, f0_up_key / 12)
    p_len = audio.shape[0] // window
    pitchf = np.zeros(p_len)
    pitchf[-f0.shape[0] :] = f0[: pitchf.shape[0]]
    f0bak = f0.copy()
    f0_mel = 1127.0 * np.log(1.0 + f0bak / 700.0)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
    )
    f0_coarse = np.rint(f0_mel).astype(int)

    return f0_coarse, f0bak
