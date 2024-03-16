from pathlib import Path
from typing import Any
from typing import Optional
from typing import Tuple
from typing import cast

import numpy as np
import pyworld
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from scipy import signal
from torch import Tensor
from torch import cuda

from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig


class PitchExtractor:
    pass


def prepare_pitch_extractor(config: RvcConfig):
    if config.f0_extractor_type == F0ExtractorType.harvest:
        pass
    elif config.f0_extractor_type == F0ExtractorType.dio:
        pass
    elif config.f0_extractor_type == F0ExtractorType.crepe:
        providers = ["CPUExecutionProvider"]
        providers_options: list[dict[str, Any]] = [{}]
        if cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")
            providers_options.insert(0, {"device_id": config.gpu_id})
        return InferenceSession(
            str(config.crepe_model_file.expanduser()),
            providers=providers,
            provider_options=providers_options,
        )


def create_crepe_session(model_file: Path, gpu_id: int):
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(0, {"device_id": gpu_id})
    return InferenceSession(
        str(model_file.expanduser()),
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


def pitch_extract_crepe(
    audio: Tensor,
    f0_max: int,
    f0_min: int,
    sr: int,
    session: InferenceSession,
    precision: float = 10.0,
):
    from vspeech.lib import onnxcrepe

    audio_num = audio.cpu()
    onnx_f0, onnx_pd = onnxcrepe.predict(
        session,
        audio_num,
        sr,
        precision=precision,
        fmin=f0_min,
        fmax=f0_max,
        batch_size=256,
        return_periodicity=True,
        decoder=onnxcrepe.decode.weighted_argmax,
    )

    f0 = onnxcrepe.filter.median(onnx_f0, 3)
    pd = onnxcrepe.filter.median(onnx_pd, 3)

    f0[pd < 0.1] = 0
    return cast(NDArray[np.double], f0.squeeze())


def pitch_extract(
    audio: Tensor,
    f0_up_key: int,
    sr: int,
    window: int,
    f0_extractor: F0ExtractorType,
    crepe_session: Optional[InferenceSession],
    silence_front: int = 0,
) -> Tuple[NDArray[Any], NDArray[np.floating[Any]]]:
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
    elif f0_extractor == F0ExtractorType.crepe:
        if not crepe_session:
            raise ValueError("Crepe onnx session is not provided.")
        f0 = pitch_extract_crepe(
            audio, f0_max=f0_max, f0_min=f0_min, sr=sr, session=crepe_session
        )

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
