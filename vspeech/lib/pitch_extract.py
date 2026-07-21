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


def _pyworld():
    """pyworld を遅延 import する。cp314 wheel が無く runtime 依存から外したため、
    dio/harvest を選んだときだけ必要。既定の rmvpe では読み込まない。"""
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
    # 単一フレーム (T=1) では squeeze が 0-d に潰れ、呼び出し側の f0[:p_len] が
    # IndexError になる。atleast_1d で 1-D を保証する。
    return cast(NDArray[np.double], np.atleast_1d(onnx_f0.squeeze()))


# 焼き込んだ reflect-pad (432) は N>=433 (約27ms @16kHz) を要求する。短い入力での
# ONNXRuntime クラッシュを避けるためこの最小長まで左ゼロパッドで底上げする。
# scripts/export_fcpe_onnx.py の FLOOR と同値 (どちらも win_size-hop の pad から導かれる)。
# mel config を変えて再 export するときは両方を見直すこと。
FCPE_MIN_SAMPLES = 433


def pitch_extract_fcpe(
    audio: Tensor,
    session: InferenceSession,
) -> NDArray[np.double]:
    """FCPE onnx から f0 を取る。

    export 時に threshold / sample_rate / decoder_mode を焼き込んであるので、runtime の
    入力は 16kHz mono 波形 (batched ``(1, N)``) のみ。出力は f0 (Hz)。FCPE の閾値デコード
    (threshold=0.006) が無声フレームを 0 にする。

    `.infer()` の f0_min/f0_max 後処理は焼き込まない = rmvpe.onnx と同じ「mel -> net ->
    閾値 voicing -> f0」契約に揃える (rmvpe も pitch_extract_rmvpe で生の f0 を返す)。この
    対称性が forward-only を安全とみなす根拠。閾値や後処理を変えたいときは export し直す。

    N < FCPE_MIN_SAMPLES は焼き込んだ reflect-pad が要求する最小長に満たず onnx が落ちる
    ので左ゼロパッドで底上げする (実際の vc 経路は _quality_padding で十分長いので通常は
    発生しない防御)。
    """
    audio_np = audio.detach().cpu().numpy().astype(np.float32)
    if audio_np.shape[-1] < FCPE_MIN_SAMPLES:
        audio_np = np.pad(audio_np, (FCPE_MIN_SAMPLES - audio_np.shape[-1], 0))
    audio_num = np.expand_dims(audio_np, axis=0)
    onnx_f0 = cast(
        NDArray[np.float32],
        session.run(None, {"waveform": audio_num})[0],
    )
    # FCPE の decode は完全無声フレームで NaN (0/0) を出しうる。新しい export はグラフ内で
    # 0 に潰すが、古い/別の fcpe.onnx から NaN が来ても RVC の NSF (pitchf) に漏らさない
    # よう runtime でも 0 に潰す (無声=0 で rmvpe と同契約)。
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
    # >= p_len frames, so the zero-padded right-aligned `pitchf` that used to be
    # built here was dead code (computed, never returned) and identical to
    # f0bak[:p_len] for every real input.
    f0bak = f0.copy()
    f0_mel = 1127.0 * np.log(1.0 + f0bak / 700.0)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
    )
    f0_coarse = np.rint(f0_mel).astype(int)

    return f0_coarse, f0bak
