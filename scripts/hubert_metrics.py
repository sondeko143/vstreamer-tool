"""HuBERT 置換の等価判定に使うメトリクス（純関数）。

変換ツール（scripts/convert_hubert.py）と等価テスト（tests/test_hubert_equivalence.py、
tests/test_change_voice_golden.py）が同じ数式・同じしきい値で合否を出せるよう、判定
ロジックとしきい値をここに一本化する。値をコピーせず必ず import すること。

設計方針: いずれの指標も **壊れた入力に「満点」を返してはならない**（fail-closed）。
"""

import math

import numpy as np
from numpy.typing import NDArray

# --- 合格しきい値（単一情報源） ---
# HuBERT 特徴量の等価判定（fp32）。
COSINE_MIN = 0.9999
MAX_ABS_MAX = 1e-4
# change_voice 出力音声の回帰判定。
CORR_MIN = 0.999
SNR_MIN_DB = 40.0
# fp16 ONNX グラフ vs **torch fp16 参照**（fp32 golden ではない）。
# hidden state は O(1)-O(2.5) あり、半精度の絶対誤差はもともと 1e-1 オーダー。fp32 golden に
# 対しては現行 runtime の HubertModel.half() 自身が cosine 0.987 / max_abs 0.435 を出すので、
# fp32 golden を fp16 の参照にすること自体が誤り。問うべきは「ONNX 化で fp16 の振る舞いが
# 変わっていないか」であり、参照は置き換え対象の torch fp16 である。
# 実測 (2026-07-10, RTX 4060, ONNX fp16 vs torch fp16):
#   l9_proj  cosine=0.99999010 max_abs=1.379e-02
#   l12_raw  cosine=0.99997235 max_abs=1.074e-02
COSINE_MIN_FP16 = 0.9999
MAX_ABS_MAX_FP16 = 5e-2


def _as_2d(x: NDArray) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1, arr.shape[-1])


def feature_cosine(a: NDArray, b: NDArray) -> float:
    """最終軸を特徴次元とみなしたフレーム毎 cosine 類似度の平均。

    **両方**のノルムが 0 のフレームだけを一致 (1.0) とみなす。片方だけが 0 のフレームは
    不一致 (0.0)。ゼロノルムのフレームを一律除外すると `feature_cosine(非ゼロ, 全ゼロ)` が
    1.0 を返し、主ゲートがゴミに満点を出してしまう。
    """
    if np.asarray(a).shape != np.asarray(b).shape:
        raise ValueError(
            f"shape mismatch: {np.asarray(a).shape} vs {np.asarray(b).shape}"
        )
    x = _as_2d(a)
    y = _as_2d(b)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    den = norm_x * norm_y
    cosine = np.zeros(den.shape, dtype=np.float64)
    usable = den > 0.0
    cosine[usable] = (x[usable] * y[usable]).sum(axis=-1) / den[usable]
    cosine[(norm_x == 0.0) & (norm_y == 0.0)] = 1.0
    if cosine.size == 0:
        return 1.0
    return float(cosine.mean())


def feature_max_abs_diff(a: NDArray, b: NDArray) -> float:
    """要素毎の絶対差の最大値。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    return float(np.abs(x - y).max())


def waveform_correlation(a: NDArray, b: NDArray) -> float:
    """零ラグの正規化相互相関（-1..1）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den == 0.0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float((x * y).sum() / den)


def waveform_snr(reference: NDArray, test: NDArray) -> float:
    """`reference` に対する `test` の全体 SNR(dB)。`10*log10(Σref² / Σnoise²)`。

    **フレーム分割も中央値も使わない。** セグメンタル中央値版を廃した理由: 中央値は外れ値に
    頑健なので少数フレームの破損に原理的に鈍感で、回帰検出という目的に寄与しない。一方で
    フレーム分割 + マスク + 中央値の組み合わせは「壊れた信号に inf（完璧）を返す」経路を
    5 通り生んだ（tiny 除算 overflow / 中央値の上方飽和 / 無音参照フレームの除外 /
    `NaN > 0` が False / 末尾端数の破棄）。局所破損は相関ゲートが捕らえる。

    戻り値:
      - 完全一致（noise == 0）→ `inf`（両方とも無音の場合も含む）
      - 参照が全無音（signal == 0）でテストにエネルギーがある → `-inf`（破損）
      - それ以外 → 有限の dB

    非有限入力（NaN / inf）は破損なので ValueError。エネルギー和が float64 で overflow した
    場合も ValueError（inf を「完璧」と誤報告させない）。
    商を取らず log 空間で引くため、極端なダイナミックレンジでも比が overflow しない。
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    tst = np.asarray(test, dtype=np.float64).ravel()
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(tst).all():
        raise ValueError("waveform_snr: inputs must be finite (got NaN or inf)")
    # errstate: overflow は下の isfinite で ValueError にするので警告は出させない。
    with np.errstate(over="ignore"):
        signal = float((ref**2).sum())
        noise = float(((ref - tst) ** 2).sum())
    if not math.isfinite(signal) or not math.isfinite(noise):
        raise ValueError("waveform_snr: energy overflowed float64; rescale the inputs")
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * (math.log10(signal) - math.log10(noise))
