"""Metrics used to judge the equivalence of the HuBERT replacement, and the verdict.

The decision logic and the thresholds live here alone, so that the conversion tool
(scripts/convert_hubert.py) and the equivalence tests (tests/test_hubert_equivalence.py,
tests/test_change_voice_golden.py) reach a verdict from the same formulas and the same
thresholds. Always import the values; never copy them.

Design rule: no metric may ever return a "perfect score" for broken input (fail-closed).

The metrics are pure; `should_write_assets` is not (it is the exit-code contract of
scripts/export_hubert_onnx.py). It lives here rather than next to its caller because this
module imports no torch, and the export script does -- at module level, for the wrapper's
base class. With torch gone from the dependency table (ADR-0080) an environment that can
run the test suite cannot import the export script at all, so keeping the verdict there
left the two tests that pin it uncollectable, and the exit-code inversion they guard had
already shipped once.
"""

import math

import numpy as np
from numpy.typing import NDArray

# --- Pass thresholds (single source of truth) ---
# Equivalence of the HuBERT features (fp32).
COSINE_MIN = 0.9999
MAX_ABS_MAX = 1e-4
# Regression check on the change_voice output audio. The golden comes **from fairseq** and
# was captured with an fp16 HuBERT, so this gate is sensitive to HuBERT's execution engine.
# Measured (2026-07-10, RTX 5060, a real RVC model):
#   transformers fp16 (spec 1)  corr 0.99998675  SNR 44.59 dB
#   ONNX fp16 (spec 2)          corr 0.99995400  SNR 39.52 dB
# The difference is rounding between ORT's and torch's fp16 kernels, not a defect in the
# graph. Running the same ONNX in fp32 reproduces fairseq's features to max_abs 1.010e-05
# (tests/test_hubert_equivalence.py).
# Sensitivity cross-check: the fp16-vs-fp32 feature difference of max_abs 0.43 drops SNR to
# 3.03 dB. For an error ratio of about 30x, 20*log10(30) ~ 29.5 dB, which is consistent
# with the 39.52 -> 3.03 drop.
# The basis for relaxing 40.0 -> 35.0 is the 39.52 dB measured above (about 4.5 dB of
# margin -- the same margin spec 1 had at 44.59/40).
# The golden is not re-baselined: we do not want to lose the guarantee that spans the
# fairseq implementation here.
CORR_MIN = 0.999
SNR_MIN_DB = 35.0
# The fp16 ONNX graph vs **a torch fp16 reference** (not the fp32 golden).
# Hidden states are O(1)-O(2.5) and half precision's absolute error is inherently on the
# order of 1e-1. Against the fp32 golden, the current runtime's own HubertModel.half()
# produces cosine 0.987 / max_abs 0.435, so using the fp32 golden as an fp16 reference is
# itself wrong. The question to ask is "did going to ONNX change the fp16 behaviour", and
# the reference for that is the torch fp16 being replaced.
# Measured (2026-07-10, RTX 4060, ONNX fp16 vs torch fp16):
#   l9_proj  cosine=0.99999010 max_abs=1.379e-02
#   l12_raw  cosine=0.99997235 max_abs=1.074e-02
COSINE_MIN_FP16 = 0.9999
MAX_ABS_MAX_FP16 = 5e-2


def should_write_assets(ok: bool, measure_only: bool) -> bool:
    """Judge the run, then decide whether the assets get written. Never the reverse.

    `--measure-only` decides *what is written*; it must not decide *what the exit code
    means*. Honouring it first -- which is what this used to do -- made a failing
    equivalence gate exit 0, so a `--measure-only` run reported success while printing
    FAIL lines nobody was required to read. The gate is judged for both modes here, and
    the caller writes nothing unless this returns True.
    """
    if not ok:
        raise SystemExit("等価ゲートに落ちました。資産は書き出しません。")
    if measure_only:
        print("--measure-only: 資産は更新していません")
        return False
    return True


def _as_2d(x: NDArray) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1, arr.shape[-1])


def feature_cosine(a: NDArray, b: NDArray) -> float:
    """The mean per-frame cosine similarity, treating the last axis as the feature
    dimension.

    Only frames where **both** norms are 0 count as a match (1.0). A frame where just one
    is 0 counts as a mismatch (0.0). Excluding zero-norm frames wholesale would make
    `feature_cosine(nonzero, all zeros)` return 1.0, i.e. the main gate would award a
    perfect score to garbage.
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
    """The maximum elementwise absolute difference."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    return float(np.abs(x - y).max())


def waveform_correlation(a: NDArray, b: NDArray) -> float:
    """Zero-lag normalized cross-correlation (-1..1)."""
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
    """Overall SNR (dB) of `test` against `reference`: `10*log10(sum(ref^2) /
    sum(noise^2))`.

    **No framing and no median.** The segmental-median version was dropped because a
    median is robust to outliers and therefore inherently insensitive to corruption in a
    handful of frames, which does not serve regression detection. Meanwhile the
    framing + mask + median combination created five separate paths that "return inf
    (perfect) for a broken signal" (tiny-divisor overflow / upward saturation of the
    median / excluding silent reference frames / `NaN > 0` being False / discarding the
    trailing partial frame). Local corruption is caught by the correlation gate.

    Return value:
      - exact match (noise == 0) -> `inf` (including when both are silent)
      - the reference is entirely silent (signal == 0) while the test has energy ->
        `-inf` (corrupt)
      - otherwise -> a finite dB value

    Non-finite input (NaN / inf) is corruption, hence ValueError. An energy sum
    overflowing float64 is also a ValueError (never let inf be misreported as "perfect").
    The subtraction happens in log space rather than as a quotient, so the ratio cannot
    overflow even with extreme dynamic range.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    tst = np.asarray(test, dtype=np.float64).ravel()
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(tst).all():
        raise ValueError("waveform_snr: inputs must be finite (got NaN or inf)")
    # errstate: overflow is turned into a ValueError by the isfinite check below, so
    # suppress the warning here.
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
