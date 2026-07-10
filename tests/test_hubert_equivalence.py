"""ONNX 版 HuBERT の数値等価ゲート。

fp32 グラフ: fairseq 時代に scripts/convert_hubert.py が捕獲した特徴量（fp32）を正解とし、
(9, use_final_proj=True) と (12, use_final_proj=False) の両方を厳密に照合する。

fp16 グラフ: 参照は **fp32 golden ではなく torch fp16**（置き換え対象の実装）。半精度の
絶対誤差は hidden state のスケール (O(1)-O(2.5)) に対して 1e-1 オーダーで、現行 runtime の
HubertModel.half() 自身が fp32 golden 比 cosine 0.987 / max_abs 0.435 を出す。したがって
fp32 golden を fp16 の参照にすること自体が誤り。問うべきは「ONNX 化で fp16 の振る舞いが
変わっていないか」であり、参照は scripts/export_hubert_onnx.py が捕獲した
hubert_golden_fp16.npz。GPU 依存の参照なので CUDA gating 済みの開発機でのみ走る。

資産と golden は派生物なので gitignore してある。環境変数が未設定なら skip し、
CPU/CI のスイートを壊さない（tests/test_change_voice_golden.py と同じ流儀）。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import COSINE_MIN_FP16
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import MAX_ABS_MAX_FP16
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

_ASSET_ENV = "VSPEECH_HUBERT_ASSET_DIR"
_GOLDEN_ENV = "VSPEECH_HUBERT_GOLDEN_DIR"

_asset = os.environ.get(_ASSET_ENV)
_golden = os.environ.get(_GOLDEN_ENV)
ASSET_DIR = Path(_asset) if _asset else None
GOLDEN_NPZ = Path(_golden) / "hubert_golden.npz" if _golden else None
GOLDEN_FP16_NPZ = Path(_golden) / "hubert_golden_fp16.npz" if _golden else None

# しきい値 (COSINE_MIN / MAX_ABS_MAX / *_FP16) の単一情報源は scripts/hubert_metrics.py。
# 緩めるときはそこで変更し、実測値を根拠としてコメントに残すこと（実測の 10 倍まで）。
pytestmark = pytest.mark.skipif(
    ASSET_DIR is None
    or not ASSET_DIR.exists()
    or GOLDEN_NPZ is None
    or not GOLDEN_NPZ.exists(),
    reason=f"${_ASSET_ENV} / ${_GOLDEN_ENV} not available",
)

CASES = [(9, True, "l9_proj"), (12, False, "l12_raw")]


def _compare(device: torch.device, is_half: bool, case) -> tuple[float, float]:
    """`is_half` は判定に使う参照 npz も選ぶ。fp16 の参照は torch fp16。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    emb_output_layer, use_final_proj, golden_key = case
    assert ASSET_DIR is not None and GOLDEN_NPZ is not None  # skipif guarantees
    assert GOLDEN_FP16_NPZ is not None

    data = np.load(GOLDEN_FP16_NPZ if is_half else GOLDEN_NPZ)
    wav = np.load(GOLDEN_NPZ)["wav"].astype(np.float32)
    reference = data[golden_key].astype(np.float32)

    model = load_hubert_model(ASSET_DIR, device, is_half=is_half)
    assert model.is_half == is_half, "期待した精度のグラフが選ばれていない"

    out = extract_features(
        model,
        torch.from_numpy(wav).unsqueeze(0),
        device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    candidate = out.squeeze(0).float().cpu().numpy()
    assert candidate.shape == reference.shape, f"{candidate.shape} vs {reference.shape}"
    return feature_cosine(candidate, reference), feature_max_abs_diff(
        candidate, reference
    )


@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp32_features_match_fairseq_golden(
    emb_output_layer, use_final_proj, golden_key
):
    cosine, max_abs = _compare(
        torch.device("cpu"), False, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN, f"cosine {cosine:.8f} < {COSINE_MIN}"
    assert max_abs <= MAX_ABS_MAX, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX:.1e}"


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or GOLDEN_FP16_NPZ is None
    or not GOLDEN_FP16_NPZ.exists(),
    reason="fp16 graph needs CUDA and hubert_golden_fp16.npz",
)
@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp16_features_match_the_torch_fp16_reference(
    emb_output_layer, use_final_proj, golden_key
):
    """ONNX 化で fp16 の振る舞いが変わっていないこと。fp32 golden とは比べない。"""
    cosine, max_abs = _compare(
        torch.device("cuda", 0), True, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN_FP16, f"cosine {cosine:.8f} < {COSINE_MIN_FP16}"
    assert max_abs <= MAX_ABS_MAX_FP16, (
        f"max-abs {max_abs:.3e} > {MAX_ABS_MAX_FP16:.1e}"
    )
