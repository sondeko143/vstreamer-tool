"""transformers 版 HuBERT が fairseq 版と数値等価であることの主ゲート。

fairseq 時代に scripts/convert_hubert.py が捕獲した特徴量（fp32）を正解とし、
(9, use_final_proj=True) と (12, use_final_proj=False) の両方で照合する。

資産と golden は派生物なので gitignore してある。環境変数が未設定なら skip し、
CPU/CI のスイートを壊さない（tests/test_change_voice_golden.py と同じ流儀）。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

_ASSET_ENV = "VSPEECH_HUBERT_ASSET_DIR"
_GOLDEN_ENV = "VSPEECH_HUBERT_GOLDEN_DIR"

_asset = os.environ.get(_ASSET_ENV)
_golden = os.environ.get(_GOLDEN_ENV)
ASSET_DIR = Path(_asset) if _asset else None
GOLDEN_NPZ = Path(_golden) / "hubert_golden.npz" if _golden else None

# しきい値 (COSINE_MIN / MAX_ABS_MAX) の単一情報源は scripts/hubert_metrics.py。
# 緩めるときはそこで変更し、実測値を根拠としてコメントに残すこと（実測の 10 倍まで）。
pytestmark = pytest.mark.skipif(
    ASSET_DIR is None
    or not ASSET_DIR.exists()
    or GOLDEN_NPZ is None
    or not GOLDEN_NPZ.exists(),
    reason=f"${_ASSET_ENV} / ${_GOLDEN_ENV} not available",
)


@pytest.mark.parametrize(
    ("emb_output_layer", "use_final_proj", "golden_key"),
    [(9, True, "l9_proj"), (12, False, "l12_raw")],
)
def test_features_match_fairseq_golden(emb_output_layer, use_final_proj, golden_key):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    assert (
        ASSET_DIR is not None and GOLDEN_NPZ is not None
    )  # skipif guarantees; narrows for ty

    data = np.load(GOLDEN_NPZ)
    wav = data["wav"].astype(np.float32)
    reference = data[golden_key].astype(np.float32)

    device = torch.device("cpu")
    bundle = load_hubert_model(ASSET_DIR, device, is_half=False)
    out = extract_features(
        bundle,
        torch.from_numpy(wav).unsqueeze(0),
        device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    candidate = out.squeeze(0).float().cpu().numpy()

    assert candidate.shape == reference.shape, f"{candidate.shape} vs {reference.shape}"
    cosine = feature_cosine(candidate, reference)
    max_abs = feature_max_abs_diff(candidate, reference)
    assert cosine >= COSINE_MIN, f"cosine {cosine:.8f} < {COSINE_MIN}"
    assert max_abs <= MAX_ABS_MAX, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX:.1e}"
