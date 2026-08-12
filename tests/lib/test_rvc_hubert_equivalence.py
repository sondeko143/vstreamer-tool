"""The numeric equivalence gate of the ONNX HuBERT.

fp32 graph: the features captured by scripts/convert_hubert.py back in the fairseq era
(fp32) are the reference, and both (9, use_final_proj=True) and
(12, use_final_proj=False) are matched strictly.

fp16 graph: the reference is **torch fp16, not the fp32 golden** (the implementation being
replaced). Half precision's absolute error is on the order of 1e-1 relative to the scale
of the hidden states (O(1)-O(2.5)), and the current runtime's own HubertModel.half()
scores cosine 0.987 / max_abs 0.435 against the fp32 golden. Using the fp32 golden as an
fp16 reference is therefore wrong in itself. The question to ask is "did going to ONNX
change the fp16 behaviour", and the reference for that is the hubert_golden_fp16.npz
captured by scripts/export_hubert_onnx.py. Being a GPU-dependent reference, it runs only
on a CUDA-gated development machine.

The assets and goldens are derived artifacts and are gitignored, so the whole module is
gated on `requires_hubert_assets` (and the fp16 case additionally on `requires_cuda` /
`requires_hubert_fp16_golden`); see `uv run pytest --markers` (ADR-0089).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import COSINE_MIN_FP16
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import MAX_ABS_MAX_FP16
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff
from vspeech.lib.cuda_util import Device

_ASSET_ENV = "VSPEECH_HUBERT_ASSET_DIR"
_GOLDEN_ENV = "VSPEECH_HUBERT_GOLDEN_DIR"

_asset = os.environ.get(_ASSET_ENV)
_golden = os.environ.get(_GOLDEN_ENV)
ASSET_DIR = Path(_asset) if _asset else None
GOLDEN_NPZ = Path(_golden) / "hubert_golden.npz" if _golden else None
GOLDEN_FP16_NPZ = Path(_golden) / "hubert_golden_fp16.npz" if _golden else None

# scripts/hubert_metrics.py is the single source of truth for the thresholds
# (COSINE_MIN / MAX_ABS_MAX / *_FP16). To relax one, change it there and leave the
# measured value in a comment as the justification (up to 10x the measurement).
pytestmark = pytest.mark.requires_hubert_assets

CASES = [(9, True, "l9_proj"), (12, False, "l12_raw")]


def _compare(device: Device, is_half: bool, case) -> tuple[float, float]:
    """`is_half` also selects the reference npz used for the check. The fp16 reference is
    torch fp16."""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    emb_output_layer, use_final_proj, golden_key = case
    assert ASSET_DIR is not None and GOLDEN_NPZ is not None  # the gate guarantees
    assert GOLDEN_FP16_NPZ is not None

    data = np.load(GOLDEN_FP16_NPZ if is_half else GOLDEN_NPZ)
    wav = np.load(GOLDEN_NPZ)["wav"].astype(np.float32)
    reference = data[golden_key].astype(np.float32)

    model = load_hubert_model(ASSET_DIR, device, is_half=is_half)
    assert model.is_half == is_half, "期待した精度のグラフが選ばれていない"

    out = extract_features(
        model,
        wav.reshape(1, -1),
        device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    candidate = out[0].astype(np.float32)
    assert candidate.shape == reference.shape, f"{candidate.shape} vs {reference.shape}"
    return feature_cosine(candidate, reference), feature_max_abs_diff(
        candidate, reference
    )


@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp32_features_match_fairseq_golden(
    emb_output_layer, use_final_proj, golden_key
):
    cosine, max_abs = _compare(
        Device("cpu"), False, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN, f"cosine {cosine:.8f} < {COSINE_MIN}"
    assert max_abs <= MAX_ABS_MAX, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX:.1e}"


@pytest.mark.requires_cuda
@pytest.mark.requires_hubert_fp16_golden
@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp16_features_match_the_torch_fp16_reference(
    emb_output_layer, use_final_proj, golden_key
):
    """Going to ONNX did not change the fp16 behaviour. Never compared against the fp32
    golden."""
    cosine, max_abs = _compare(
        Device("cuda", 0), True, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN_FP16, f"cosine {cosine:.8f} < {COSINE_MIN_FP16}"
    assert max_abs <= MAX_ABS_MAX_FP16, (
        f"max-abs {max_abs:.3e} > {MAX_ABS_MAX_FP16:.1e}"
    )
