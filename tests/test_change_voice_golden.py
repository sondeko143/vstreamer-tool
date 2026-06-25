"""Numeric-golden equivalence test for the refactored change_voice.

The RVC synthesizer is stochastic by design, so equivalence is checked under a
fixed seed (see scripts/capture_change_voice_golden.py): seeding torch +
onnxruntime immediately before the call makes the int16 output bit-exact, so a
behavior-preserving refactor reproduces the captured golden exactly while any
real change to the infer-input orchestration diverges the RNG stream and shifts
the output by thousands of LSB.

Skips unless the golden npz, CUDA, and an RVC worker config (path supplied via
the $VSPEECH_RVC_GOLDEN_CONFIG env var) are all present, so it never breaks the
CPU/CI suite and bakes no machine-specific path into the repo.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_NPZ = REPO_ROOT / "tests" / "assets" / "rvc_golden" / "change_voice_golden.npz"

# Path to the RVC worker TOML config, supplied out-of-band via env var so no
# machine-specific path lives in the repo. Unset -> the test skips.
_CONFIG_ENV = "VSPEECH_RVC_GOLDEN_CONFIG"
_config_path = os.environ.get(_CONFIG_ENV)
GOLDEN_CONFIG = Path(_config_path) if _config_path else None

pytestmark = pytest.mark.skipif(
    not GOLDEN_NPZ.exists()
    or not torch.cuda.is_available()
    or GOLDEN_CONFIG is None
    or not GOLDEN_CONFIG.exists(),
    reason=f"golden npz / CUDA / ${_CONFIG_ENV} config not available",
)


def test_change_voice_matches_seeded_golden():
    from scripts import capture_change_voice_golden as cap

    assert GOLDEN_CONFIG is not None  # skipif guarantees this; narrows for ty

    data = np.load(GOLDEN_NPZ)
    voice_frames = data["voice_frames"].astype(np.int16).tobytes()
    voice_sample_rate = int(data["voice_sample_rate"])
    seed = int(data["seed"])
    golden = data["output"]

    rt = cap.build_rvc_runtime(GOLDEN_CONFIG)
    cap.seed_all(seed)
    out = cap.run_change_voice(rt, voice_frames, voice_sample_rate)

    assert out.shape == golden.shape, f"length changed: {out.shape} vs {golden.shape}"
    diff = np.abs(out.astype(np.int32) - golden.astype(np.int32))
    # Seeded => expect bit-exact; allow a 1-LSB margin for any platform jitter.
    # A broken refactor would diverge the seeded RNG stream and miss by thousands.
    assert diff.max() <= 1, (
        f"max diff {int(diff.max())} (mean {float(diff.mean()):.3f})"
    )
