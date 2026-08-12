"""Audio regression test for change_voice.

The whole path is onnxruntime inference now (ADR-0081), and only the RVC synthesizer's
VITS-style `infer` draws random numbers -- HuBERT and the f0 graph draw none. So after
seed_all(), which is `ort.set_seed` alone, the RNG stream is fully determined and any
difference from the golden comes only from differences in the feature values. (This used
to be argued in torch's terms -- eval + inference_mode, plus torch seeding. Neither
applies: no framework tensor is on the path, and torch seeding was measured to contribute
nothing before it was dropped.)

The output is not bit-exact against the golden, so the verdict uses tolerances
(correlation + segmental SNR).

Gated on `requires_cuda` / `requires_rvc_config` / `requires_rvc_golden`; run
`uv run pytest --markers` for what each of those needs (ADR-0089).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from scripts.hubert_metrics import CORR_MIN
from scripts.hubert_metrics import SNR_MIN_DB
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_NPZ = REPO_ROOT / "tests" / "assets" / "rvc_golden" / "change_voice_golden.npz"

# Path to the RVC worker TOML config, supplied out-of-band via env var so no
# machine-specific path lives in the repo. Unset -> the test skips.
_CONFIG_ENV = "VSPEECH_RVC_GOLDEN_CONFIG"
_config_path = os.environ.get(_CONFIG_ENV)
GOLDEN_CONFIG = Path(_config_path) if _config_path else None

pytestmark = [
    pytest.mark.requires_cuda,
    pytest.mark.requires_rvc_config,
    pytest.mark.requires_rvc_golden,
]


def test_change_voice_matches_seeded_golden():
    from scripts import capture_change_voice_golden as cap

    assert GOLDEN_CONFIG is not None  # the gate guarantees this; narrows for ty

    data = np.load(GOLDEN_NPZ)
    voice_frames = data["voice_frames"].astype(np.int16).tobytes()
    voice_sample_rate = int(data["voice_sample_rate"])
    seed = int(data["seed"])
    golden = data["output"]

    rt = cap.build_rvc_runtime(GOLDEN_CONFIG)
    cap.seed_all(seed)
    out = cap.run_change_voice(rt, voice_frames, voice_sample_rate)

    assert out.shape == golden.shape, f"length changed: {out.shape} vs {golden.shape}"
    # out is not bit-exact against the golden, so it is matched with tolerances.
    # When relaxing them, leave the measured value in this comment (up to 10x the
    # measurement).
    correlation = waveform_correlation(out, golden)
    snr_db = waveform_snr(golden, out)
    assert correlation >= CORR_MIN, f"correlation {correlation:.6f} < {CORR_MIN}"
    assert snr_db >= SNR_MIN_DB, f"waveform SNR {snr_db:.2f} dB < {SNR_MIN_DB} dB"
