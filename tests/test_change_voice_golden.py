"""change_voice の音声回帰テスト。

HuBERT は eval + inference_mode 下で RNG を一切消費しないため、seed_all() 後の
RNG ストリームは RVC synthesizer の infer だけで決まる。したがって golden との差は
特徴量の値の差だけに由来する。

出力は golden と bit-exact にはならないので、判定は許容誤差（相関 + セグメンタル SNR）。

golden npz / CUDA / RVC worker config ($VSPEECH_RVC_GOLDEN_CONFIG) が揃わなければ skip。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hubert_metrics import CORR_MIN
from scripts.hubert_metrics import SNR_MIN_DB
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr

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
    # out は golden と bit-exact にはならないので、許容誤差で照合する。
    # 緩めるときは実測値をこのコメントに残すこと（実測の 10 倍まで）。
    correlation = waveform_correlation(out, golden)
    snr_db = waveform_snr(golden, out)
    assert correlation >= CORR_MIN, f"correlation {correlation:.6f} < {CORR_MIN}"
    assert snr_db >= SNR_MIN_DB, f"waveform SNR {snr_db:.2f} dB < {SNR_MIN_DB} dB"
