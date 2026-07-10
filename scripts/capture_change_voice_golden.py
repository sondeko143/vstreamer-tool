"""Capture a numeric golden for change_voice BEFORE refactoring.

Run on the pre-refactor code with the real RVC model + GPU, passing your RVC
worker's TOML config (the `[rvc]` section: model_file / hubert_model_file /
rmvpe_model_file / gpu):
    uv run --extra rvc python scripts/capture_change_voice_golden.py \
        --config path/to/your-rvc-config.toml

The RVC synthesizer (VITS-style) injects random noise, so change_voice is
stochastic run-to-run by design (verified: ~3.6% mean self-noise on both CUDA
and CPU). That randomness is *seedable*, though: seeding torch + onnxruntime
immediately before the call makes the int16 output bit-exact-reproducible
(verified self-noise 0). Everything the refactor touches (the orchestration
producing infer's inputs, and postprocessing) is deterministic on its own; only
the untouched `infer` synthesizer is stochastic. So this captures a *seeded*
output as the golden. The golden test re-seeds identically; it asserted exact
equality while HuBERT ran under fairseq, and now asserts a tight tolerance
(correlation + waveform SNR) because the content encoder moved to transformers.
This npz must be captured while `vspeech/lib/rvc.py` still uses fairseq -- it is the
fairseq-side reference the migration is validated against. Rebuilds the
device/hubert/RVC-session/rmvpe-session exactly as rvc_worker does and writes
tests/assets/rvc_golden/change_voice_golden.npz (gitignored): the input, the
seed, and the seeded output.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "assets" / "rvc_golden"
GOLDEN_NPZ = GOLDEN_DIR / "change_voice_golden.npz"


def seed_all(seed: int = 0) -> None:
    """Seed every RNG change_voice's stochastic synthesizer can consume.

    Makes the otherwise-stochastic int16 output bit-exact-reproducible. Must be
    called immediately before each change_voice invocation.
    """
    import onnxruntime as ort
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ort.set_seed(seed)


def make_fixed_input(voice_sample_rate: int = 48000, seconds: float = 1.0) -> bytes:
    """A deterministic 220 Hz sine as int16 PCM bytes (no RNG -> reproducible)."""
    n = int(voice_sample_rate * seconds)
    t = np.arange(n, dtype=np.float64) / voice_sample_rate
    wave = (0.3 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    return wave.tobytes()


def build_rvc_runtime(config_path: Path) -> dict[str, Any]:
    """Reconstruct the rvc_worker runtime (device + models + metadata)."""
    from vspeech.config import Config
    from vspeech.config import F0ExtractorType
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.onnx_session import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    rvc_config = config.rvc

    device, _ = get_device(rvc_config.gpu_id, rvc_config.gpu_name)
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file, device=device, is_half=half_available
    )
    session = create_session(rvc_config.model_file, device)
    if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
        rmvpe_session = create_session(rvc_config.rmvpe_model_file, device)
    else:
        rmvpe_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc_config,
        "device": device,
        "hubert_model": hubert_model,
        "session": session,
        "rmvpe_session": rmvpe_session,
        "target_sample_rate": metadata["samplingRate"],
        "f0_enabled": metadata["f0"],
        "emb_output_layer": metadata.get("embOutputLayer", 9),
        "use_final_proj": metadata.get("useFinalProj", True),
    }


def run_change_voice(
    rt: dict[str, Any], voice_frames: bytes, voice_sample_rate: int
) -> NDArray[np.int16]:
    from vspeech.lib.rvc import change_voice

    return change_voice(
        voice_frames=voice_frames,
        rvc_config=rt["rvc_config"],
        voice_sample_rate=voice_sample_rate,
        target_sample_rate=rt["target_sample_rate"],
        device=rt["device"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_enabled=rt["f0_enabled"],
        rmvpe_session=rt["rmvpe_session"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rt = build_rvc_runtime(args.config)
    voice_frames = make_fixed_input(args.sample_rate, args.seconds)

    seed_all(args.seed)
    out1 = run_change_voice(rt, voice_frames, args.sample_rate)
    seed_all(args.seed)
    out2 = run_change_voice(rt, voice_frames, args.sample_rate)

    # Sanity: seeding must make the stochastic synthesizer reproducible.
    n = min(out1.shape[0], out2.shape[0])
    diff = np.abs(out1[:n].astype(np.int32) - out2[:n].astype(np.int32))
    print(
        f"output_len={out1.shape[0]} len2={out2.shape[0]} "
        f"seeded self-noise max={int(diff.max()) if n else 0} "
        f"(expected 0 -> reproducible)"
    )
    if not n or diff.max() != 0:
        raise SystemExit(
            "seeded runs are not bit-exact; golden equality is unsafe -- investigate"
        )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GOLDEN_NPZ,
        voice_frames=np.frombuffer(voice_frames, dtype=np.int16),
        voice_sample_rate=np.int64(args.sample_rate),
        seed=np.int64(args.seed),
        output=out1,
    )
    print(f"wrote {GOLDEN_NPZ}")


if __name__ == "__main__":
    main()
