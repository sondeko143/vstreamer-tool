"""Measurement harness for the RTF of RVC streaming VC.

It reads **only the [rvc] section** of `--config` and never looks at [stream_vc.rvc] (per
ADR-0054 the two are configured independently, so different contents mean a different
subject of measurement). To measure a [stream_vc.rvc] configuration, prepare a config
whose [rvc] mirrors it and pass that.

It loads the models once from `--config`'s [rvc] section, pushes a synthetic voiced signal
through StreamingVc in fixed blocks, and sweeps per-block latency and context-inclusive
RTF. It prints a table marking the feasible rows and recommends the feasible config with
the lowest latency. The final block/context/latency budget and the go/no-go call are made
by a human. The measured span deliberately includes the input H2D and output D2H copies
(so it reflects the real per-block streaming cost as deployed). margin (0.5 by default) is
2x headroom for transport/jitter/crossfade.

  uv run poe stream-vc-rtf --config ./config.toml

The pure analysis helpers (make_voiced_signal / parse_grid / summarize / format_table /
recommend / go_no_go) depend only on numpy and can be imported and tested on a CPU with no
GPU. The torch / vspeech / StreamingVc imports are deferred into the functions that run
the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import torch
    from onnxruntime import InferenceSession

    from vspeech.config import RvcConfig


def make_voiced_signal(
    rate: int, seconds: float, f0: float = 150.0, seed: int = 0
) -> NDArray[np.float32]:
    """Return a deterministic voiced signal (harmonics + slight vibrato + slight noise) in
    [-1, 1].

    The f0 extractors (rmvpe/fcpe) return unvoiced (0) for every frame on silence, which
    makes the measurement unrepresentative, so the signal has a clear fundamental. Seeded,
    hence reproducible.
    """
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    f0_t = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))  # 5 Hz vibrato
    phase = 2 * np.pi * np.cumsum(f0_t) / rate
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):  # 5 harmonics, 1/k rolloff
        sig += np.sin(k * phase) / k
    peak = np.max(np.abs(sig)) or 1.0
    sig = 0.3 * sig / peak
    rng = np.random.default_rng(seed)
    sig = sig + 0.005 * rng.standard_normal(n)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def parse_grid(text: str) -> list[float]:
    """Expand `"20,40,80"` into `[20.0, 40.0, 80.0]`."""
    return [float(x) for x in text.split(",") if x.strip()]


@dataclass
class BlockResult:
    block_ms: float
    context_ms: float
    f0: str
    p50_ms: float
    p95_ms: float
    max_ms: float
    rtf_p95: float
    latency_ms: float
    feasible: bool


def summarize(
    latencies_s: list[float],
    block_seconds: float,
    margin: float,
    block_ms: float,
    context_ms: float,
    f0: str,
) -> BlockResult:
    """Per-block latency series -> p50/p95/max, RTF (based on p95), one-way latency,
    feasibility.

    RTF = per-block compute / the block's real-time duration. The context is recomputed
    every tick and therefore lands in that numerator (the redundant inference ADR-0053
    pointed out). feasible means RTF_p95 < margin (0.5 by default = 2x headroom for
    transport/jitter/crossfade). The one-way algorithmic latency is about
    block_ms + compute_p95.
    """
    arr = np.asarray(latencies_s, dtype=np.float64)
    p50 = float(np.percentile(arr, 50)) * 1000.0
    p95 = float(np.percentile(arr, 95)) * 1000.0
    mx = float(arr.max()) * 1000.0
    rtf_p95 = (p95 / 1000.0) / block_seconds
    latency_ms = block_ms + p95
    return BlockResult(
        block_ms=block_ms,
        context_ms=context_ms,
        f0=f0,
        p50_ms=p50,
        p95_ms=p95,
        max_ms=mx,
        rtf_p95=rtf_p95,
        latency_ms=latency_ms,
        feasible=rtf_p95 < margin,
    )


def recommend(results: list[BlockResult]) -> BlockResult | None:
    """The feasible result with the lowest one-way latency, or None if there is none."""
    feasible = [r for r in results if r.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r.latency_ms)


def go_no_go(results: list[BlockResult]) -> bool:
    """go if at least one result is feasible."""
    return any(r.feasible for r in results)


def format_table(results: list[BlockResult]) -> str:
    """Render the sweep results as an aligned text table (feasible rows get
    [FEASIBLE])."""
    header = (
        f"{'block':>6} {'ctx':>6} {'f0':>6} "
        f"{'p50ms':>7} {'p95ms':>7} {'maxms':>7} {'RTF':>6} {'lat_ms':>7}  mark"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        mark = "[FEASIBLE]" if r.feasible else ""
        row = (
            f"{r.block_ms:>6.0f} {r.context_ms:>6.0f} {r.f0:>6} "
            f"{r.p50_ms:>7.2f} {r.p95_ms:>7.2f} {r.max_ms:>7.2f} "
            f"{r.rtf_p95:>6.2f} {r.latency_ms:>7.1f}  {mark}"
        )
        lines.append(row.rstrip())
    return "\n".join(lines)


def load_shared_runtime(
    config_path: Path, gpu_id_override: int | None
) -> dict[str, Any]:
    """Load the models once from [rvc] (f0_session is built per extractor by the sweep)."""
    import json

    from vspeech.config import Config
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.onnx_session import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    rvc_config = config.rvc
    gpu_id = gpu_id_override if gpu_id_override is not None else rvc_config.gpu_id

    device, device_name = get_device(gpu_id, rvc_config.gpu_name)
    print(f"device: {device} ({device_name})")
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file, device=device, is_half=half_available
    )
    session = create_session(rvc_config.model_file, device)
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc_config,
        "device": device,
        "hubert_model": hubert_model,
        "session": session,
        "target_sample_rate": metadata["samplingRate"],
        "f0_enabled": metadata["f0"],
        "emb_output_layer": metadata.get("embOutputLayer", 9),
        "use_final_proj": metadata.get("useFinalProj", True),
    }


def make_f0_session(
    rvc_config: RvcConfig, f0: str, device: torch.device
) -> InferenceSession | None:
    """Build the f0 session for "rmvpe"/"fcpe". None when the file is unset or missing."""
    from pathlib import Path

    from vspeech.lib.onnx_session import create_session

    if f0 == "none":
        return None
    file_map = {
        "rmvpe": rvc_config.rmvpe_model_file,
        "fcpe": rvc_config.fcpe_model_file,
    }
    model_file = file_map.get(f0)
    if (
        model_file is None
        or model_file == Path()
        or not model_file.expanduser().exists()
    ):
        return None
    return create_session(model_file, device)


def next_block(
    signal: NDArray[np.float32], block_len: int, i: int
) -> NDArray[np.float32]:
    """Take the i-th block out of the signal, wrapping around (length block_len)."""
    span = max(1, signal.shape[0] - block_len)
    start = (i * block_len) % span
    return signal[start : start + block_len]


def run_sweep(
    rt: dict[str, Any],
    signal: NDArray[np.float32],
    block_ms_list: list[float],
    context_ms_list: list[float],
    f0_list: list[str],
    iters: int,
    warmup_iters: int,
    margin: float,
) -> list[BlockResult]:
    """Sweep and measure the per-block latency of each (block, context, f0)."""
    import time

    import torch

    from vspeech.config import F0ExtractorType
    from vspeech.lib.stream_vc import StreamingVc

    rate = 16000
    device = rt["device"]
    is_cuda = device.type == "cuda"
    results: list[BlockResult] = []

    if not rt["f0_enabled"]:
        print("model is f0-less; collapsing f0 axis to ['none']")
        f0_list = ["none"]

    for f0 in f0_list:
        if f0 == "none" and rt["f0_enabled"]:
            print("skip f0=none: model is f0-enabled (needs an f0 extractor)")
            continue
        f0_session = make_f0_session(rt["rvc_config"], f0, device)
        if f0 != "none" and f0_session is None:
            if f0 not in ("rmvpe", "fcpe"):
                print(f"skip f0={f0}: unsupported extractor (use rmvpe/fcpe/none)")
            else:
                print(f"skip f0={f0}: model file not configured/found")
            continue
        if f0 == "none":
            cfg_f0 = rt["rvc_config"]
        else:
            cfg_f0 = rt["rvc_config"].model_copy(
                update={"f0_extractor_type": F0ExtractorType(f0)}
            )
        for block_ms in block_ms_list:
            block_len = round(block_ms * rate / 1000.0)
            for context_ms in context_ms_list:
                context_len = round(context_ms * rate / 1000.0)
                try:
                    sv = StreamingVc(
                        rvc_config=cfg_f0,
                        device=device,
                        hubert_model=rt["hubert_model"],
                        session=rt["session"],
                        f0_session=f0_session,
                        target_sample_rate=rt["target_sample_rate"],
                        f0_enabled=rt["f0_enabled"],
                        emb_output_layer=rt["emb_output_layer"],
                        use_final_proj=rt["use_final_proj"],
                        block_len=block_len,
                        context_len=context_len,
                    )
                    sv.warmup()
                    latencies: list[float] = []
                    for i in range(iters + warmup_iters):
                        block = next_block(signal, block_len, i)
                        if is_cuda:
                            torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        sv.process_block(block)
                        if is_cuda:
                            torch.cuda.synchronize(device)
                        t1 = time.perf_counter()
                        if i >= warmup_iters:
                            latencies.append(t1 - t0)
                    results.append(
                        summarize(
                            latencies,
                            block_seconds=block_len / rate,
                            margin=margin,
                            block_ms=block_ms,
                            context_ms=context_ms,
                            f0=f0,
                        )
                    )
                    print(f"  done block={block_ms}ms ctx={context_ms}ms f0={f0}")
                except Exception as e:
                    print(
                        f"  FAILED block={block_ms}ms ctx={context_ms}ms f0={f0}: {e}"
                    )
                    continue
    return results


def main() -> None:
    # The go/no-go line contains Japanese. When stdout/stderr is a pipe/redirect, Windows
    # picks cp1252 by default and main() dies with UnicodeEncodeError before writing
    # `--json` (the encoding guard this project keeps needing; same shape as
    # convert_hubert.py / export_hubert_onnx.py). typeshed types sys.stdout/stderr as a
    # TextIO without .reconfigure, but at runtime they are TextIOWrapper.
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RVC streaming VC RTF harness")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--block-ms", default="20,40,80,160")
    parser.add_argument("--context-ms", default="0,100,200,400,800")
    parser.add_argument("--f0", default="rmvpe,fcpe")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    block_ms_list = parse_grid(args.block_ms)
    context_ms_list = parse_grid(args.context_ms)
    f0_list = [x.strip() for x in args.f0.split(",") if x.strip()]

    rt = load_shared_runtime(args.config, args.gpu_id)
    if args.wav is not None:
        signal = load_wav_16k(args.wav)
    else:
        signal = make_voiced_signal(16000, args.seconds, seed=0)

    # Ensure the signal covers the largest block, so every cell gets a full
    # fixed-shape block (a short --wav would otherwise yield variable-length
    # tail blocks and break the one-warmup assumption).
    max_block_len = round(max(block_ms_list) * 16000 / 1000.0)
    if signal.shape[0] < max_block_len + 1:
        reps = -(-(max_block_len + 1) // signal.shape[0])  # ceil division
        signal = np.tile(signal, reps)

    results = run_sweep(
        rt,
        signal,
        block_ms_list=block_ms_list,
        context_ms_list=context_ms_list,
        f0_list=f0_list,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        margin=args.margin,
    )

    print()
    if not results:
        print(
            "no cells measured -- every cell was skipped or failed "
            "(check f0 model files / the grid / the warnings above)"
        )
    else:
        print(format_table(results))
        print()
        if go_no_go(results):
            best = recommend(results)
            assert best is not None  # go_no_go(...) => a feasible config exists
            print(
                f"go/no-go: GO. recommend block={best.block_ms:.0f}ms "
                f"ctx={best.context_ms:.0f}ms f0={best.f0} "
                f"(latency ~{best.latency_ms:.1f}ms, RTF {best.rtf_p95:.2f}). "
                "-> 最終の block/context/遅延予算は人が確定する。"
            )
        else:
            print(f"go/no-go: NO-GO (no config with RTF_p95 < {args.margin})")
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")


def load_wav_16k(path: Path) -> NDArray[np.float32]:
    """Return the wav as 16kHz mono float32 in [-1,1]."""
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono
    if sr != 16000:
        from vspeech.lib.rvc import get_resampler

        wav = get_resampler(sr, 16000, torch.device("cpu"))(wav)
    return wav.numpy().astype(np.float32)


if __name__ == "__main__":
    main()
