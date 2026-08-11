"""Capture and re-check a numeric baseline of Stream VC (the A/B gate for ADR-0080/0081).

The torch-based conversion path is about to be replaced by numpy + onnxruntime-native
`OrtValue`. This harness freezes what the torch version does -- the exact input blocks,
the seed, and the emitted int16 blocks -- so the replacement can be judged against it
instead of against a memory of how it sounded.

Two subcommands over one runtime:

    uv run poe stream-vc-baseline capture   # writes the baseline npz + a latency series
    uv run poe stream-vc-baseline compare   # re-runs the same input and judges the diff

`compare` is the gate, and it has **three** exit codes, because it has three answers:

    0  bit-exact           -- the contract, and measured to be achievable
    2  within tolerance    -- close but not identical: 要判断, NOT a pass
    1  fail                -- outside the tolerance, or the shapes disagree

Anything judging this command by `$?` must therefore test for 0, not for "not 1".
It also refuses to run at all when the baseline's geometry, models or model parameters
differ from the current config, since a difference that came from config drift would
otherwise be read as a difference caused by the code under test.

The config path is supplied out-of-band through `$VSPEECH_STREAM_VC_BASELINE_CONFIG`
(`--config` overrides it), the same way tests/test_change_voice_golden.py takes its
config, so no machine-specific path lives in the repo. The npz lands under
tests/assets/rvc_golden/, which is gitignored.

Why the whole `[stream_vc]` runtime and not a hand-built StreamingVc: the geometry that
matters is the one the producer actually runs (block/context/crossfade/SOLA/lookahead
from the config), and `vspeech.stream_vc.runner` already knows how to build it. The
runner's surrounding stages (input_boost, the VAD gate, the envelope) are deliberately
left out: none of them is touched by the torch removal, and including them would mix
their behaviour into a measurement about `StreamingVc.process_block`.

Determinism, and why `--seed-mode` exists: the RVC synthesizer is VITS-style and draws
random noise on every inference, so conversion is stochastic run-to-run. The existing
utterance golden (scripts/capture_change_voice_golden.py) seeds torch *and*
onnxruntime. ADR-0080 deletes torch from the runtime, so only `ort.set_seed()` will
survive -- hence "ort" is a first-class mode here, and "none" measures the unseeded
spread the seeding is supposed to remove. Reproducibility within one process is also not
the same question as reproducibility across processes: `capture` answers the first (it
runs the block sequence twice in one process), and `compare` -- which by construction
runs in a later process -- answers the second.

The timing loop holds no torch on purpose. `process_block` ends with a device-to-host
copy of the decoder output, which is synchronous, so the tick is complete when it
returns and no explicit `torch.cuda.synchronize` is needed. That keeps this harness
runnable, unchanged, against the numpy implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.hubert_metrics import CORR_MIN
from scripts.hubert_metrics import SNR_MIN_DB
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr
from scripts.stream_vc_rtf import make_voiced_signal

if TYPE_CHECKING:
    from vspeech.config import StreamVcConfig
    from vspeech.lib.stream_vc import StreamingVc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = REPO_ROOT / "tests" / "assets" / "rvc_golden" / "stream_vc_baseline.npz"

# Path to the [stream_vc] TOML config, supplied out-of-band so no machine-specific path
# lives in the repo (same contract as $VSPEECH_RVC_GOLDEN_CONFIG).
CONFIG_ENV = "VSPEECH_STREAM_VC_BASELINE_CONFIG"

SEED_MODES = ("both", "ort", "none")

# StreamingVc takes blocks at HuBERT's rate; the producer's audio capture has already
# resampled to it by the time a block reaches process_block.
INPUT_RATE = 16000


def seed_runtime(seed: int, mode: str) -> None:
    """Seed every RNG the stochastic RVC synthesizer can consume, per `mode`.

    - "both": onnxruntime + torch, the recipe scripts/capture_change_voice_golden.py
      uses today.
    - "ort": onnxruntime only. This is the only recipe that survives ADR-0080, since
      torch leaves the runtime; whether it is sufficient is measured, not assumed.
    - "none": no seeding at all, which measures the synthesizer's own stochastic spread.

    Must be called immediately before the block sequence it governs.
    """
    if mode not in SEED_MODES:
        raise ValueError(f"unknown seed mode {mode!r} (expected one of {SEED_MODES})")
    if mode == "none":
        return
    import onnxruntime as ort

    ort.set_seed(seed)
    if mode == "both":
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def make_input_blocks(
    block_len: int, n_blocks: int, seed: int = 0
) -> NDArray[np.float32]:
    """`n_blocks` contiguous blocks of a deterministic voiced signal, shape (n, block_len).

    Reuses the RTF harness's generator: the f0 extractors return unvoiced on silence,
    which would make both the numeric comparison and the latency uninformative.
    """
    total = block_len * n_blocks
    signal = make_voiced_signal(INPUT_RATE, total / INPUT_RATE, seed=seed)
    if signal.shape[0] < total:
        # int(rate * seconds) can land one sample short of `total` through float
        # rounding; pad rather than shrink so every block is full length.
        signal = np.pad(signal, (0, total - signal.shape[0]))
    return np.ascontiguousarray(signal[:total].reshape(n_blocks, block_len))


def geometry(sv_config: StreamVcConfig) -> dict[str, float]:
    """The analysis-window geometry that defines what was measured.

    Recorded in the npz and re-checked on compare: a baseline captured at a different
    block/context/crossfade is a different subject, and comparing across the two would
    silently report nonsense.
    """
    return {
        "block_ms": sv_config.block_ms,
        "context_ms": sv_config.context_ms,
        "crossfade_ms": sv_config.crossfade_ms,
        "sola_search_ms": sv_config.sola_search_ms,
        "lookahead_ms": sv_config.lookahead_ms,
    }


# Fields of [stream_vc.rvc] that select *which GPU* rather than *what is computed*.
# Excluded from the provenance fingerprint so the same baseline can be re-checked on
# another card, which is the whole point of recording a device-independent bit-exact
# result.
_DEVICE_SELECTION_FIELDS = frozenset({"gpu_id", "gpu_name"})


def provenance(sv_config: StreamVcConfig, target_sample_rate: int) -> dict[str, Any]:
    """Everything besides the geometry that decides what the emitted samples are.

    The geometry guard exists so a baseline captured under one window is never compared
    against another; the same argument applies to the models and their parameters. A
    different RVC checkpoint, a different f0 extractor or a different f0_up_key produces
    a different waveform, and without this the resulting mismatch would be blamed on the
    code change under test -- the exact confusion the guard is for.

    `target_sample_rate` comes from the RVC model's own metadata, so it doubles as a
    check that the recorded model file really is the one that was loaded.
    """
    # set(...) rather than the frozenset itself: pydantic's IncEx type is invariant on
    # the mutable set.
    rvc = sv_config.rvc.model_dump(mode="json", exclude=set(_DEVICE_SELECTION_FIELDS))
    return {"rvc": rvc, "target_sample_rate": int(target_sample_rate)}


def provenance_mismatches(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Human-readable lines describing every field that differs. Empty means identical."""
    lines: list[str] = []
    if baseline.get("target_sample_rate") != current.get("target_sample_rate"):
        lines.append(
            f"target_sample_rate: baseline={baseline.get('target_sample_rate')} "
            f"current={current.get('target_sample_rate')}"
        )
    base_rvc = baseline.get("rvc", {})
    cur_rvc = current.get("rvc", {})
    for key in sorted(set(base_rvc) | set(cur_rvc)):
        if base_rvc.get(key) != cur_rvc.get(key):
            lines.append(
                f"rvc.{key}: baseline={base_rvc.get(key)!r} current={cur_rvc.get(key)!r}"
            )
    return lines


def build_runtime(config_path: Path) -> tuple[StreamVcConfig, dict[str, Any]]:
    """Load [stream_vc] and build the same device/models the producer builds."""
    from vspeech.config import Config
    from vspeech.stream_vc.runner import build_stream_vc_runtime

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    sv_config = config.stream_vc
    return sv_config, build_stream_vc_runtime(sv_config)


def make_vc(rt: dict[str, Any], sv_config: StreamVcConfig) -> StreamingVc:
    """A StreamingVc with fresh rolling state, sharing the already-loaded sessions."""
    from vspeech.stream_vc.runner import make_streaming_vc

    return make_streaming_vc(rt, sv_config)


def run_blocks(
    sv: StreamingVc, blocks: NDArray[np.float32]
) -> tuple[NDArray[np.int16], NDArray[np.float64]]:
    """Push every block through `process_block`; return the emits and per-tick wall time.

    The emit length is constant by construction (it is derived from the real-time clock,
    not from the render length), so a varying length means the geometry changed
    mid-run -- fail loud rather than record a ragged baseline.
    """
    outputs: list[NDArray[np.int16]] = []
    latencies = np.empty(blocks.shape[0], dtype=np.float64)
    for i in range(blocks.shape[0]):
        t0 = perf_counter()
        out = sv.process_block(blocks[i])
        t1 = perf_counter()
        outputs.append(np.asarray(out, dtype=np.int16))
        latencies[i] = t1 - t0
    lengths = {int(o.shape[0]) for o in outputs}
    if len(lengths) != 1:
        raise SystemExit(f"emit length is not constant across ticks: {sorted(lengths)}")
    return np.stack(outputs), latencies


@dataclass
class Latency:
    n: int
    p50_ms: float
    p95_ms: float
    max_ms: float

    def line(self, label: str) -> str:
        return (
            f"{label}: N={self.n} p50={self.p50_ms:.2f}ms "
            f"p95={self.p95_ms:.2f}ms max={self.max_ms:.2f}ms"
        )


def latency_stats(latencies_s: NDArray[np.float64]) -> Latency:
    arr = np.asarray(latencies_s, dtype=np.float64)
    return Latency(
        n=int(arr.size),
        p50_ms=float(np.percentile(arr, 50)) * 1000.0,
        p95_ms=float(np.percentile(arr, 95)) * 1000.0,
        max_ms=float(arr.max()) * 1000.0,
    )


def one_lsb_snr_db(reference: NDArray[np.int16]) -> float:
    """The SNR this reference would show if *every* sample were off by exactly 1 int16 LSB.

    A yardstick, not a gate. Bit equality is reproducible here, so a seeded run's own
    noise gives no basis for any non-zero tolerance; this does, and it is derived from
    the captured waveform rather than picked.

    Read it only together with `max_abs_diff`. SNR bounds *total* error energy, so
    scoring at or above this line does not by itself mean "every sample is within a
    quantization step": one sample off by sqrt(n) (about 1240 LSB at n = 1.5M samples)
    scores exactly the same as n samples off by 1. The pair
    `max|diff| <= 1 and snr_db >= one_lsb_snr_db` is what actually says "at or below the
    quantization step of the format"; the SNR alone says "as much total error as
    rounding would produce, distributed somehow".
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    signal = float((ref**2).sum())
    if signal == 0.0 or ref.size == 0:
        return float("-inf")
    # Noise energy of a constant 1-LSB error is exactly the sample count.
    return 10.0 * (float(np.log10(signal)) - float(np.log10(ref.size)))


@dataclass
class Verdict:
    """The outcome of comparing two runs of the same input at the same seed."""

    n_blocks: int
    bit_exact: bool
    max_abs_diff: int
    blocks_differing: int
    correlation: float
    snr_db: float
    worst_block_correlation: float
    worst_block_snr_db: float
    one_lsb_snr_db: float
    corr_min: float
    snr_min_db: float

    @property
    def within_tolerance(self) -> bool:
        """The thresholds hold both over the whole stream and on the worst single block.

        The whole-stream pair matches tests/test_change_voice_golden.py, but on its own
        it is too coarse here: at N=200 the gate would still admit one block sitting at
        about -12 dB SNR, because that block's noise is only 1/200 of the total energy.
        A per-block floor at the same thresholds closes that, and costs nothing on a
        genuinely equivalent run (an identical block scores corr 1.0 / SNR inf).
        """
        return (
            self.correlation >= self.corr_min
            and self.snr_db >= self.snr_min_db
            and self.worst_block_correlation >= self.corr_min
            and self.worst_block_snr_db >= self.snr_min_db
        )

    @property
    def outcome(self) -> str:
        """BIT_EXACT / TOLERANCE / FAIL. TOLERANCE is "要判断", not a pass."""
        if self.bit_exact:
            return "BIT_EXACT"
        return "TOLERANCE" if self.within_tolerance else "FAIL"

    @property
    def exit_code(self) -> int:
        """0 = bit-exact, 2 = within tolerance but not bit-exact, 1 = fail.

        Three states need three codes. Bit equality is the contract and is measured to
        be achievable, so folding "close enough" into 0 would make the documented
        "要判断" state indistinguishable from a pass to anything judging by exit code --
        which is exactly how this project is required to judge commands.
        """
        return {"BIT_EXACT": 0, "TOLERANCE": 2, "FAIL": 1}[self.outcome]

    def report(self) -> str:
        head = {
            "BIT_EXACT": "verdict: BIT-EXACT (exit 0)",
            "TOLERANCE": "verdict: WITHIN TOLERANCE, NOT BIT-EXACT (exit 2) -- 要判断",
            "FAIL": "verdict: FAIL (exit 1)",
        }[self.outcome]
        return "\n".join(
            [
                head,
                f"  blocks              : {self.n_blocks} "
                f"({self.blocks_differing} differing)",
                f"  max |diff| (int16)  : {self.max_abs_diff}",
                f"  correlation         : {self.correlation:.9f} (min {self.corr_min})",
                f"  waveform SNR        : {self.snr_db:.2f} dB "
                f"(min {self.snr_min_db} dB)",
                f"  worst block corr    : {self.worst_block_correlation:.9f} "
                f"(min {self.corr_min})",
                f"  worst block SNR     : {self.worst_block_snr_db:.2f} dB "
                f"(min {self.snr_min_db} dB)",
                f"  1 LSB yardstick SNR : {self.one_lsb_snr_db:.2f} dB "
                "(only WITH max|diff| <= 1 does this mean 'below int16 quantization'; "
                "SNR alone bounds total error energy, not the worst sample)",
            ]
        )


def judge(
    reference: NDArray[np.int16],
    test: NDArray[np.int16],
    corr_min: float = CORR_MIN,
    snr_min_db: float = SNR_MIN_DB,
) -> Verdict:
    """Compare two (n_blocks, emit_len) int16 arrays.

    Bit equality is the verdict; the correlation/SNR thresholds only separate "close but
    not identical" (which needs a human) from "broken". They are applied both over the
    concatenated stream and to the worst single block, so a local defect cannot be
    diluted by the other N-1 blocks.
    """
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")
    diff = np.abs(reference.astype(np.int64) - test.astype(np.int64))
    block_corr = [
        waveform_correlation(reference[i], test[i]) for i in range(reference.shape[0])
    ]
    block_snr = [waveform_snr(reference[i], test[i]) for i in range(reference.shape[0])]
    return Verdict(
        n_blocks=int(reference.shape[0]),
        bit_exact=bool(diff.max() == 0),
        max_abs_diff=int(diff.max()),
        blocks_differing=int((diff.max(axis=1) > 0).sum()),
        correlation=waveform_correlation(reference.ravel(), test.ravel()),
        snr_db=waveform_snr(reference.ravel(), test.ravel()),
        worst_block_correlation=min(block_corr),
        worst_block_snr_db=min(block_snr),
        one_lsb_snr_db=one_lsb_snr_db(reference),
        corr_min=corr_min,
        snr_min_db=snr_min_db,
    )


def resolve_config(explicit: Path | None) -> Path:
    """--config, else $VSPEECH_STREAM_VC_BASELINE_CONFIG. Missing -> a usage error."""
    if explicit is not None:
        return explicit
    from_env = os.environ.get(CONFIG_ENV)
    if not from_env:
        raise SystemExit(
            f"config が指定されていません。--config か ${CONFIG_ENV} を設定してください。"
        )
    return Path(from_env)


def capture(args: argparse.Namespace) -> int:
    config_path = resolve_config(args.config)
    sv_config, rt = build_runtime(config_path)
    geo = geometry(sv_config)
    from vspeech.stream_vc.capture import ms_to_samples

    block_len = ms_to_samples(sv_config.block_ms)
    blocks = make_input_blocks(block_len, args.n_blocks, seed=args.input_seed)
    print(f"config: {config_path}")
    print(f"geometry: {geo}")
    print(f"blocks: {blocks.shape} seed={args.seed} seed_mode={args.seed_mode}")

    sv = make_vc(rt, sv_config)
    sv.warmup(args.warmup)
    seed_runtime(args.seed, args.seed_mode)
    out1, lat1 = run_blocks(sv, blocks)

    # The same sequence again, in this same process, from fresh rolling state. This is
    # the within-process half of the determinism question; `compare` is the
    # across-process half.
    sv2 = make_vc(rt, sv_config)
    sv2.warmup(args.warmup)
    seed_runtime(args.seed, args.seed_mode)
    out2, lat2 = run_blocks(sv2, blocks)

    self_noise = judge(out1, out2)
    print()
    print("in-process self-noise (same seed, same input, two StreamingVc instances):")
    print(self_noise.report())
    print()
    print(latency_stats(lat1).line("latency run1"))
    print(latency_stats(lat2).line("latency run2"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        input_blocks=blocks,
        output_blocks=out1,
        latencies_s=lat1,
        seed=np.int64(args.seed),
        seed_mode=np.str_(args.seed_mode),
        input_seed=np.int64(args.input_seed),
        warmup=np.int64(args.warmup),
        provenance_json=np.str_(
            json.dumps(
                provenance(sv_config, rt["target_sample_rate"]),
                sort_keys=True,
                ensure_ascii=False,
            )
        ),
        config_path=np.str_(str(config_path)),
        self_noise_max_abs_diff=np.int64(self_noise.max_abs_diff),
        self_noise_correlation=np.float64(self_noise.correlation),
        self_noise_snr_db=np.float64(self_noise.snr_db),
        # As a name/value pair rather than one key per field: `**{...}` would widen the
        # keyword type and collide with savez's own `allow_pickle: bool` parameter.
        geometry_names=np.array(list(geo), dtype=np.str_),
        geometry_values=np.array(list(geo.values()), dtype=np.float64),
    )
    print(f"\nwrote {args.out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    if not args.baseline.exists():
        raise SystemExit(
            f"baseline がありません: {args.baseline} (先に capture を実行)"
        )
    data = np.load(args.baseline)
    blocks = data["input_blocks"].astype(np.float32)
    reference = data["output_blocks"].astype(np.int16)
    seed = int(data["seed"])
    seed_mode = str(data["seed_mode"])
    warmup = int(data["warmup"])
    base_geo = dict(
        zip(
            (str(name) for name in data["geometry_names"]),
            (float(value) for value in data["geometry_values"]),
            strict=True,
        )
    )

    config_path = resolve_config(args.config)
    sv_config, rt = build_runtime(config_path)
    geo = geometry(sv_config)
    if geo != base_geo:
        raise SystemExit(
            f"geometry が baseline と一致しません: baseline={base_geo} config={geo}"
        )
    mismatches = provenance_mismatches(
        json.loads(str(data["provenance_json"])),
        provenance(sv_config, rt["target_sample_rate"]),
    )
    if mismatches:
        raise SystemExit(
            "baseline と設定/モデルが一致しません (差分を実装の変更と取り違えるため中止):\n  "
            + "\n  ".join(mismatches)
        )

    print(f"baseline: {args.baseline}")
    print(f"config: {config_path}")
    print(f"geometry: {geo} seed={seed} seed_mode={seed_mode}")
    print(
        "baseline in-process self-noise: "
        f"max|diff|={int(data['self_noise_max_abs_diff'])} "
        f"corr={float(data['self_noise_correlation']):.9f} "
        f"snr={float(data['self_noise_snr_db']):.2f}dB"
    )

    sv = make_vc(rt, sv_config)
    sv.warmup(warmup)
    seed_runtime(seed, seed_mode)
    test, latencies = run_blocks(sv, blocks)

    verdict = judge(reference, test)
    print()
    print(verdict.report())
    print()
    print(
        latency_stats(data["latencies_s"].astype(np.float64)).line("latency baseline")
    )
    print(latency_stats(latencies).line("latency this run"))
    return verdict.exit_code


def main() -> int:
    # vspeech's exception and log messages are Japanese; on a redirected stdout Windows
    # would pick cp1252 and die with UnicodeEncodeError before printing them.
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    parser = argparse.ArgumentParser(description="Stream VC numeric baseline harness")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None)

    p_capture = sub.add_parser("capture", parents=[common])
    p_capture.add_argument("--out", type=Path, default=DEFAULT_NPZ)
    p_capture.add_argument("--n-blocks", type=int, default=200)
    p_capture.add_argument("--warmup", type=int, default=3)
    p_capture.add_argument("--seed", type=int, default=0)
    p_capture.add_argument("--input-seed", type=int, default=0)
    p_capture.add_argument("--seed-mode", choices=SEED_MODES, default="both")
    p_capture.set_defaults(func=capture)

    # No threshold overrides on purpose: this subcommand *is* the gate, and a gate whose
    # thresholds the operator can move at the call site is not one.
    p_compare = sub.add_parser("compare", parents=[common])
    p_compare.add_argument("--baseline", type=Path, default=DEFAULT_NPZ)
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
