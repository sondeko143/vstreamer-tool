"""Measure what lookahead buys, against the batch path as the reference (ADR-0070).

Converts one wav both ways with **the same model** -- streaming at several
`lookahead_ms` settings, and the batch `change_voice` that has full two-sided context --
then reports a log-mel distance per setting plus the wavs for an ear A/B. The batch output
is a ceiling to approach, not perceptual ground truth, so the numbers rank the settings
and the wavs decide.

Unlike `stream_vc_rtf.py`, this reads **[stream_vc.rvc]** directly, so there is no need to
mirror it into [rvc].

    uv sync --all-extras
    uv run poe stream-vc-lookahead-eval --config ./config.toml --input voice.wav \
        --lookahead 0,40,80,160 --out-dir ./lookahead_eval

The helpers above `main` are pure numpy so they can be unit tested on CPU with no model
(scripts/tests/test_stream_vc_lookahead_eval.py); everything touching torch lives below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# HuBERT's receptive field truncates about 320 input samples off the render's tail at
# 16kHz (see the `_emit_with_crossfade` / `_emit_no_crossfade` docstrings in
# vspeech/lib/stream_vc.py) -- that much of the raw emit delay is render truncation,
# not real right-context, so it has to be subtracted back out when turning the measured
# delay into a right-context figure.
_HUBERT_TRUNCATION_MS = 20.0


def frame_energy(x: NDArray[np.float32], hop: int) -> NDArray[np.float64]:
    """Per-hop RMS, used for the coarse alignment search."""
    n = x.shape[0] // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = x[: n * hop].astype(np.float64).reshape(n, hop)
    return np.sqrt(np.mean(frames**2, axis=1))


def best_offset(
    ref: NDArray[np.floating[Any]],
    test: NDArray[np.floating[Any]],
    hint: int,
    search: int,
) -> int:
    """The offset o in [hint-search, hint+search] maximising the normalized correlation
    between ref[:m] and test[o:o+m]. Brute force -- the caller keeps `search` small."""
    best, best_score = hint, -np.inf
    for o in range(hint - search, hint + search + 1):
        if o < 0 or o >= test.shape[0]:
            continue
        m = min(ref.shape[0], test.shape[0] - o)
        if m <= 0:
            continue
        a = ref[:m].astype(np.float64)
        b = test[o : o + m].astype(np.float64)
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 0.0:
            continue
        score = float(a @ b) / den
        if score > best_score:
            best_score, best = score, o
    return best


def align_offset(
    ref: NDArray[np.float32],
    test: NDArray[np.float32],
    hint: int,
    coarse_hop: int = 256,
    coarse_search: int = 64,
    fine_search: int = 240,
    excerpt: int = 80000,
) -> int:
    """How many samples `test` lags `ref` by, searched coarse (energy envelope) then fine
    (waveform).

    A single brute-force pass over the whole signal would be O(search * n); the coarse
    stage runs on one value per `coarse_hop` samples and the fine stage only on the first
    `excerpt` samples, which keeps this a couple of seconds offline.
    """
    coarse = (
        best_offset(
            frame_energy(ref, coarse_hop),
            frame_energy(test, coarse_hop),
            hint // coarse_hop,
            coarse_search,
        )
        * coarse_hop
    )
    m = min(excerpt, ref.shape[0])
    return best_offset(ref[:m], test, coarse, fine_search)


def spectral_distance(
    ref_logmel: NDArray[np.float64],
    test_logmel: NDArray[np.float64],
    floor_db: float = -40.0,
) -> tuple[float, float]:
    """(mean, p95) of the per-frame log-mel L2 distance, over the frames the reference
    actually has energy in.

    Both arrays are (n_mels, frames) on a 10*log10 scale. Frames more than `floor_db`
    below the reference's loudest frame are excluded, so silence does not dilute the
    number into meaninglessness.
    """
    m = min(ref_logmel.shape[1], test_logmel.shape[1])
    if m == 0:
        return 0.0, 0.0
    a = ref_logmel[:, :m]
    b = test_logmel[:, :m]
    energy = a.mean(axis=0)
    mask = energy >= (energy.max() + floor_db)
    if not mask.any():
        mask = np.ones_like(energy, dtype=bool)
    d = np.sqrt(np.mean((a - b) ** 2, axis=0))[mask]
    return float(d.mean()), float(np.percentile(d, 95))


def write_wav(path: Path, samples: NDArray[np.int16], rate: int) -> None:
    """Write mono int16 PCM. stdlib `wave` -- no torchaudio backend dependency."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def warmup_skip_samples(
    context_ms: float, lookahead_ms: float, block_ms: float, rate: int
) -> int:
    """How many samples of the run's start to drop before comparing it to the batch
    reference.

    `run_streaming` builds `StreamingVc` with a `context_ms + lookahead_ms` long context
    buffer (ADR-0070), and that buffer starts as zeros -- it only holds real audio once
    that much real time has rolled through it. The batch reference has no such warm-up,
    so both must have this span dropped before comparing them. The span therefore grows
    with the lookahead: skipping a fixed amount leaves partially-cold output in the tail
    of every run with `lookahead_ms > 0`, and being an extreme-value statistic,
    `logmel_p95` is exactly what that would distort.
    """
    return round((context_ms + lookahead_ms + block_ms) / 1000.0 * rate)


def right_context_ms(delay_samples: int, rate: int) -> float:
    """Convert a measured emit delay into the right-context this run actually bought.

    `delay_samples` (`StreamingVc.emit_delay_samples`) is the analytic emit delay, which
    mixes real right-context with a fixed HuBERT render-truncation cost
    (`_HUBERT_TRUNCATION_MS`, ~320 input samples at 16kHz -- see the `_emit_delay` /
    `_emit_with_crossfade` docstrings in vspeech/lib/stream_vc.py). Subtracting the
    truncation back out is what makes this a genuine per-run measurement rather than an
    assumption baked in from the geometry: `crossfade_ms` and `sola_search_ms` are both
    user-settable (`ge=0`), so a config with non-default values would otherwise make the
    printed right(ms) column silently wrong.
    """
    return delay_samples * 1000.0 / rate - _HUBERT_TRUNCATION_MS


def load_config_and_runtime(config_path: Path) -> tuple[Any, dict[str, Any]]:
    """Read the config and build the streaming runtime from **[stream_vc.rvc]**.

    Reusing `build_stream_vc_runtime` means the reference and every streaming run share
    one model load, and that the comparison is against the same weights the streaming path
    would really use.
    """
    from vspeech.config import Config
    from vspeech.stream_vc.runner import build_stream_vc_runtime

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    return config.stream_vc, build_stream_vc_runtime(config.stream_vc)


def run_batch_reference(
    rt: dict[str, Any], signal_16k: NDArray[np.float32], seed: int
) -> NDArray[np.int16]:
    """The batch `change_voice` over the whole signal = the two-sided-context ceiling.

    `run_change_voice` takes the same runtime dict shape `build_stream_vc_runtime`
    produces, so no second model load is needed. Seeded, because the RVC synthesizer is
    stochastic by design but reproducible under a seed.
    """
    from scripts.capture_change_voice_golden import run_change_voice
    from scripts.capture_change_voice_golden import seed_all

    frames = (np.clip(signal_16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    seed_all(seed)
    return run_change_voice(rt, frames, 16000)


def run_streaming(
    rt: dict[str, Any],
    sv_config: Any,
    signal_16k: NDArray[np.float32],
    lookahead_ms: float,
    seed: int,
) -> tuple[NDArray[np.int16], NDArray[np.float64], int]:
    """Convert the signal block by block at one lookahead setting.

    Returns (emitted int16, per-block seconds, emit_delay_samples).
    """
    import time

    from scripts.capture_change_voice_golden import seed_all
    from vspeech.lib.stream_vc import StreamingVc
    from vspeech.stream_vc.capture import ms_to_samples

    block_len = ms_to_samples(sv_config.block_ms)
    sv = StreamingVc(
        rvc_config=rt["rvc_config"],
        device=rt["device"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_session=rt["f0_session"],
        target_sample_rate=rt["target_sample_rate"],
        f0_enabled=rt["f0_enabled"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        block_len=block_len,
        context_len=ms_to_samples(sv_config.context_ms + lookahead_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
        sola_search_len=ms_to_samples(sv_config.sola_search_ms),
        lookahead_len=ms_to_samples(lookahead_ms),
    )
    sv.warmup()
    seed_all(seed)
    emits: list[NDArray[np.int16]] = []
    durations: list[float] = []
    for i in range(signal_16k.shape[0] // block_len):
        block = signal_16k[i * block_len : (i + 1) * block_len]
        t0 = time.perf_counter()
        emits.append(sv.process_block(block))
        durations.append(time.perf_counter() - t0)
    return (
        np.concatenate(emits),
        np.array(durations, dtype=np.float64),
        sv.emit_delay_samples,
    )


def log_mel(x: NDArray[np.int16], rate: int) -> NDArray[np.float64]:
    """(n_mels, frames) log-mel on a 10*log10 scale."""
    import torch
    import torchaudio.transforms as T

    mel = T.MelSpectrogram(sample_rate=rate, n_fft=1024, hop_length=256, n_mels=80)
    spec = mel(torch.from_numpy(x.astype(np.float32) / 32768.0))
    return (10.0 * torch.log10(spec + 1e-10)).numpy().astype(np.float64)


def format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "  L(ms) right(ms)  added(ms)  window(ms)  rtf_p50  rtf_p95  "
        "align_err  logmel_mean  logmel_p95"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['lookahead_ms']:>7.0f} {r['right_ms']:>9.1f} "
            f"{r['added_ms']:>10.0f} {r['window_ms']:>11.0f} "
            f"{r['rtf_p50']:>8.2f} {r['rtf_p95']:>8.2f} "
            f"{r['align_err']:>10d} {r['logmel_mean']:>12.3f} "
            f"{r['logmel_p95']:>11.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    # The description and the ranking guidance below are Japanese. When stdout/stderr is
    # a pipe/redirect, Windows picks cp1252 by default and this dies with
    # UnicodeEncodeError before anything is printed (the same encoding guard this project
    # keeps needing; same shape as stream_vc_rtf.py / convert_hubert.py /
    # export_hubert_onnx.py). typeshed types sys.stdout/stderr as a TextIO without
    # .reconfigure, but at runtime they are TextIOWrapper.
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    import argparse
    import json

    from scripts.stream_vc_rtf import load_wav_16k
    from scripts.stream_vc_rtf import parse_grid

    parser = argparse.ArgumentParser(
        description="lookahead ごとに streaming VC の出力をバッチ変換と比較する"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="入力 wav")
    parser.add_argument(
        "--lookahead", default="0,40,80,160", help="lookahead_ms のリスト"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./lookahead_eval"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    sv_config, rt = load_config_and_runtime(args.config)
    rate = rt["target_sample_rate"]
    signal = load_wav_16k(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = run_batch_reference(rt, signal, args.seed)
    write_wav(args.out_dir / "batch_reference.wav", ref, rate)
    ref_f = ref.astype(np.float32)

    rows: list[dict[str, Any]] = []
    for lookahead_ms in parse_grid(args.lookahead):
        stream, durations, delay = run_streaming(
            rt, sv_config, signal, lookahead_ms, args.seed
        )
        write_wav(args.out_dir / f"lookahead_{lookahead_ms:.0f}ms.wav", stream, rate)
        skip = warmup_skip_samples(
            sv_config.context_ms, lookahead_ms, sv_config.block_ms, rate
        )
        ref_t = ref_f[skip:]
        stream_t = stream.astype(np.float32)[skip:]
        offset = align_offset(ref_t, stream_t, hint=delay)
        aligned = stream_t[offset : offset + ref_t.shape[0]]
        m = min(aligned.shape[0], ref_t.shape[0])
        mean, p95 = spectral_distance(
            log_mel(ref_t[:m].astype(np.int16), rate),
            log_mel(aligned[:m].astype(np.int16), rate),
        )
        block_seconds = sv_config.block_ms / 1000.0
        rows.append(
            {
                "lookahead_ms": lookahead_ms,
                "right_ms": right_context_ms(delay, rate),
                "added_ms": lookahead_ms,
                "window_ms": sv_config.context_ms + lookahead_ms + sv_config.block_ms,
                "rtf_p50": float(np.percentile(durations, 50)) / block_seconds,
                "rtf_p95": float(np.percentile(durations, 95)) / block_seconds,
                # Should land near 0: the analytic emit delay is what we aligned by.
                "align_err": int(offset - delay),
                "logmel_mean": mean,
                "logmel_p95": p95,
            }
        )
        print(f"lookahead={lookahead_ms:.0f}ms done", flush=True)

    print()
    print(format_table(rows))
    print()
    print(f"wav: {args.out_dir}  (batch_reference.wav と聞き比べること)")
    print("logmel 距離が小さいほどバッチ変換に近い。ただしこれは代理指標なので、")
    print("順位付けに使い、最終判断は wav の耳 A/B で行うこと。")
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
