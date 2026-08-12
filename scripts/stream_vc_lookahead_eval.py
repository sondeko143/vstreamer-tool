"""Measure what lookahead buys, against the batch path as the reference (ADR-0072).

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
(tests/scripts/test_stream_vc_lookahead_eval.py); `main` and below need the rvc extra to
build a real `StreamingVc`. `log_mel` is a plain numpy STFT + mel filterbank
(ADR-0081/ADR-0082).
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
    buffer (ADR-0072), and that buffer starts as zeros -- it only holds real audio once
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
    from vspeech.stream_vc.capture import ms_to_samples
    from vspeech.stream_vc.runner import make_streaming_vc

    block_len = ms_to_samples(sv_config.block_ms)
    # Reuse the production wiring instead of hand-building StreamingVc, so drift here
    # cannot make the eval silently measure a geometry production no longer uses. The
    # only thing this run sweeps is lookahead_ms, hence the model_copy.
    sv = make_streaming_vc(
        rt, sv_config.model_copy(update={"lookahead_ms": lookahead_ms})
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


_MEL_N_FFT = 1024
_MEL_HOP_LENGTH = 256
_MEL_N_MELS = 80


def _hz_to_mel(hz: NDArray[np.float64]) -> NDArray[np.float64]:
    """HTK mel scale (matches torchaudio's `mel_scale="htk"` default)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, rate: int, n_mels: int) -> NDArray[np.float64]:
    """A triangular mel filterbank, shape (n_mels, n_fft // 2 + 1).

    Filters degenerate to all-zero where two adjacent bin edges round to the same FFT
    bin (possible at the low end with a coarse FFT resolution); that filter just
    contributes nothing to the sum below rather than dividing by zero. At the
    n_fft=1024/n_mels=80 shape this file uses, this leaves exactly 1 of the 80 channels
    dead at a 48kHz rate (none at 16kHz). Harmless for the ranking use below -- a
    constant-zero channel contributes the same (zero) term to both sides of every
    `spectral_distance` comparison, so it cancels -- but it is a real difference from
    torchaudio's filterbank, which has no dead channel at this shape. See `log_mel`'s
    docstring for the full non-interchangeability picture this contributes to.
    """
    mel_bounds = _hz_to_mel(np.array([0.0, rate / 2.0], dtype=np.float64))
    mel_edges = np.linspace(mel_bounds[0], mel_bounds[1], n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_edges) / rate).astype(np.int64)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (
                right - center
            )
    return fb


def log_mel(x: NDArray[np.int16], rate: int) -> NDArray[np.float64]:
    """(n_mels, frames) log-mel on a 10*log10 scale.

    A pure-numpy STFT (Hann window, reflect-padded, center-aligned) feeding a triangular
    HTK mel filterbank, standing in for
    `torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=256, n_mels=80)` now
    that torch/torchaudio are out of the dependency table (ADR-0081/ADR-0082). This is
    only ever used as a ranking distance metric between lookahead settings -- the module
    docstring: "the numbers rank the settings and the wavs decide" -- so internal
    consistency between the reference and candidate calls within a single run is what
    matters, and that consistency holds (both are always computed by this same function).

    It is NOT numerically interchangeable with the torchaudio mel it replaced, though:
    measured per-bin |delta| mean 0.49 dB at 16kHz and 2.09 dB at 48kHz, and the derived
    `spectral_distance` shifts by mean/p95 +0.30/+0.26 dB at 48kHz -- the operative rate
    for this file, since callers pass `rate = rt["target_sample_rate"]` (the RVC model's
    own rate, commonly 48kHz), not 16kHz. That is on the same order as the effect
    docs/adr/0072-stream-vc-lookahead.md's own logmel table reasons about (a 0.78 dB p95
    spread and a 0.28 dB mean spread across its lookahead grid) -- so **figures recorded
    by earlier (torchaudio-based) runs of this script are not comparable to figures from
    this (numpy-based) implementation**; re-run the eval to get numbers on the same
    footing as any current comparison. See also `_mel_filterbank`'s docstring for the one
    structural difference (a dead channel at 48kHz that torchaudio's filterbank does not
    have).
    """
    wav = x.astype(np.float64) / 32768.0
    pad = _MEL_N_FFT // 2
    padded = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = max(0, 1 + (padded.shape[0] - _MEL_N_FFT) // _MEL_HOP_LENGTH)
    window = np.hanning(_MEL_N_FFT + 1)[:-1]  # periodic Hann, matches torch.hann_window
    starts = np.arange(n_frames) * _MEL_HOP_LENGTH
    frames = np.stack([padded[s : s + _MEL_N_FFT] for s in starts])
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2  # (n_frames, n_fft//2+1)
    mel_fb = _mel_filterbank(_MEL_N_FFT, rate, _MEL_N_MELS)  # (n_mels, n_fft//2+1)
    mel_spec = mel_fb @ power.T  # (n_mels, n_frames)
    return 10.0 * np.log10(mel_spec + 1e-10)


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

    grid = parse_grid(args.lookahead)
    if not grid:
        raise SystemExit("--lookahead に有効な値が1つもない")

    sv_config, rt = load_config_and_runtime(args.config)
    rate = rt["target_sample_rate"]
    signal = load_wav_16k(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = run_batch_reference(rt, signal, args.seed)
    write_wav(args.out_dir / "batch_reference.wav", ref, rate)
    ref_f = ref.astype(np.float32)

    # Hoisted out of the loop and sized off the widest grid point, not each row's own
    # lookahead_ms. This table's only job is to rank settings against each other, so
    # every row must be compared over the *same* span of audio. Recomputing `skip` per
    # row would make the 160ms row drop 160ms more of the head than the 0ms row -- and
    # the head is exactly where the streaming path is worst (nearest the warm-up
    # boundary), so larger lookahead would systematically drop more of the worst
    # frames and look better than it is. `warmup_skip_samples` grows monotonically
    # with lookahead_ms, so `max(grid)` is the smallest skip that still clears every
    # row's own per-setting minimum.
    skip = warmup_skip_samples(
        sv_config.context_ms, max(grid), sv_config.block_ms, rate
    )
    required_s = 1.0
    remaining_s = (ref_f.shape[0] - skip) / rate
    if remaining_s < required_s:
        raise SystemExit(
            f"入力 wav が短すぎる: warmup skip {skip / rate:.2f}秒"
            f"(lookahead grid の最大値 {max(grid):.0f}ms 基準)を引くと"
            f"比較に残る区間が{max(0.0, remaining_s):.2f}秒しかない"
            f"(最低 {required_s:.0f}秒必要)。"
            f"入力 wav は{ref_f.shape[0] / rate:.2f}秒しかなかった。より長い wav を使うこと"
        )
    ref_t = ref_f[skip:]

    rows: list[dict[str, Any]] = []
    for lookahead_ms in grid:
        stream, durations, delay = run_streaming(
            rt, sv_config, signal, lookahead_ms, args.seed
        )
        write_wav(args.out_dir / f"lookahead_{lookahead_ms:.0f}ms.wav", stream, rate)
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
