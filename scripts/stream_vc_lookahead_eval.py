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
