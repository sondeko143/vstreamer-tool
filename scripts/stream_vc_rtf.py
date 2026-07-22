"""RVC ストリーミング VC の RTF 実測ハーネス(M1)。

`--config` の [rvc] セクションを流用してモデルを 1 回ロードし、合成有声信号を
固定ブロックで StreamingVc に流して per-block 遅延と context 込み RTF を
掃引計測する。feasible をマークした表を出し、最低遅延の feasible config を推奨
する。最終の block/context/遅延予算と go/no-go は人が判定する。

  uv run poe stream-vc-rtf --config ./config.toml

純粋な解析ヘルパ(make_voiced_signal / parse_grid / summarize / format_table /
recommend / go_no_go)は numpy のみに依存し、GPU 無し CPU から import・テスト
できる。torch / vspeech / StreamingVc の import は実行部の関数内に遅延させる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def make_voiced_signal(
    rate: int, seconds: float, f0: float = 150.0, seed: int = 0
) -> NDArray[np.float32]:
    """決定論の有声信号(倍音 + 微ビブラート + 微ノイズ)を [-1, 1] で返す。

    f0 抽出器(rmvpe/fcpe)は無音では全フレーム無声(0)を返し計測が不自然に
    なるので、明確な基音を持つ有声信号にする。seed 付きで再現可能。
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
    """`"20,40,80"` を `[20.0, 40.0, 80.0]` に開く。"""
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
    """per-block 遅延列 -> p50/p95/max・RTF(p95基準)・片道遅延・feasible。

    RTF = per-block compute / block 実時間。context は毎 tick 再計算されるので
    その分子に載る(ADR-0053 が指摘した余剰推論)。feasible は RTF_p95 < margin
    (既定 0.5 = transport/jitter/crossfade 用の 2x ヘッドルーム)。片道の
    アルゴリズム遅延 ≈ block_ms + compute_p95。
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
    """feasible の中で片道遅延が最小のもの. 無ければ None。"""
    feasible = [r for r in results if r.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r.latency_ms)


def go_no_go(results: list[BlockResult]) -> bool:
    """feasible が 1 つでもあれば go。"""
    return any(r.feasible for r in results)


def format_table(results: list[BlockResult]) -> str:
    """掃引結果を整列テキスト表にする(feasible 行に [FEASIBLE])。"""
    header = (
        f"{'block':>6} {'ctx':>6} {'f0':>6} "
        f"{'p50ms':>7} {'p95ms':>7} {'maxms':>7} {'RTF':>6} {'lat_ms':>7}  mark"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        mark = "[FEASIBLE]" if r.feasible else ""
        lines.append(
            f"{r.block_ms:>6.0f} {r.context_ms:>6.0f} {r.f0:>6} "
            f"{r.p50_ms:>7.2f} {r.p95_ms:>7.2f} {r.max_ms:>7.2f} "
            f"{r.rtf_p95:>6.2f} {r.latency_ms:>7.1f}  {mark}"
        )
    return "\n".join(lines)
