# RVC ストリーミング VC — M1: RTF 実測ハーネス Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固定ブロックをステートフル VC(rolling 左文脈)に通し、対象 GPU で per-block 遅延と context 込み RTF を掃引計測して go/no-go を判定できるハーネスを作る。

**Architecture:** 再利用コア `vspeech/lib/stream_vc.py`(`StreamingVc.process_block` = `[context|block]` を組み立て、既存 `change_voice` の内部部品 [HuBERT特徴量 / f0 / infer / int16化] をそのまま呼び、ブロック相当の出力だけ切り出して rolling context を更新)と、計測ハーネス `scripts/stream_vc_rtf.py`(`--config` の `[rvc]` を流用してモデルを1回ロード、合成有声信号を固定ブロックで流し、CUDA 同期を挟んで per-block 遅延を計測、p50/p95/max・RTF・片道遅延を掃引表で提示、feasible をマークし最低遅延の feasible config を推奨)。クロスフェード連続性の音質は M2 送り。M1 は compute の計測に集中する。

**Tech Stack:** Python 3.14 / uv / numpy / torch(CUDA)/ onnxruntime-gpu(rvc extra)/ pytest(asyncio_mode=auto)/ poethepoet。

## Global Constraints

- **Python 3.14 のみ**(`requires-python = ">=3.14,<3.15"`)。3.13 以下へ下げない。
- **既存 `change_voice` 経路は無改変**。`vspeech/lib/rvc.py` / `vspeech/worker/vc.py` / routing(`vspeech/lib/command.py`, `vspeech/shared_context.py`)を**編集しない**。M1 は `rvc.py` の関数を import して再利用するだけ(ADR-0050 / ADR-0053)。
- **M1 スコープ外(このプランで作らない)**: クロスフェード連続性の音質/耳確認(M2)、マシン間トランスポート(ADR-0051)、`[stream_vc]` config・preflight・GUI(ADR-0054)、マイクキャプチャ(ADR-0052)。
- **重い import は遅延**(`torch` / `onnxruntime` / `vspeech.lib.rvc` / `vspeech.config`)。純粋ヘルパ(numpy のみ)は rvc extra 無しの CPU でも import できること。手本は `scripts/capture_change_voice_golden.py`(numpy を top-level、torch/vspeech を関数内で import)。型注釈は `from __future__ import annotations` + `if TYPE_CHECKING:` で重い import を実行時に持ち込まない。
- **import は 1 行 1 個**(ruff `force-single-line`)。`ruff format` / `ruff check` / `ty check` が緑であること。
- **GPU/モデル依存テストのゲート**は既存 golden テストと同じにする(`tests/test_change_voice_golden.py`): `torch.cuda.is_available()` かつ 環境変数 `VSPEECH_RVC_GOLDEN_CONFIG` が実在する TOML を指すとき以外は `skip`。**ハーネス本体(スクリプト)は `--config` を取る**が、**テストのゲートだけは pytest 慣習でこの環境変数を流用する**(スクリプトに環境変数は要らない — これは pytest がテスト単位で引数を渡せない事情の回避策で、テストにのみ当てはまる)。
- **f0 抽出器の対称契約**: `rmvpe`/`fcpe` はどちらも「波形 → 閾値 voicing → f0」で無声=0。streaming コアは `reflect-pad しない`(実左文脈が左文脈を与える)。既定 `quality=zero` の `change_voice` も reflect-pad 量 0 なので compute は等価。
- **決定論**: 合成信号は seed 付き(`np.random.default_rng(seed)`)。`time.perf_counter` を使う(`time.time` ではなく)。
- **コミット**: 作業ブランチ `feat/rvc-streaming-vc`(既存、main ではない)へ直接コミット。各コミットメッセージ末尾に:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- **Create `vspeech/lib/stream_vc.py`** — 再利用コア。純粋ヘルパ `next_context` / `slice_block_output`(numpy/torch 両対応、module-level、軽い)+ `StreamingVc` クラス(`__init__` / `warmup` / `process_block`、重い import は遅延)。
- **Create `scripts/stream_vc_rtf.py`** — 計測ハーネス。純粋な解析ヘルパ(`make_voiced_signal` / `parse_grid` / `BlockResult` / `summarize` / `format_table` / `recommend` / `go_no_go`、numpy のみ top-level)+ 実行部(`load_shared_runtime` / `make_f0_session` / `next_block` / `run_sweep` / `main`、重い import は遅延)。
- **Create `tests/test_stream_vc.py`** — コアの純粋ヘルパ(CPU)+ `StreamingVc` の GPU-gated スモーク。
- **Create `tests/test_stream_vc_rtf.py`** — ハーネスの純粋解析ヘルパ(CPU)+ エントリポイント GPU-gated スモーク。
- **Modify `poe_tasks.toml`** — `stream-vc-rtf` タスクを追加。

---

## Task 1: streaming コアの純粋ヘルパ(context スライド / 出力切り出し)

**Files:**
- Create: `vspeech/lib/stream_vc.py`
- Test: `tests/test_stream_vc.py`

**Interfaces:**
- Produces:
  - `next_context(seq, context_len: int)` — `seq` の末尾 `context_len` 要素を返す。`context_len == 0` のとき空を返す(`seq[-0:]` が全体になる罠を避ける)。numpy 配列でも torch tensor でも動く(`len(seq)` ベースのスライス)。
  - `slice_block_output(out, block_len: int, seq_len: int)` — `out` の末尾 `round(len(out) * block_len / seq_len)` 要素(=ブロック相当区間)を返す。`block_len == 0` なら `out` をそのまま返す。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc.py`:
```python
import numpy as np

from vspeech.lib.stream_vc import next_context
from vspeech.lib.stream_vc import slice_block_output


def test_next_context_returns_tail():
    seq = np.arange(5)
    assert list(next_context(seq, 2)) == [3, 4]


def test_next_context_zero_is_empty():
    seq = np.arange(3)
    assert len(next_context(seq, 0)) == 0


def test_slice_block_output_takes_block_tail():
    out = np.arange(10)
    # block is last block_len/seq_len = 2/10 of the sequence -> last 2 output samples
    assert list(slice_block_output(out, block_len=2, seq_len=10)) == [8, 9]


def test_slice_block_output_rounds_proportionally():
    out = np.arange(100)
    # 40 / (200+40) of 100 -> round(16.67) = 17
    assert len(slice_block_output(out, block_len=40, seq_len=240)) == 17
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.stream_vc'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/stream_vc.py`:
```python
"""ストリーミング VC の再利用コア(ADR-0053)。

固定長ブロックを rolling 左文脈と連結してステートフル変換する。既存の
`change_voice` の内部部品(HuBERT 特徴量 / f0 / infer / int16化)をそのまま
再利用し、発話系の `change_voice` 経路は無改変で温存する。M1 はこのコアの
per-block 計測(RTF)に集中し、クロスフェード連続性の音質は M2 で足す。

純粋ヘルパ(next_context / slice_block_output)は numpy でも torch tensor でも
動くよう `len(seq)` ベースにしてあり、torch 無し・rvc extra 無しの CPU でも
import できる(重い import は StreamingVc のメソッド内でのみ行う)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession

    from vspeech.config import RvcConfig
    from vspeech.lib.rvc import HubertSession


def next_context(seq, context_len: int):
    """`seq` の末尾 `context_len` 要素(次 tick の左文脈)。

    `context_len == 0` のとき `seq[-0:]` は全体を返してしまうので、明示的に
    空スライスにする。`len(seq)` ベースなので numpy/torch 双方で同じ挙動。
    """
    if context_len <= 0:
        return seq[:0]
    return seq[len(seq) - context_len :]


def slice_block_output(out, block_len: int, seq_len: int):
    """`out` のうち、直近ブロック相当(末尾 block_len/seq_len)の区間。

    infer は `[context|block]` 全体の波形を返すので、ブロック相当の末尾だけ
    採用する。正確なシーム整列(等電力クロスフェード)は M2 の担当で、ここは
    比率で切り出す近似(RTF 計測には出力長は影響しない)。
    """
    if block_len <= 0:
        return out
    block_out = round(len(out) * block_len / seq_len)
    if block_out <= 0:
        return out
    return out[len(out) - block_out :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/stream_vc.py tests/test_stream_vc.py
git commit -m "feat(stream-vc): add pure context-slide/output-slice helpers for streaming core

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: ハーネスの純粋解析ヘルパ(合成信号 / 掃引解析 / 表)

**Files:**
- Create: `scripts/stream_vc_rtf.py`
- Test: `tests/test_stream_vc_rtf.py`

**Interfaces:**
- Consumes: (なし — 純粋 numpy)
- Produces:
  - `make_voiced_signal(rate: int, seconds: float, f0: float = 150.0, seed: int = 0) -> NDArray[np.float32]` — 有声(倍音+微ビブラート)+微ノイズの決定論信号 `[-1,1]`。f0 抽出器が無声で早期離脱しないよう有声にする。
  - `parse_grid(text: str) -> list[float]` — `"20,40,80"` → `[20.0, 40.0, 80.0]`。
  - `@dataclass BlockResult` — フィールド `block_ms, context_ms, f0, p50_ms, p95_ms, max_ms, rtf_p95, latency_ms, feasible`。
  - `summarize(latencies_s: list[float], block_seconds: float, margin: float, block_ms: float, context_ms: float, f0: str) -> BlockResult` — RTF=p95/block_seconds、片道遅延≈block_ms+p95、feasible=`rtf_p95 < margin`。
  - `format_table(results: list[BlockResult]) -> str` — 整列テキスト表(feasible 行に `[FEASIBLE]`)。
  - `recommend(results: list[BlockResult]) -> BlockResult | None` — feasible の中で `latency_ms` 最小。無ければ `None`。
  - `go_no_go(results: list[BlockResult]) -> bool` — feasible が 1 つでもあれば `True`。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_rtf.py`:
```python
import numpy as np

from scripts.stream_vc_rtf import BlockResult
from scripts.stream_vc_rtf import format_table
from scripts.stream_vc_rtf import go_no_go
from scripts.stream_vc_rtf import make_voiced_signal
from scripts.stream_vc_rtf import parse_grid
from scripts.stream_vc_rtf import recommend
from scripts.stream_vc_rtf import summarize


def test_make_voiced_signal_shape_and_range():
    sig = make_voiced_signal(rate=16000, seconds=0.5, seed=0)
    assert sig.dtype == np.float32
    assert sig.shape == (8000,)
    assert np.max(np.abs(sig)) <= 1.0
    assert np.any(sig != 0.0)  # not silent -> f0 extractor won't early-out


def test_make_voiced_signal_is_deterministic():
    a = make_voiced_signal(16000, 0.2, seed=7)
    b = make_voiced_signal(16000, 0.2, seed=7)
    assert np.array_equal(a, b)


def test_parse_grid():
    assert parse_grid("20,40,80") == [20.0, 40.0, 80.0]
    assert parse_grid("100") == [100.0]


def test_summarize_computes_rtf_and_feasible():
    # block of 40ms -> 0.04s wall. compute ~10ms p95 -> RTF 0.25 < 0.5 -> feasible.
    lat = [0.008, 0.009, 0.010, 0.011, 0.012]
    r = summarize(lat, block_seconds=0.04, margin=0.5, block_ms=40, context_ms=200, f0="rmvpe")
    assert r.feasible is True
    assert r.rtf_p95 < 0.5
    # latency ~ block(40) + p95(~12) ms
    assert 45.0 < r.latency_ms < 60.0


def test_summarize_infeasible_when_rtf_exceeds_margin():
    lat = [0.030, 0.031, 0.032]  # ~31ms vs 40ms block -> RTF ~0.79 > 0.5
    r = summarize(lat, block_seconds=0.04, margin=0.5, block_ms=40, context_ms=800, f0="rmvpe")
    assert r.feasible is False


def _mk(feasible: bool, latency_ms: float) -> BlockResult:
    return BlockResult(
        block_ms=40, context_ms=200, f0="rmvpe", p50_ms=1, p95_ms=1,
        max_ms=1, rtf_p95=0.1, latency_ms=latency_ms, feasible=feasible,
    )


def test_recommend_picks_lowest_latency_feasible():
    results = [_mk(True, 60), _mk(False, 30), _mk(True, 45)]
    best = recommend(results)
    assert best is not None and best.latency_ms == 45


def test_recommend_none_when_no_feasible():
    assert recommend([_mk(False, 30)]) is None


def test_go_no_go():
    assert go_no_go([_mk(False, 30), _mk(True, 45)]) is True
    assert go_no_go([_mk(False, 30)]) is False


def test_format_table_marks_feasible():
    out = format_table([_mk(True, 45), _mk(False, 30)])
    assert "[FEASIBLE]" in out
    assert "rmvpe" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_rtf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.stream_vc_rtf'`

- [ ] **Step 3: Write minimal implementation**

`scripts/stream_vc_rtf.py`(このステップでは純粋ヘルパのみ。実行部は Task 4 で追加):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_rtf.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/stream_vc_rtf.py tests/test_stream_vc_rtf.py
git commit -m "feat(stream-vc): add RTF-harness analysis helpers (signal/summarize/table)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `StreamingVc` GPU コア(`change_voice` 内部部品の再利用)

**Files:**
- Modify: `vspeech/lib/stream_vc.py`(`StreamingVc` クラスを追加。Task 1 の純粋ヘルパはそのまま)
- Test: `tests/test_stream_vc.py`(GPU-gated スモークを追加)

**Interfaces:**
- Consumes: `next_context` / `slice_block_output`(Task 1); `vspeech.lib.rvc` の `_extract_hubert_feats` / `_select_pitch` / `_align_pitch_to_feats` / `_to_int16` / `_is_model_half` / `infer`; `scripts.capture_change_voice_golden.build_rvc_runtime`(スモークのランタイム構築)。
- Produces:
  - `class StreamingVc(rvc_config, device, hubert_model, session, f0_session, target_sample_rate, f0_enabled, emb_output_layer, use_final_proj, block_len, context_len)` — `block_len` / `context_len` は 16kHz サンプル数(固定 → shape 固定 → warmup 1 回)。`rvc_config.f0_extractor_type` は `f0_session` と一致していること。
  - `StreamingVc.warmup(self, n: int = 3) -> None` — zeros ブロックで ONNX グラフを構築し、context を zeros に戻す。
  - `StreamingVc.process_block(self, block: NDArray[np.float32]) -> NDArray[np.int16]` — 長さ `block_len` の 16kHz float32 `[-1,1]` を受け、変換 int16 のブロック相当区間を返す。context を更新。

**重要**: `vspeech/lib/rvc.py` は編集しない。private helper を import して使う(ADR-0053 が意図する内部部品の再利用)。`change_voice` は `quality=zero` で reflect-pad 量 0 なので、reflect-pad しない streaming コアと compute は等価。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc.py` に追記:
```python
import os
from pathlib import Path

import pytest


_CONFIG_ENV = "VSPEECH_RVC_GOLDEN_CONFIG"
_config_path = os.environ.get(_CONFIG_ENV)
_GOLDEN_CONFIG = Path(_config_path) if _config_path else None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


_gpu_gate = pytest.mark.skipif(
    not _cuda_available()
    or _GOLDEN_CONFIG is None
    or not _GOLDEN_CONFIG.exists(),
    reason=f"CUDA / ${_CONFIG_ENV} config not available",
)


@_gpu_gate
def test_streaming_vc_process_block_shape_and_finite():
    from scripts import capture_change_voice_golden as cap

    from vspeech.lib.stream_vc import StreamingVc

    assert _GOLDEN_CONFIG is not None  # gate guarantees; narrows for ty
    rt = cap.build_rvc_runtime(_GOLDEN_CONFIG)

    block_len = 640  # 40ms @ 16k
    context_len = 3200  # 200ms @ 16k
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
        context_len=context_len,
    )
    sv.warmup()

    import numpy as np

    from scripts.stream_vc_rtf import make_voiced_signal

    signal = make_voiced_signal(16000, 1.0, seed=0)
    out1 = sv.process_block(signal[:block_len])
    out2 = sv.process_block(signal[block_len : 2 * block_len])

    assert out1.dtype == np.int16 and out2.dtype == np.int16
    assert out1.shape[0] > 0 and out2.shape[0] > 0
    assert np.all(np.isfinite(out1)) and np.all(np.isfinite(out2))
```

- [ ] **Step 2: Run test to verify it fails**

Run(GPU + 実 config のあるマシンで): `VSPEECH_RVC_GOLDEN_CONFIG=/path/to/config.toml uv run --all-extras pytest tests/test_stream_vc.py::test_streaming_vc_process_block_shape_and_finite -v`
Expected: FAIL — `ImportError: cannot import name 'StreamingVc'`
(GPU/config が無いマシンでは `SKIPPED` になる。その場合はコアを実装後、CPU の純粋ヘルパテストが緑であることだけ確認して次へ。)

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/stream_vc.py` の `slice_block_output` の後に追記:
```python
class StreamingVc:
    """固定ブロック + rolling 左文脈のステートフル VC(ADR-0053)。

    毎 tick `[context | block]`(16kHz)を組み立て、既存 `change_voice` の内部
    部品で HuBERT 特徴量 -> f0 -> infer -> int16 を通し、ブロック相当の出力だけ
    採用して context を更新する。block_len / context_len を固定するので入力
    shape が固定になり、warmup は 1 回で済む(以後 re-autotune なし)。

    重い依存(torch / rvc の内部部品)はここで初めて import する。`rvc_config`
    の f0_extractor_type は渡す `f0_session` と一致していること。
    """

    def __init__(
        self,
        rvc_config: RvcConfig,
        device: torch.device,
        hubert_model: HubertSession,
        session: InferenceSession,
        f0_session: InferenceSession | None,
        target_sample_rate: int,
        f0_enabled: bool,
        emb_output_layer: int,
        use_final_proj: bool,
        block_len: int,
        context_len: int,
    ) -> None:
        import torch

        from vspeech.lib.rvc import _is_model_half

        self.rvc_config = rvc_config
        self.device = device
        self.hubert_model = hubert_model
        self.session = session
        self.f0_session = f0_session
        self.target_sample_rate = target_sample_rate
        self.f0_enabled = f0_enabled
        self.emb_output_layer = emb_output_layer
        self.use_final_proj = use_final_proj
        self.block_len = block_len
        self.context_len = context_len
        self._is_half = _is_model_half(session)
        self._sid = torch.tensor(0, device=device).unsqueeze(0).long()
        self._context = torch.zeros(context_len, device=device, dtype=torch.float32)

    def warmup(self, n: int = 3) -> None:
        """zeros ブロックで ONNX グラフ / CUDA カーネルを先に構築する。

        block_len は固定なので、実値でなく shape さえ通れば以後 stall しない。
        warmup 後は context を zeros に戻す。
        """
        import numpy as np

        zeros = np.zeros(self.block_len, dtype=np.float32)
        for _ in range(n):
            self.process_block(zeros)
        self._reset_context()

    def _reset_context(self) -> None:
        import torch

        self._context = torch.zeros(
            self.context_len, device=self.device, dtype=torch.float32
        )

    def process_block(self, block: NDArray[np.float32]) -> NDArray[np.int16]:
        """長さ block_len の 16kHz float32 [-1,1] を変換し int16 ブロックを返す。"""
        import numpy as np
        import torch

        from vspeech.lib.rvc import _align_pitch_to_feats
        from vspeech.lib.rvc import _extract_hubert_feats
        from vspeech.lib.rvc import _select_pitch
        from vspeech.lib.rvc import _to_int16
        from vspeech.lib.rvc import infer

        block_t = torch.from_numpy(np.ascontiguousarray(block)).to(
            device=self.device, dtype=torch.float32
        )
        seq = torch.cat([self._context, block_t])  # 固定長 L = context_len + block_len

        feats = _extract_hubert_feats(
            hubert_model=self.hubert_model,
            audio_pad=seq,
            device=self.device,
            emb_output_layer=self.emb_output_layer,
            use_final_proj=self.use_final_proj,
        )

        p_len = seq.shape[0] // self.rvc_config.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        pitch, pitchf = _select_pitch(
            seq,
            self.rvc_config,
            self.f0_enabled,
            p_len,
            self.device,
            self.f0_session,
        )

        feats_len = feats.shape[1]
        pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
        p_len_tensor = torch.tensor([feats_len], device=self.device).long()

        with torch.inference_mode():
            audio_i16 = _to_int16(
                infer(
                    is_half=self._is_half,
                    session=self.session,
                    feats=feats,
                    pitch_length=p_len_tensor,
                    pitch=pitch,
                    pitchf=pitchf,
                    sid=self._sid,
                )[0]
            )

        out = audio_i16.detach().cpu().numpy()
        self._context = next_context(seq, self.context_len).detach()
        return slice_block_output(out, self.block_len, seq.shape[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run(GPU + config): `VSPEECH_RVC_GOLDEN_CONFIG=/path/to/config.toml uv run --all-extras pytest tests/test_stream_vc.py -v`
Expected: PASS(GPU 有り)/ SKIPPED(GPU 無し)。CPU の純粋ヘルパテストは常に PASS。
Run(型/lint): `uv run --all-extras poe type && uv run ruff check vspeech/lib/stream_vc.py`
Expected: 型エラー・lint エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/stream_vc.py tests/test_stream_vc.py
git commit -m "feat(stream-vc): add StreamingVc fixed-block core reusing change_voice internals

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: ハーネス実行部(掃引ループ / 計測 / main / poe タスク)

**Files:**
- Modify: `scripts/stream_vc_rtf.py`(実行部を追加。Task 2 の純粋ヘルパはそのまま)
- Modify: `poe_tasks.toml`(`stream-vc-rtf` タスクを追加)
- Test: `tests/test_stream_vc_rtf.py`(エントリポイント GPU-gated スモークを追加)

**Interfaces:**
- Consumes: Task 2 の解析ヘルパ; Task 3 の `StreamingVc`; `vspeech.config.Config.read_config_from_file` / `F0ExtractorType`; `vspeech.lib.cuda_util.get_device`; `vspeech.lib.onnx_session.create_session`; `vspeech.lib.rvc.load_hubert_model` / `half_precision_available`。
- Produces:
  - `load_shared_runtime(config_path: Path, gpu_id_override: int | None) -> dict` — device / hubert_model / synth session / metadata(target_sample_rate, f0_enabled, emb_output_layer, use_final_proj)/ rvc_config を返す(f0_session は掃引で抽出器ごとに作る)。
  - `make_f0_session(rvc_config, f0: str, device) -> InferenceSession | None` — `"rmvpe"`/`"fcpe"` のモデルファイルから session を作る。ファイル未設定/不在なら `None`。`"none"` は `None`。
  - `next_block(signal, block_len, i) -> NDArray[np.float32]` — 信号から i 番目のブロックを巡回で取り出す。
  - `run_sweep(rt, signal, block_ms_list, context_ms_list, f0_list, iters, warmup_iters, margin) -> list[BlockResult]`。
  - `main() -> None` — argparse、ロード、掃引、表・推奨・go/no-go を print、`--json` で dump。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_rtf.py` に追記:
```python
import os
import subprocess
import sys
from pathlib import Path

import pytest


_CONFIG_ENV = "VSPEECH_RVC_GOLDEN_CONFIG"
_config_path = os.environ.get(_CONFIG_ENV)
_GOLDEN_CONFIG = Path(_config_path) if _config_path else None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _cuda_available() or _GOLDEN_CONFIG is None or not _GOLDEN_CONFIG.exists(),
    reason=f"CUDA / ${_CONFIG_ENV} config not available",
)
def test_harness_entrypoint_runs_one_iter():
    # エントリポイントを実際に起動する(「テストだけでなくエントリポイントを走らせる」)。
    repo_root = Path(__file__).resolve().parents[1]
    assert _GOLDEN_CONFIG is not None
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.stream_vc_rtf",
            "--config", str(_GOLDEN_CONFIG),
            "--block-ms", "40",
            "--context-ms", "200",
            "--f0", "rmvpe",
            "--iters", "1",
            "--warmup-iters", "1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RTF" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run(GPU + config): `VSPEECH_RVC_GOLDEN_CONFIG=/path/to/config.toml uv run --all-extras pytest tests/test_stream_vc_rtf.py::test_harness_entrypoint_runs_one_iter -v`
Expected: FAIL(`main` 未実装 → non-zero exit / `SystemExit`)。GPU 無しでは SKIPPED。

- [ ] **Step 3: Write minimal implementation**

`scripts/stream_vc_rtf.py` の末尾に追記(top-level import はそのまま numpy/dataclass のみ。重い import は関数内):
```python
def load_shared_runtime(config_path, gpu_id_override):
    """[rvc] からモデルを 1 回ロード(f0_session は掃引で抽出器ごとに作る)。"""
    import json
    from typing import Any

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


def make_f0_session(rvc_config, f0: str, device):
    """"rmvpe"/"fcpe" の f0 session を作る。ファイル未設定/不在なら None。"""
    from pathlib import Path

    from vspeech.lib.onnx_session import create_session

    if f0 == "none":
        return None
    file_map = {"rmvpe": rvc_config.rmvpe_model_file, "fcpe": rvc_config.fcpe_model_file}
    model_file = file_map.get(f0)
    if model_file is None or model_file == Path() or not model_file.expanduser().exists():
        return None
    return create_session(model_file, device)


def next_block(signal: NDArray[np.float32], block_len: int, i: int) -> NDArray[np.float32]:
    """信号から i 番目の block を巡回で取り出す(長さ block_len)。"""
    span = max(1, signal.shape[0] - block_len)
    start = (i * block_len) % span
    return signal[start : start + block_len]


def run_sweep(
    rt, signal, block_ms_list, context_ms_list, f0_list, iters, warmup_iters, margin
) -> list[BlockResult]:
    """掃引して各 (block, context, f0) の per-block 遅延を計測する。"""
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
        f0_session = make_f0_session(rt["rvc_config"], f0, device)
        if f0 != "none" and f0_session is None:
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
    return results


def main() -> None:
    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RVC streaming VC RTF harness (M1)")
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

    rt = load_shared_runtime(args.config, args.gpu_id)
    if args.wav is not None:
        signal = _load_wav_16k(args.wav)
    else:
        signal = make_voiced_signal(16000, args.seconds, seed=0)

    results = run_sweep(
        rt,
        signal,
        block_ms_list=parse_grid(args.block_ms),
        context_ms_list=parse_grid(args.context_ms),
        f0_list=[x.strip() for x in args.f0.split(",") if x.strip()],
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        margin=args.margin,
    )

    print()
    print(format_table(results))
    print()
    best = recommend(results)
    if best is None:
        print(f"go/no-go: NO-GO (no config with RTF_p95 < {args.margin})")
    else:
        print(
            f"go/no-go: GO. recommend block={best.block_ms:.0f}ms "
            f"ctx={best.context_ms:.0f}ms f0={best.f0} "
            f"(latency ~{best.latency_ms:.1f}ms, RTF {best.rtf_p95:.2f}). "
            "-> 最終の block/context/遅延予算は人が確定する。"
        )
    if args.json is not None:
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")


def _load_wav_16k(path) -> NDArray[np.float32]:
    """wav を 16kHz mono float32 [-1,1] にして返す。"""
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
```

`poe_tasks.toml` の `[tool.poe.tasks]` セクション(offline 工程の並び)に追記:
```toml
# RVC ストリーミング VC の RTF 実測ハーネス(M1)。--all-extras で sync 済みの
# プロジェクト環境で走らせる(rvc extra の torch/onnxruntime-gpu が要る)。
#   uv sync --all-extras
#   uv run poe stream-vc-rtf --config ./config.toml
# 掃引軸は --block-ms / --context-ms / --f0 / --iters / --margin で上書き可。
stream-vc-rtf = { cmd = "python scripts/stream_vc_rtf.py", help = "Measure per-block latency/RTF of streaming VC across block/context/f0 (M1 go/no-go)" }
```

- [ ] **Step 4: Run test to verify it passes**

Run(純粋ヘルパ、常時): `uv run pytest tests/test_stream_vc_rtf.py -v`
Expected: 純粋ヘルパテスト PASS、エントリポイントスモークは GPU 有り=PASS / 無し=SKIPPED。
Run(型/lint/format): `uv run --all-extras poe type && uv run ruff check scripts/stream_vc_rtf.py && uv run ruff format --check scripts/stream_vc_rtf.py poe_tasks.toml`
Expected: エラー無し。
Run(タスク登録確認): `uv run poe --help`(`stream-vc-rtf` が一覧に出る)
Run(実機、GPU + 実 config のあるマシンで手動): `uv run --all-extras poe stream-vc-rtf --config ./config.toml`
Expected: 掃引表 + go/no-go + 推奨 config が出力される。

- [ ] **Step 5: Commit**

```bash
git add scripts/stream_vc_rtf.py poe_tasks.toml tests/test_stream_vc_rtf.py
git commit -m "feat(stream-vc): wire RTF sweep harness + stream-vc-rtf poe task (M1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Wrap-up(実装後・別枠)

- 実機(対象 GPU = 4060 / 5060Ti)で `poe stream-vc-rtf` を回し、掃引表と推奨 config、go/no-go を **人が確認**して最終の block/context/遅延予算を決める。数値は `--json` で記録可(PII 無し)。
- **ADR-0053 は Proposed のまま**にする。M1 は RTF-cost の裏づけ(余剰推論コスト)を与えるが、クロスフェード連続性の正しさは M2 で検証するまで未確定。昇格は M2 以降。
- `superpowers:finishing-a-development-branch` で結線(`poe check` 緑・実機確認済みを確認してから)。
- メモリ `rvc-streaming-vc.md` を M1 実測結果(feasible な block/context、選定トランスポート方針)で更新。

---

## Self-Review

**Spec coverage(`2026-07-22-rvc-streaming-vc-split-machine-design.md` の M1 受入基準に対して):**
- 「固定ブロックをステートフル VC に通し per-block 遅延/context 込み RTF を計測できる」→ Task 3(StreamingVc)+ Task 4(run_sweep/summarize)。✓
- 「達成可能な block/context 長・遅延予算・トランスポート方式を確定し go/no-go を判定」→ Task 4(format_table / recommend / go_no_go)+ Wrap-up(人が確定)。✓
- 「対象 GPU での RTF/遅延が計測でき、達成可能な遅延予算を判定できる」(全体 spec の受入基準)→ Task 4。✓
- 非ゴール(発話系無改変 / change_voice 経路無改変 / config 共有しない)→ Global Constraints で `rvc.py`/`vc.py`/routing 編集禁止、`[stream_vc]` は M1 スコープ外。✓

**Placeholder scan:** TODO/TBD/「適切に」等なし。全ステップに実コード。✓

**Type consistency:** `StreamingVc` の `__init__` 引数(block_len/context_len/f0_session/emb_output_layer/use_final_proj…)は Task 3 定義と Task 4 の `run_sweep` 呼び出し・Task 3 スモークテストで一致。`BlockResult` のフィールドは Task 2 定義と `summarize`/`format_table`/`recommend` で一致。`next_context`/`slice_block_output` は Task 1 定義を Task 3 が使用、シグネチャ一致。`build_rvc_runtime` の戻り dict キー(`rvc_config`/`device`/`hubert_model`/`session`/`f0_session`/`target_sample_rate`/`f0_enabled`/`emb_output_layer`/`use_final_proj`)は実ファイルと一致(Task 3 スモークで使用)。✓
