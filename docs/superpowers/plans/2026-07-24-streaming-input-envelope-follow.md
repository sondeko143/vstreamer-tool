# Streaming Input-Envelope-Follow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make streaming VC output follow the input voice's loudness envelope (softer attack/decay, closer to the batch path) via an opt-in per-block envelope follower normalized against a rolling-EMA of the input level.

**Architecture:** A new pure `StreamingEnvelope` component (mirrors `gate.py`'s `StreamingVadGate`) computes the input block's per-frame RMS, normalizes it against a rolling-EMA reference of the input level, and applies `clip(shape**strength, min_gain, max_gain)` as a duck gain onto the output block. Wired into `vc_loop` **before** the VAD gate (mirroring batch's `envelope → gate` order). Default off = bit-identical output.

**Tech Stack:** Python 3.14, numpy (method-local import), pydantic v2. No new third-party dependency, no GPU/model.

## Global Constraints

- Python 3.14 only; **uv** (`uv run --no-sync ...` — a live pipeline may hold the venv lock; no new deps).
- **Imports one-per-line** (ruff `force-single-line`); `ty check` clean project-wide; `ruff check` + `ruff format --check` clean.
- **Pydantic v2 only.**
- **`envelope_follow=false` (default) ⇒ output is bit-identical** to today (the runner does not construct/call the envelope). Same discipline as `input_boost=1.0` / `vad_gate=false`.
- **`worker/vc.py` and the streaming core `lib/stream_vc.py` stay UNCHANGED**, as do the 6 發話系 files (`worker/vc.py`, `worker/recording.py`, `worker/playback.py`, `lib/rvc.py`, `lib/command.py`, `shared_context.py`). The streaming envelope is new code + a runner edit; it may reuse batch's math conceptually but must NOT edit or import `worker/vc.py` (importing it would pull torch and break the intent). Reimplement the ~8-line frame-RMS locally.
- `StreamingEnvelope` (`envelope.py`) must be **pure/CPU-testable**: numpy imported method-local, no torch/sounddevice, no module-level heavy imports (mirrors `gate.py`).
- Envelope is applied **before** the VAD gate; it uses the **raw input block** (pre-`input_boost`; `input_boost` is a uniform gain and does not change the relative shape).
- Envelope state resets on the same points the VC context/gate reset (capture REOPEN sentinel, pause/resume).
- ADR of record: **0057** (rolling-EMA reference) — `Proposed`; promote to `Accepted` after the on-rig ear-check (a verification step here, not a code task).
- Do NOT stage/touch `.vscode/settings.json`. Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

**New:**
- `vspeech/stream_vc/envelope.py` — `StreamingEnvelope` (pure, CPU-testable). Rolling-EMA reference + duck gain.
- `tests/test_stream_vc_envelope.py`.

**Modified:**
- `vspeech/config.py` — `StreamVcConfig` gains `envelope_follow` + `envelope_*` fields.
- `vspeech/stream_vc/runner.py` — `vc_loop` constructs the envelope (if `envelope_follow`), captures the raw block, applies the envelope before the gate, and resets it at the REOPEN/pause points.
- `config.toml.example` — document the `[stream_vc]` envelope keys.

---

### Task 1: Config — envelope fields

**Files:**
- Modify: `vspeech/config.py` (`StreamVcConfig`, after the `vad_*` fields ~line 456-492)
- Test: `tests/test_config_stream_vc.py` (extend)

**Interfaces:**
- Produces: `StreamVcConfig.envelope_follow: bool = False`, `.envelope_strength: float = 1.0`, `.envelope_min_gain: float = 0.1`, `.envelope_max_gain: float = 1.0`, `.envelope_window_ms: float = 25.0`, `.envelope_ema_ms: float = 2000.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_stream_vc.py  (add)
from vspeech.config import Config


def test_stream_vc_envelope_defaults_off():
    sv = Config().stream_vc
    assert sv.envelope_follow is False
    assert sv.envelope_strength == 1.0
    assert sv.envelope_min_gain == 0.1
    assert sv.envelope_max_gain == 1.0
    assert sv.envelope_window_ms == 25.0
    assert sv.envelope_ema_ms == 2000.0


def test_stream_vc_envelope_parses():
    sv = Config.model_validate(
        {"stream_vc": {"envelope_follow": True, "envelope_strength": 1.5,
                       "envelope_min_gain": 0.2, "envelope_ema_ms": 1500.0}}
    ).stream_vc
    assert sv.envelope_follow is True
    assert (sv.envelope_strength, sv.envelope_min_gain, sv.envelope_ema_ms) == (
        1.5, 0.2, 1500.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_config_stream_vc.py -v`
Expected: FAIL (`AttributeError: envelope_follow`).

- [ ] **Step 3: Add the fields**

In `StreamVcConfig` (after the `vad_*` fields), add:

```python
    # 入力エンベロープ追従 (opt-in, ADR-0057)。出力音量を入力の相対ラウドネス包絡へ
    # duck 追従させ、アタック/ディケイをバッチ変換に近づける。既定 off でビット不変。
    # 参照は入力平均 RMS の rolling EMA (envelope_ema_ms)。VAD ゲートの前に適用。
    envelope_follow: bool = Field(
        default=False,
        description="出力音量を入力の相対ラウドネス包絡へ追従させる (アタック/"
        "ディケイを滑らかに)。off だと RVC 生出力のままで立ち上がりが急峻",
    )
    envelope_strength: float = Field(
        default=1.0, ge=0, description="包絡形状の指数。>1 で追従を強調、0 で無効相当"
    )
    envelope_min_gain: float = Field(
        default=0.1, ge=0.0, le=1.0, description="duck の下限ゲイン (静音部の残し量)"
    )
    envelope_max_gain: float = Field(
        default=1.0,
        ge=0.0,
        description="ゲイン上限。既定 1.0 = duck のみ (クリップしない)。>1 は "
        "loud 部を int16 域外へ持ち上げてハードクリップするのでヘッドルームがある時のみ",
    )
    envelope_window_ms: float = Field(
        default=25.0, gt=0, description="入力 RMS のフレーム窓 ms"
    )
    envelope_ema_ms: float = Field(
        default=2000.0,
        gt=0,
        description="参照レベル (入力平均 RMS の rolling EMA) の時定数 ms。"
        "短いと loud onset で参照が跳ねて過敏、長いとレベル変化に鈍い。実測で調整",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_config_stream_vc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py tests/test_config_stream_vc.py
git commit -m "feat(stream-vc): add envelope-follow config fields (ADR-0057)"
```

---

### Task 2: StreamingEnvelope (`envelope.py`)

**Files:**
- Create: `vspeech/stream_vc/envelope.py`
- Test: `tests/test_stream_vc_envelope.py`

**Interfaces:**
- Produces: `class StreamingEnvelope` with
  `__init__(strength: float, min_gain: float, max_gain: float, window_ms: float, ema_ms: float, block_ms: float)`,
  `apply(out_i16: NDArray[np.int16], in_block: NDArray[np.float32]) -> NDArray[np.int16]`,
  `reset() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_envelope.py
import numpy as np

from vspeech.stream_vc.envelope import StreamingEnvelope


def _env(**kw):
    base = dict(strength=1.0, min_gain=0.1, max_gain=1.0, window_ms=25.0,
               ema_ms=2000.0, block_ms=160.0)
    base.update(kw)
    return StreamingEnvelope(**base)


def _block(level, n=2560):  # 160ms @ 16k
    return np.full(n, level, dtype=np.float32)


def _out(n=7680):  # 160ms @ 48k, full-scale-ish
    return np.full(n, 10000, dtype=np.int16)


def test_first_block_is_near_unity():
    # cold start: ref := block mean, flat block -> shape 1 -> gain 1 -> unchanged.
    env = _env()
    out = _out()
    got = env.apply(out.copy(), _block(0.2))
    assert np.allclose(got, out, atol=1)


def test_quiet_block_after_loud_is_ducked():
    env = _env()
    for _ in range(20):  # establish ema at the loud level
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.03))  # a decay-tail block, 10x quieter
    assert got.max() < 10000 * 0.5  # ducked well below the loud level


def test_steady_level_stays_near_unity():
    env = _env()
    for _ in range(10):
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.3))
    assert np.allclose(got, _out(), atol=20)  # duck-only, steady -> ~unity


def test_within_block_attack_ramp_is_ducked_at_the_quiet_lead_in():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3))  # ref at speech level
    # a block that is quiet in its first half, loud in its second (an onset)
    onset = np.concatenate([_block(0.02, 1280), _block(0.3, 1280)])
    got = env.apply(_out(), onset)
    assert got[0] < got[-1]  # gain rises across the block = attack ramp recovered


def test_min_gain_clamps_the_duck():
    env = _env(min_gain=0.25)
    for _ in range(20):
        env.apply(_out(), _block(0.3))
    got = env.apply(_out(), _block(0.0001))  # near-silent block
    assert got.max() >= 10000 * 0.25 - 2  # not ducked below min_gain


def test_reset_clears_ema():
    env = _env()
    for _ in range(20):
        env.apply(_out(), _block(0.3))
    env.reset()
    got = env.apply(_out().copy(), _block(0.03))  # cold start again -> ~unity, not ducked
    assert np.allclose(got, _out(), atol=1)


def test_empty_and_silent_passthrough():
    env = _env()
    out = _out()
    assert np.array_equal(env.apply(out.copy(), np.zeros(0, dtype=np.float32)), out)
    zero_env = _env(strength=0.0)
    assert np.array_equal(zero_env.apply(out.copy(), _block(0.3)), out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_stream_vc_envelope.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `envelope.py`**

```python
"""ストリーミング VC の入力エンベロープ追従 (ADR-0057)。

入力ブロックの相対ラウドネス包絡を、入力平均 RMS の rolling EMA を参照に正規化し、
duck ゲイン (clip(shape^strength, min_gain, max_gain)) として出力ブロックへ掛ける。
バッチ apply_input_envelope (worker/vc.py) と同じダック思想だが、参照を「発話全体の
平均」→「rolling EMA」に置換したストリーミング版 (単一ブロックしか手に入らない)。

判定と適用だけの pure ロジックで、numpy はメソッド内 import (torch/sounddevice を
引かず CPU・モデル無しで単体テストできる。gate.py と同型)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# 入力ブロックのサンプルレート (capture.py CAPTURE_RATE と同じ 16k)。capture.py を
# import すると sounddevice を引くのでここでは定数で持つ (この module を CPU で単体
# テストできるようにするため)。
_INPUT_RATE = 16000


class StreamingEnvelope:
    """rolling-EMA 参照の入力エンベロープ追従 (duck, ADR-0057)。

    状態は参照レベル `_ema_level` (スカラ) のみ。`apply()` が出力ブロックへ現在の
    入力ブロックの相対ラウドネス包絡を掛け、参照 EMA を次ブロック用に更新する。
    """

    def __init__(
        self,
        strength: float,
        min_gain: float,
        max_gain: float,
        window_ms: float,
        ema_ms: float,
        block_ms: float,
    ) -> None:
        self.strength = strength
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.window_ms = window_ms
        # 時定数 ema_ms の per-block EMA 係数 alpha = 1 - exp(-block_ms/ema_ms)。
        self._alpha = 1.0 - math.exp(-block_ms / ema_ms) if ema_ms > 0 else 1.0
        self._ema_level: float | None = None  # 初回 apply で block mean から init

    def reset(self) -> None:
        """参照レベルを未初期化へ戻す (pause/resume・capture 再 open で runner が呼ぶ)。

        実時間が飛んだあと古い参照レベルが次ブロックを妙に duck しないよう、
        次の apply で改めて cold start (block mean で init) させる。
        """
        self._ema_level = None

    def apply(
        self, out_i16: NDArray[np.int16], in_block: NDArray[np.float32]
    ) -> NDArray[np.int16]:
        """出力ブロック out_i16 に、入力ブロック in_block (16k float32) の相対
        ラウドネス包絡を rolling EMA 参照で duck 適用する。

        参照は **過去の** EMA (履歴)。cold start / reset 直後は現ブロックの平均で
        初期化する (初回ブロックが不自然に duck されないため)。参照を更新してから
        返すので、次ブロックはこのブロックを織り込んだ EMA を使う。

        **既知の特性 (ADR-0057, 実機耳確認で調整):** 長い無音では参照 EMA が入力の
        ノイズ床へ寄る (envelope_ema_ms で減衰)。その直後の発話頭 (phrase onset) は
        低い参照に対して全フレームが loud 判定になり duck されにくい = このブロック
        単独では整形が弱い。連続発話中の語間 dip / decay tail は参照が発話レベルに
        あるので正しく整形される。phrase onset は VAD ゲートが受け持つ。ema_ms を
        長くすると参照が無音を跨いで発話レベルを保ち、onset 整形が効きやすくなる。
        """
        import numpy as np

        out_len = int(out_i16.shape[0])
        if out_len == 0 or in_block.shape[0] == 0 or self.strength <= 0.0:
            return out_i16
        # 入力の per-frame RMS (絶対スケールは参照正規化で相殺されるので無関係)。
        frame_len = max(1, round(self.window_ms * _INPUT_RATE / 1000.0))
        n_frames = max(1, in_block.shape[0] // frame_len)
        bounds = np.linspace(0, in_block.shape[0], n_frames + 1).astype(np.int64)
        frame_rms = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            seg = in_block[bounds[i] : bounds[i + 1]].astype(np.float64)
            if seg.shape[0]:
                frame_rms[i] = np.sqrt(np.mean(seg**2))
        block_mean = float(frame_rms.mean())
        if self._ema_level is None:
            self._ema_level = block_mean
        ref = self._ema_level
        self._ema_level = self._alpha * block_mean + (1.0 - self._alpha) * ref
        if ref < 1e-8:  # 実質デジタル無音 (init 直後の完全無音等) → 素通し
            return out_i16
        # 相対形状 (mean~1 ではなく参照相対) を出力サンプル格子へ線形補間。
        src_x = (np.arange(n_frames) + 0.5) / n_frames
        dst_x = (np.arange(out_len) + 0.5) / out_len
        shape = np.interp(dst_x, src_x, frame_rms / ref)
        gain = np.clip(np.power(shape, self.strength), self.min_gain, self.max_gain)
        out_f = out_i16.astype(np.float32)
        return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_stream_vc_envelope.py -v`
Expected: PASS (7 tests). Also confirm torch-free import: `uv run --no-sync python -c "import sys; import vspeech.stream_vc.envelope; assert 'torch' not in sys.modules"` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/envelope.py tests/test_stream_vc_envelope.py
git commit -m "feat(stream-vc): rolling-EMA input envelope follower (ADR-0057)"
```

---

### Task 3: Wire the envelope into `vc_loop`

**Files:**
- Modify: `vspeech/stream_vc/runner.py` (`vc_loop`, ~lines 197-343)
- Test: `tests/test_stream_vc_runner.py` (extend — a construction/gating test)

**Interfaces:**
- Consumes: `StreamingEnvelope` (Task 2), `StreamVcConfig.envelope_*` (Task 1).
- Produces: `def make_stream_envelope(sv_config: StreamVcConfig) -> StreamingEnvelope | None` — returns the envelope when `envelope_follow`, else `None` (the gating authority, unit-testable without running the loop).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_runner.py  (add)
from vspeech.config import StreamVcConfig
from vspeech.stream_vc.runner import make_stream_envelope


def test_make_stream_envelope_gated_by_flag():
    assert make_stream_envelope(StreamVcConfig()) is None  # default off
    env = make_stream_envelope(StreamVcConfig(envelope_follow=True))
    assert env is not None
    assert env.strength == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_stream_vc_runner.py::test_make_stream_envelope_gated_by_flag -v`
Expected: FAIL (`ImportError: cannot import name 'make_stream_envelope'`).

- [ ] **Step 3: Implement the helper and wire it in**

Add the factory to `runner.py` (top-level):

```python
def make_stream_envelope(sv_config: StreamVcConfig):
    """envelope_follow のとき StreamingEnvelope を作る (off なら None)。純関数。"""
    if not sv_config.envelope_follow:
        return None
    from vspeech.stream_vc.envelope import StreamingEnvelope

    return StreamingEnvelope(
        strength=sv_config.envelope_strength,
        min_gain=sv_config.envelope_min_gain,
        max_gain=sv_config.envelope_max_gain,
        window_ms=sv_config.envelope_window_ms,
        ema_ms=sv_config.envelope_ema_ms,
        block_ms=sv_config.block_ms,
    )
```

In `vc_loop`, construct it next to the gate (after the `gate = ...` block, before `seq = 0`):

```python
    envelope = make_stream_envelope(sv_config)
```

At BOTH reset points, reset the envelope alongside the gate. In the REOPEN-sentinel branch (`if block is CaptureSignal.REOPEN:`) and the pause/resume branch (`if not context.running.is_set():`), add after the existing `gate.reset()`:

```python
                if envelope is not None:
                    envelope.reset()
```
(In each branch, next to `if gate is not None: gate.reset()`.)

Capture the raw block before `apply_input_boost`, and apply the envelope after `process_block` but BEFORE the gate ramp. Change the main-path section so it reads:

```python
            # envelope/gate は input_boost 前の素の入力で判定/整形する (boost は一様
            # ゲインで相対形状を変えない)。raw を保持してから boost する。
            raw_block = block
            target_gain = 1.0
            if gate is not None:
                target_gain = await gate_target_gain(gate, vad_session, raw_block)
            block = apply_input_boost(raw_block, sv_config.rvc.input_boost)
            t0 = perf_counter()
            ...
            try:
                out_i16 = await to_thread(sv.process_block, block)
            except RuntimeError as e:
                ...
            consecutive_errors = 0
            telemetry.record("stream_vc", perf_counter() - t0)
            # 入力エンベロープ追従 (ADR-0057) → VAD ゲートの順 (バッチ同様)。envelope は
            # 安価な numpy 演算なので inline (to_thread 不要)。
            if envelope is not None:
                out_i16 = envelope.apply(out_i16, raw_block)
            if gate is not None:
                out_i16 = gate.ramp(out_i16, target_gain)
                if target_gain != 1.0:
                    telemetry.record("stream_vc_vad_gated", 1.0)
```

(Adjust to the current exact lines; the key moves are: `raw_block = block` before boost, gate judged on `raw_block`, `apply_input_boost(raw_block, ...)`, and `envelope.apply(out_i16, raw_block)` inserted between the `stream_vc` telemetry and the gate ramp.)

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest tests/test_stream_vc_runner.py -v` and `uv run --no-sync pytest -q -k stream_vc`
Expected: PASS (new gating test + no regression). `uv run --no-sync ty check` clean; `uv run --no-sync ruff check .` / `ruff format --check .` clean.

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/runner.py tests/test_stream_vc_runner.py
git commit -m "feat(stream-vc): apply envelope follower before the gate in vc_loop (ADR-0057)"
```

---

### Task 4: Document the config keys

**Files:**
- Modify: `config.toml.example` (the `[stream_vc]` section)

- [ ] **Step 1: Add commented docs**

In the `[stream_vc]` block (near the `vad_*` keys), add (all commented — opt-in, default off):

```toml
# 入力エンベロープ追従 (ADR-0057)。出力音量を入力の相対ラウドネス包絡へ duck させ、
# アタック/ディケイをバッチ変換に近づける。既定 off で出力はビット不変。VAD ゲートの
# 前に適用し、ゲートとは併用 (ゲート=無音ノイズ抑制 / エンベロープ=発話中の整形)。
# envelope_follow = false
# envelope_strength = 1.0        # >1 で追従を強調
# envelope_min_gain = 0.1        # duck の下限 (静音部の残し量)
# envelope_max_gain = 1.0        # 1.0=duck のみ。>1 は loud 部をクリップさせるので非推奨
# envelope_window_ms = 25.0      # 入力 RMS のフレーム窓
# envelope_ema_ms = 2000.0       # 参照レベルの時定数。長いほど無音を跨いで発話レベルを保つ
```

- [ ] **Step 2: Verify parse + commit**

Run: `uv run --no-sync python -c "from vspeech.config import Config; import tomllib; Config.model_validate(tomllib.load(open('config.toml.example','rb')))"` → exit 0.
```bash
git add config.toml.example
git commit -m "docs(stream-vc): document envelope-follow config keys (ADR-0057)"
```

---

### Task 5: Gate + on-rig ear-check (verification)

No new production code. Verify and, if the ear-check passes, promote ADR-0057.

- [ ] **Step 1: Full local gate**

Run: `uv run --no-sync poe check` and `uv run --no-sync pytest -q`.
Expected: ty/ruff clean; suite green; `poe check` exit 1 only on the pre-existing accepted findings (torch CVE, pyasn1 transitive CVEs, subtitle_tk/tts vulture) — no NEW envelope-introduced finding (envelope.py adds no bandit/vulture surface: no bind, no unused symbols once wired).

- [ ] **Step 2: Default-off bit-identity check**

With `envelope_follow` unset/false, confirm the streaming output path is unchanged: `make_stream_envelope` returns `None` and `vc_loop` never calls `apply`. (Asserted by the Task 3 gating test; note it in the report.)

- [ ] **Step 3: On-rig ear-check (user)**

On the producer machine, enable `envelope_follow=true` in the producer config and compare A/B (off vs on) and vs the batch path:
- Output amplitude follows the input's loudness (attack/decay softer than off).
- Continuous-speech dynamics (word-to-word dips, decay tails) are ducked; voiced body stays near full.
- Tune `envelope_strength` / `envelope_min_gain` / `envelope_ema_ms` to taste. If phrase-onset (post-silence) attacks still feel steep, raise `envelope_ema_ms` (holds the reference across silences) and/or lean on the VAD gate.

- [ ] **Step 4: Promote ADR-0057**

If the ear-check confirms softer, input-following attack/decay: flip `docs/adr/0057-*.md` Status `Proposed → Accepted` and the README index row; record the settled `envelope_*` values (a commit message or a comment at the config defaults). Then `finishing-a-development-branch`.

---

## Self-Review

**Spec coverage** (against `2026-07-24-streaming-input-envelope-follow-design.md`):
- 出力が入力の相対エンベロープに追従 (opt-in, 既定無効) → Task 2 + 3 (gated by `envelope_follow`). ✅
- アタック/ディケイが無効時より滑らか・バッチに近い → Task 2 (duck) + Task 5 ear-check. ✅
- マイクゲイン非依存 → rolling-EMA reference normalizes the shape (Task 2). ✅
- 無音/語間で下がり有声はフルスケール付近 → duck-only + rolling ref (Task 2). ✅
- 無効時ビット同一 → default off, runner does not construct/apply (Task 1/3, Task 5 Step 2). ✅
- 發話系不変 → no 發話系 file edited (Global Constraints; envelope is new + runner-only). ✅
- VAD ゲート併用 → applied before gate, both duck, compose (Task 3). ✅

**Placeholder scan:** none — every step has full code.

**Type consistency:** `StreamingEnvelope.__init__` / `apply` / `reset` signatures match across Task 2 (impl + tests), Task 3 (`make_stream_envelope` constructs with the same kwargs), and Task 1 (config field names `envelope_*` used verbatim in Task 3's factory). `apply(out_i16, in_block)` order consistent everywhere.
