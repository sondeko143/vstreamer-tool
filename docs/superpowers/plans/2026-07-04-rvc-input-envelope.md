# RVC 入力音声エンベロープ転写の再設計 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the RVC output volume-matching (`adjust_output_vol_to_input_voice`) with a normalized-shape RMS envelope transfer that follows the input voice's relative dynamics while preserving the RVC output's own level.

**Architecture:** A new pure numpy helper `apply_input_envelope` reduces the input PCM and the RVC output to per-frame RMS envelopes, mean-normalizes each (so only the *shape*, not the absolute level, transfers), linearly interpolates the ratio to sample resolution, and applies it as a clamped gain. `VcConfig` is cleanly replaced with the new knobs. `rvc_worker` calls the helper once after `change_voice`, dropping the old `audioop`/`sqrt`/stepped-chunk path.

**Tech Stack:** Python 3.11, numpy, Pydantic v2, pytest.

## Global Constraints

- Python **3.11 only** (`>=3.11,<3.12`). Do not lower the floor; `TaskGroup`/`except*` in use.
- **Pydantic v2 APIs only** — no `parse_obj`/`.dict()`/`root_validator`/`Field(env=)`/`json_encoders`.
- **Imports one-per-line**, auto-sorted (ruff `force-single-line = true`, `extend-select = ["I","UP"]`).
- Type checker is **ty** (`uv run ty check`); must pass clean.
- **numpy is only in the `rvc`/`whisper` extras**, not base. Run tests/checks with `uv run --all-extras` (base `uv run pytest` already errors on `test_rvc_helpers` collection — that is pre-existing and expected).
- **Do not** reintroduce `audioop` for the envelope path (deprecated, removed in 3.13). The one remaining `from audioop import mul` for `input_boost` stays — out of scope.
- Config is a **clean replacement** — no back-compat aliases for the removed fields (config files are gitignored/personal).
- Branch: `feat/rvc-input-envelope` (already created; the design doc commit is on it).
- Spec: [docs/superpowers/specs/2026-07-04-rvc-input-envelope-design.md](../specs/2026-07-04-rvc-input-envelope-design.md).

---

### Task 1: `apply_input_envelope` helper + tests

**Files:**
- Modify: `vspeech/worker/vc.py` (add imports + three module-level functions; do NOT touch `rvc_worker` yet)
- Test: `tests/test_vc_helpers.py` (append numpy-based tests)

**Interfaces:**
- Produces:
  - `apply_input_envelope(output_i16: NDArray[np.int16], input_pcm: bytes, input_sample_width: int, input_rate: int, window_ms: float, strength: float, min_gain: float, max_gain: float) -> NDArray[np.int16]`
  - `_framewise_rms(samples: NDArray[np.float32], n_frames: int) -> NDArray[np.float64]`
  - `_dtype_for_width(width: int) -> np.dtype[Any]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vc_helpers.py`:

```python
import numpy as np

from vspeech.worker.vc import apply_input_envelope


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def test_envelope_strength_zero_returns_output_unchanged():
    rng = np.random.default_rng(0)
    out = rng.integers(-3000, 3000, 8000, dtype=np.int16)
    inp = _pcm(rng.integers(-3000, 3000, 4000))
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 0.0, 0.1, 4.0)
    np.testing.assert_array_equal(res, out)


def test_silent_input_returns_output_unchanged():
    out = (np.tile([1, -1], 4000) * 8000).astype(np.int16)
    inp = _pcm(np.zeros(4000))
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    np.testing.assert_array_equal(res, out)


def test_output_length_is_preserved():
    inp = _pcm(np.tile([1, -1], 2000) * 5000)
    out = (np.tile([1, -1], 5000) * 3000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    assert res.shape[0] == out.shape[0]


def test_constant_input_preserves_output_level():
    # Flat-amplitude input and flat-amplitude output -> gain ~1 everywhere.
    inp = _pcm(np.tile([1, -1], 2000) * 5000)
    out = (np.tile([1, -1], 4000) * 3000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.1, 4.0)
    np.testing.assert_allclose(res.astype(np.float32), out.astype(np.float32), atol=1.0)


def test_ramp_input_gain_follows_input_dynamics():
    # Input amplitude grows over the clip; flat output -> louder toward the end.
    n = 8000
    env = np.linspace(0.05, 1.0, n)
    carrier = np.tile([1.0, -1.0], n // 2)
    inp = _pcm(env * carrier * 20000)
    out = (np.tile([1, -1], n // 2) * 8000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.01, 10.0)
    first = np.abs(res[:1000].astype(np.float32)).mean()
    last = np.abs(res[-1000:].astype(np.float32)).mean()
    assert last > first * 3


def test_gain_transition_is_smooth_not_stepped():
    # Quiet first half, loud second half: the boundary must ramp, not jump.
    n = 8000
    amp = np.where(np.arange(n) < n // 2, 2000.0, 20000.0)
    carrier = np.tile([1.0, -1.0], n // 2)
    inp = _pcm(amp * carrier)
    out = (np.tile([1, -1], n // 2) * 8000).astype(np.int16)
    res = apply_input_envelope(out, inp, 2, 16000, 25.0, 1.0, 0.05, 10.0)
    gain = np.abs(res.astype(np.float32)) / (np.abs(out.astype(np.float32)) + 1e-9)
    assert np.abs(np.diff(gain)).max() < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --all-extras pytest tests/test_vc_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_input_envelope' from 'vspeech.worker.vc'`

- [ ] **Step 3: Add imports to `vspeech/worker/vc.py`**

Add these two imports (keep one-per-line, sorted with the existing block):

```python
import numpy as np
from numpy.typing import NDArray
```

- [ ] **Step 4: Add the three helper functions to `vspeech/worker/vc.py`**

Place them near the top-level helpers (e.g. after `check_cuda_provider`):

```python
def _dtype_for_width(width: int) -> np.dtype[Any]:
    """Signed-integer numpy dtype for a PCM byte-width.

    Matches the integer-PCM assumption of the old audioop path; vc input is
    INT16 in practice.
    """
    if width == 1:
        return np.dtype(np.int8)
    if width == 2:
        return np.dtype(np.int16)
    if width == 4:
        return np.dtype(np.int32)
    raise ValueError(f"unsupported input sample width: {width}")


def _framewise_rms(samples: NDArray[np.float32], n_frames: int) -> NDArray[np.float64]:
    """RMS of each of n_frames near-equal contiguous segments of samples."""
    bounds = np.linspace(0, samples.shape[0], n_frames + 1).astype(np.int64)
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = samples[bounds[i] : bounds[i + 1]]
        if seg.shape[0]:
            rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
    return rms


def apply_input_envelope(
    output_i16: NDArray[np.int16],
    input_pcm: bytes,
    input_sample_width: int,
    input_rate: int,
    window_ms: float,
    strength: float,
    min_gain: float,
    max_gain: float,
) -> NDArray[np.int16]:
    """Impose the input voice's *relative* loudness envelope onto RVC output.

    Both the input PCM and the RVC output are reduced to per-frame RMS
    envelopes, each normalized by its own mean so only the *shape* (not
    absolute level) carries over. The ratio is linearly interpolated to
    sample resolution and applied as a clamped gain, preserving the RVC
    output's own overall level while following how the speaker's volume rose
    and fell. Returns the RVC output unchanged when disabled implicitly
    (strength <= 0), when either side is empty, or when the input is
    effectively silent.
    """
    out_len = int(output_i16.shape[0])
    if out_len == 0 or not input_pcm or strength <= 0.0:
        return output_i16

    in_i = np.frombuffer(input_pcm, dtype=_dtype_for_width(input_sample_width))
    if in_i.shape[0] == 0:
        return output_i16
    full_scale = float(1 << (input_sample_width * 8 - 1))
    in_f = (in_i.astype(np.float32) / full_scale).astype(np.float32)
    out_f = output_i16.astype(np.float32)

    frame_len = max(1, round(window_ms * input_rate / 1000.0))
    n_frames = max(1, in_f.shape[0] // frame_len)

    rms_in = _framewise_rms(in_f, n_frames)
    rms_out = _framewise_rms((out_f / 32768.0).astype(np.float32), n_frames)

    # Frame centers on a shared normalized [0, 1] time axis, then stretch both
    # envelopes onto the output sample grid (linear interp = smooth gain).
    src_x = (np.arange(n_frames) + 0.5) / n_frames
    dst_x = (np.arange(out_len) + 0.5) / out_len
    env_in = np.interp(dst_x, src_x, rms_in)
    env_out = np.interp(dst_x, src_x, rms_out)

    eps = 1e-8
    mean_in = float(env_in.mean())
    if mean_in < eps:
        return output_i16
    mean_out = float(env_out.mean())
    shape_in = env_in / mean_in
    shape_out = env_out / mean_out if mean_out > eps else np.ones_like(env_out)

    gain = np.power(shape_in / (shape_out + eps), strength)
    gain = np.clip(gain, min_gain, max_gain)

    return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --all-extras pytest tests/test_vc_helpers.py -v`
Expected: PASS (all tests, including the pre-existing `check_cuda_provider` tests)

- [ ] **Step 6: Format, lint, type-check**

Run: `uv run ruff format vspeech/worker/vc.py tests/test_vc_helpers.py && uv run ruff check vspeech/worker/vc.py tests/test_vc_helpers.py && uv run --all-extras ty check vspeech/worker/vc.py`
Expected: no errors. (If ty flags the `_dtype_for_width` return, keep `np.dtype[Any]`; `Any` is already imported in vc.py.)

- [ ] **Step 7: Commit**

```bash
git add vspeech/worker/vc.py tests/test_vc_helpers.py
git commit -m "feat(vc): add normalized-shape input-envelope helper

apply_input_envelope reduces input PCM and RVC output to mean-normalized
per-frame RMS envelopes, interpolates the ratio to sample resolution and
applies a clamped gain. Pure numpy, unit-tested; not yet wired in.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Replace `VcConfig` fields + config example

**Files:**
- Modify: `vspeech/config.py:234-239` (`VcConfig`)
- Modify: `config.toml.example:72-73` (`[vc]` section)
- Test: `tests/test_vc_helpers.py` (add one config-defaults test)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `VcConfig` with fields `enable`, `adjust_output_vol_to_input_voice`, `envelope_strength`, `min_gain`, `max_gain`, `volume_adjust_window_ms` (used by Task 3).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vc_helpers.py`:

```python
def test_vc_config_envelope_defaults():
    from vspeech.config import VcConfig

    cfg = VcConfig()
    assert cfg.adjust_output_vol_to_input_voice is True
    assert cfg.envelope_strength == 1.0
    assert cfg.min_gain == 0.1
    assert cfg.max_gain == 4.0
    assert cfg.volume_adjust_window_ms == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --all-extras pytest tests/test_vc_helpers.py::test_vc_config_envelope_defaults -v`
Expected: FAIL — `AttributeError: 'VcConfig' object has no attribute 'envelope_strength'`

- [ ] **Step 3: Replace the `VcConfig` body**

In `vspeech/config.py`, replace:

```python
class VcConfig(BaseModel):
    enable: bool = False
    adjust_output_vol_to_input_voice: bool = True
    min_volume: float = 0.1
    max_volume: float = 1.0
    volume_adjust_window: int = 160
```

with:

```python
class VcConfig(BaseModel):
    enable: bool = False
    adjust_output_vol_to_input_voice: bool = True
    envelope_strength: float = 1.0
    min_gain: float = 0.1
    max_gain: float = 4.0
    volume_adjust_window_ms: float = 25.0
```

- [ ] **Step 4: Update `config.toml.example`**

Replace the `[vc]` section (`enable = true`) with:

```toml
[vc]
enable = true
# 入力音声の音量変化（抑揚）を出力へ転写する。既定 true。
adjust_output_vol_to_input_voice = true
# 転写の深さ 0(無効)..1(フル)。
envelope_strength = 1.0
# 適用ゲインのクランプ。max_gain は 1 超で平均より大きい部分も持ち上げる。
min_gain = 0.1
max_gain = 4.0
# RMS エンベロープのフレーム長(ms)。小さいほど追従は速いが粗い。
volume_adjust_window_ms = 25.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --all-extras pytest tests/test_vc_helpers.py::test_vc_config_envelope_defaults -v`
Expected: PASS

- [ ] **Step 6: Format, lint, type-check**

Run: `uv run ruff format vspeech/config.py && uv run ruff check vspeech/config.py && uv run --all-extras ty check vspeech/config.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add vspeech/config.py config.toml.example tests/test_vc_helpers.py
git commit -m "feat(config): replace vc volume knobs with envelope-transfer knobs

Drop min_volume/max_volume/volume_adjust_window; add envelope_strength,
min_gain, max_gain, volume_adjust_window_ms. Clean replacement (config is
personal/gitignored). Document the new [vc] knobs in config.toml.example.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire the helper into `rvc_worker` and remove the old path

**Files:**
- Modify: `vspeech/worker/vc.py` (imports cleanup + `rvc_worker` body `L110-186` region)

**Interfaces:**
- Consumes: `apply_input_envelope` (Task 1), the new `VcConfig` fields (Task 2).
- Produces: `rvc_worker` output built via the new envelope path.

- [ ] **Step 1: Replace the per-utterance body in `rvc_worker`**

In `vspeech/worker/vc.py`, replace the block from `input_sample_width = get_sample_size(...)` through the `output_data`/`vc_end_time`/`record_vc_elapsed` lines (current [L114-176](../../../vspeech/worker/vc.py), the `input_vols` precompute, the `change_voice` call, and the stepped `audioop.mul` loop) with:

```python
            input_sample_width = get_sample_size(speech.sound.format)
            vc_start_time = time.perf_counter()
            audio = await to_thread(
                change_voice,
                voice_frames=mul(
                    speech.sound.data,
                    input_sample_width,
                    rvc_config.input_boost,
                ),
                voice_sample_rate=speech.sound.rate,
                rvc_config=rvc_config,
                half_available=half_available,
                target_sample_rate=target_sample_rate,
                device=device,
                emb_output_layer=metadata.get("embOutputLayer", 9),
                use_final_proj=metadata.get("useFinalProj", True),
                hubert_model=hubert_model,
                session=session,
                f0_enabled=f0_enabled,
                rmvpe_session=rmvpe_session,
            )
            if vc_config.adjust_output_vol_to_input_voice:
                audio = apply_input_envelope(
                    audio,
                    speech.sound.data,
                    input_sample_width,
                    input_rate=speech.sound.rate,
                    window_ms=vc_config.volume_adjust_window_ms,
                    strength=vc_config.envelope_strength,
                    min_gain=vc_config.min_gain,
                    max_gain=vc_config.max_gain,
                )
            output_data = audio.tobytes()
            vc_end_time = time.perf_counter()
            record_vc_elapsed(vc_end_time - vc_start_time, trace_id=speech.trace_id)
            worker_output = WorkerOutput.from_input(speech)
```

(The subsequent `worker_output.sound = SoundOutput(...)`, `worker_output.text = ...`, `yield worker_output` lines remain unchanged.)

- [ ] **Step 2: Remove now-dead imports and the `chunks` helper**

In `vspeech/worker/vc.py` delete these lines (no longer referenced):
- `import audioop` (the module; the `from audioop import mul` line stays for `input_boost`)
- `from collections.abc import Generator`
- `from math import floor`
- `from math import sqrt`
- the `chunks(...)` function definition

- [ ] **Step 3: Verify no stale references remain**

Run: `uv run ruff check vspeech/worker/vc.py`
Expected: PASS with no `F401` (unused import) or `F821` (undefined name). If ruff reports an unused import you missed, remove it; if it reports an undefined name, you removed one still in use — restore it.

Also confirm the old symbols are gone:
Run: `grep -nE "audioop\.|sqrt|floor|input_vols|chunks\(|min_volume|max_volume|volume_adjust_window\b" vspeech/worker/vc.py`
Expected: no output (empty).

- [ ] **Step 4: Run the full vc test file + event-chain regression**

Run: `uv run --all-extras pytest tests/test_vc_helpers.py tests/test_event_chains.py -v`
Expected: PASS (envelope helpers, config defaults, `check_cuda_provider`, and routing all green).

- [ ] **Step 5: Format, lint, type-check the whole touched surface**

Run: `uv run ruff format . && uv run ruff check . && uv run --all-extras ty check`
Expected: no new errors on `vspeech/worker/vc.py` or `vspeech/config.py`. (Pre-existing ty diagnostics elsewhere in the repo are out of scope — do not fix unrelated files.)

- [ ] **Step 6: Commit**

```bash
git add vspeech/worker/vc.py
git commit -m "feat(vc): apply input envelope via apply_input_envelope

Wire the normalized-shape envelope transfer into rvc_worker and drop the
old sqrt / absolute-gain / stepped-chunk audioop path (fixes flattened
dynamics, mic-gain-dependent level, and zipper artifacts). Remove the now
-dead audioop/sqrt/floor/Generator imports and chunks helper.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 helper `apply_input_envelope` (+ `_framewise_rms`, `_dtype_for_width`) → Task 1. Signature incl. `input_rate` matches spec §3.1. ✓
- §3.2 edge cases (empty, silent, strength=0, length preserved) → Task 1 tests + guard clauses. ✓
- §3.3 config clean replacement (`envelope_strength`/`min_gain`/`max_gain`/`volume_adjust_window_ms`; drop old three) + `config.toml.example` → Task 2. ✓
- §3.4 worker integration (call after `change_voice`; remove old block; drop `audioop.rms`/`sqrt`/`chunks`/`floor`; keep `from audioop import mul`) → Task 3. ✓
- §4 tests (constant/ramp/smoothness/silent/strength-0/length) → Task 1. ✓
- §5 verification (pytest + ruff + ty; event-chains unaffected) → Tasks 1–3 verify steps. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows full code. ✓

**3. Type consistency:** `apply_input_envelope` signature identical in Task 1 (definition), spec, and Task 3 (call, keyword args `input_rate`/`window_ms`/`strength`/`min_gain`/`max_gain`). `VcConfig` field names identical across Task 2 definition, Task 2 test, and Task 3 call sites. `_framewise_rms`/`_dtype_for_width` names consistent. ✓
