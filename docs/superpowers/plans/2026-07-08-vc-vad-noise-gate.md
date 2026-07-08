# VC VAD ノイズゲート Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Silero VAD (ONNX) で vc worker にノイズゲートを追加する — 非音声チャンクは RVC 推論ごとスキップし、発話内の非音声区間 (ブレス・環境音) は RVC 出力をダックする。

**Architecture:** 新規モジュール `vspeech/lib/vad.py` に純粋なゲート計算 (numpy) と Silero VAD v5 ONNX セッション管理を置く。`vspeech/worker/vc.py` の `rvc_worker` が RVC 推論の **前に** VAD を実行し、speech 比率が閾値未満ならチャンクを破棄 (WorkerOutput を出さない)、それ以外は窓ごとのゲインマスクを RVC 出力に乗算する。RVC への入力は無加工 (HuBERT 特徴量の不連続を避ける)。opt-in (`vc.vad_gate = false` 既定)。

**Tech Stack:** numpy, onnxruntime (rvc extra に既存、CPUExecutionProvider 固定), Silero VAD v5 ONNX モデル (snakers4/silero-vad の `silero_vad.onnx`, MIT, 手動取得), pydantic v2 config。

**Spec:** `docs/superpowers/specs/2026-07-08-vc-vad-noise-gate-design.md`

## Global Constraints

- Python 3.11 のみ。import は 1 行 1 import (`ruff` `force-single-line = true`)。
- Pydantic v2 API のみ (`Field(default=...)`; v1 API 禁止)。
- `vspeech/lib/vad.py` は **モジュール import 時に onnxruntime / torch を要求しない** (base 環境での pytest collection を壊さない — `onnxruntime` は遅延 import、型注釈は `TYPE_CHECKING`)。過去に `tests/test_pitch_extract.py` が base 環境で collection error を起こした前例があるため厳守。
- 実モデル依存テストは env var `VSPEECH_VAD_MODEL` 指定時のみ実行 (`test_change_voice_golden.py` の `VSPEECH_RVC_GOLDEN_CONFIG` と同じ方式)。
- 既定値 (`vad_gate = false`) では従来と完全に同一の出力・挙動。
- コミットは Conventional Commits (`feat:` / `test:` / `docs:`)。committed ファイルに機械固有の絶対パスを書かない (gitleaks pre-commit が PII を検査する)。
- 検証コマンド: `uv run pytest <file> -v`、最終ゲートは `uv run poe fix` → `uv run poe check`。

## File Structure

- **Create** `vspeech/lib/vad.py` — VAD の全ロジック: 純粋関数 (`should_skip_vc`, `speech_gate_mask`, `apply_vad_gate`)、ONNX セッション (`create_vad_session`)、窓ごと推論 (`speech_probs`)、定数 (`VAD_SAMPLE_RATE` 等)。torch 非依存。
- **Create** `tests/test_vad_gate.py` — 上記の単体テスト (純粋関数 + スタブセッション + env-gated 実モデル) と `VcConfig` 既定値テスト。
- **Modify** `vspeech/config.py` — `VcConfig` に `vad_gate` ほか 6 フィールド追加。
- **Modify** `config.toml.example` — `[vc]` セクションに設定例とモデル取得コメント追記。
- **Modify** `vspeech/worker/vc.py` — `_input_as_float32_16k` ヘルパー追加、`rvc_worker` に VAD セッション構築 + スキップ/ダック組み込み。
- **Modify** `tests/test_vc_helpers.py` — `_input_as_float32_16k` のテスト追加。

---

### Task 1: 純粋なゲート計算 (`vspeech/lib/vad.py` 前半)

**Files:**

- Create: `vspeech/lib/vad.py`
- Test: `tests/test_vad_gate.py`

**Interfaces:**

- Produces (後続タスクが依存):
  - `VAD_SAMPLE_RATE: int = 16000`
  - `should_skip_vc(probs: NDArray[np.float64], threshold: float, min_speech_ratio: float) -> tuple[bool, float]` — `(スキップすべきか, speech比率)`
  - `speech_gate_mask(probs: NDArray[np.float64], threshold: float, pad_ms: float, min_gain: float) -> NDArray[np.float64]` — 窓ごとゲイン列 (speech=1.0, 非speech=min_gain)
  - `apply_vad_gate(output_i16: NDArray[np.int16], window_gains: NDArray[np.float64]) -> NDArray[np.int16]`

- [ ] **Step 1: Write the failing tests**

`tests/test_vad_gate.py` を新規作成:

```python
import numpy as np

from vspeech.lib.vad import apply_vad_gate
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_gate_mask


def test_should_skip_when_no_window_reaches_threshold():
    probs = np.array([0.1, 0.2, 0.05, 0.3])
    skip, ratio = should_skip_vc(probs, threshold=0.5, min_speech_ratio=0.1)
    assert skip
    assert ratio == 0.0


def test_should_not_skip_when_enough_speech_windows():
    probs = np.array([0.9, 0.8, 0.1, 0.1])
    skip, ratio = should_skip_vc(probs, threshold=0.5, min_speech_ratio=0.1)
    assert not skip
    assert ratio == 0.5


def test_should_skip_empty_probs():
    skip, ratio = should_skip_vc(
        np.zeros(0), threshold=0.5, min_speech_ratio=0.1
    )
    assert skip
    assert ratio == 0.0


def test_gate_mask_full_speech_is_all_ones():
    probs = np.full(10, 0.9)
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=0.0, min_gain=0.0)
    np.testing.assert_array_equal(mask, np.ones(10))


def test_gate_mask_nonspeech_gets_min_gain():
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.9])
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=0.0, min_gain=0.25)
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.25, 0.25, 1.0])


def test_gate_mask_pad_dilates_speech_regions():
    # 1 窓 = 32ms。pad_ms=32 -> speech の前後 1 窓ずつ開く。
    probs = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
    mask = speech_gate_mask(probs, threshold=0.5, pad_ms=32.0, min_gain=0.0)
    np.testing.assert_array_equal(mask, [0.0, 1.0, 1.0, 1.0, 0.0])


def test_apply_vad_gate_mutes_nonspeech_half():
    out = (np.ones(1000) * 10000).astype(np.int16)
    gains = np.array([1.0, 0.0])
    res = apply_vad_gate(out, gains)
    assert res.dtype == np.int16
    assert res.shape[0] == out.shape[0]
    assert res[0] == 10000
    assert res[-1] == 0


def test_apply_vad_gate_empty_gains_returns_output_unchanged():
    out = (np.ones(100) * 5000).astype(np.int16)
    res = apply_vad_gate(out, np.zeros(0))
    np.testing.assert_array_equal(res, out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vad_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.vad'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/vad.py` を新規作成:

```python
"""Silero VAD gate for the vc worker.

Pure-numpy gate math plus a thin onnxruntime wrapper. This module must stay
importable without onnxruntime/torch installed (base extras): onnxruntime is
imported lazily inside create_vad_session and only type-checked here.
"""

import numpy as np
from numpy.typing import NDArray

# Silero VAD v5 operates on 16kHz mono, 512-sample (32ms) windows, with a
# 64-sample context carried between windows and a (2, 1, 128) recurrent state.
VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_WINDOW_MS = VAD_WINDOW_SAMPLES * 1000.0 / VAD_SAMPLE_RATE


def should_skip_vc(
    probs: NDArray[np.float64], threshold: float, min_speech_ratio: float
) -> tuple[bool, float]:
    """Decide whether a chunk has too little speech to be worth RVC inference.

    Returns (skip, speech_ratio) where speech_ratio is the fraction of windows
    whose speech probability reaches threshold. An empty chunk is skipped.
    """
    if probs.shape[0] == 0:
        return True, 0.0
    ratio = float(np.mean(probs >= threshold))
    return ratio < min_speech_ratio, ratio


def speech_gate_mask(
    probs: NDArray[np.float64], threshold: float, pad_ms: float, min_gain: float
) -> NDArray[np.float64]:
    """Per-window gains: 1.0 on speech windows, min_gain elsewhere.

    The binary speech mask is dilated by pad_ms on both sides before gains are
    assigned, so consonant onsets and word tails just outside the VAD's speech
    region are not clipped.
    """
    speech = (probs >= threshold).astype(np.float64)
    pad_windows = round(pad_ms / _WINDOW_MS)
    if pad_windows > 0 and speech.shape[0]:
        kernel = np.ones(2 * pad_windows + 1)
        speech = (np.convolve(speech, kernel, mode="same") > 0).astype(np.float64)
    return np.where(speech > 0, 1.0, min_gain)


def apply_vad_gate(
    output_i16: NDArray[np.int16], window_gains: NDArray[np.float64]
) -> NDArray[np.int16]:
    """Multiply the RVC output by the window-resolution gain mask.

    Window centers are mapped onto the output sample grid through a normalized
    0..1 time axis (input and output cover the same duration at different
    sample rates) and linearly interpolated, so gain transitions ramp over a
    ~32ms window instead of stepping (no clicks).
    """
    out_len = int(output_i16.shape[0])
    n_windows = int(window_gains.shape[0])
    if out_len == 0 or n_windows == 0:
        return output_i16
    src_x = (np.arange(n_windows) + 0.5) / n_windows
    dst_x = (np.arange(out_len) + 0.5) / out_len
    gain = np.interp(dst_x, src_x, window_gains)
    out_f = output_i16.astype(np.float32) * gain
    return np.clip(out_f, -32768.0, 32767.0).astype(np.int16)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_gate.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/vad.py tests/test_vad_gate.py
git commit -m "feat(vc): pure VAD gate math (skip decision, gate mask, output duck)"
```

---

### Task 2: ONNX セッションと窓ごと推論 (`vspeech/lib/vad.py` 後半)

**Files:**

- Modify: `vspeech/lib/vad.py`
- Test: `tests/test_vad_gate.py` (追記)

**Interfaces:**

- Consumes: Task 1 の定数 (`VAD_WINDOW_SAMPLES`, `_CONTEXT_SAMPLES`)
- Produces (Task 4 が依存):
  - `create_vad_session(model_file: Path) -> "InferenceSession"` — 不在/v4 モデルは fail loudly
  - `speech_probs(session: Any, audio_16k: NDArray[np.float32]) -> NDArray[np.float64]` — 窓ごと speech 確率。`session` は onnxruntime `InferenceSession` (テストでスタブ可能にするため `Any`)

- [ ] **Step 1: Write the failing tests**

`tests/test_vad_gate.py` に追記:

```python
import os
from pathlib import Path

import pytest

from vspeech.lib.vad import speech_probs


class _StubSession:
    """Duck-typed InferenceSession: records feeds, returns a fixed prob and
    increments the state by 1 per call so state threading is observable."""

    def __init__(self, prob: float = 0.7):
        self.feeds: list[dict] = []
        self._prob = prob

    def run(self, output_names, input_feed):
        self.feeds.append({k: np.copy(v) for k, v in input_feed.items()})
        state = input_feed["state"] + 1.0
        return np.array([[self._prob]], dtype=np.float32), state


def test_speech_probs_window_count_and_feed_shapes():
    session = _StubSession()
    audio = np.zeros(512 * 3 + 100, dtype=np.float32)
    probs = speech_probs(session, audio)
    # 3 full windows + 1 zero-padded tail window
    assert probs.shape[0] == 4
    np.testing.assert_allclose(probs, 0.7)
    first = session.feeds[0]
    assert first["input"].shape == (1, 512 + 64)
    assert first["state"].shape == (2, 1, 128)
    assert first["sr"] == 16000


def test_speech_probs_carries_context_and_state():
    session = _StubSession()
    audio = np.arange(1024, dtype=np.float32)
    speech_probs(session, audio)
    second = session.feeds[1]
    # window 2 の context = window 1 の末尾 64 サンプル
    np.testing.assert_array_equal(
        second["input"][0, :64], np.arange(448, 512, dtype=np.float32)
    )
    # state はスタブが 1 ずつ加算 -> 2 呼び出し目にはゼロ +1 が届いている
    np.testing.assert_array_equal(second["state"], np.ones((2, 1, 128)))


def test_speech_probs_empty_audio_returns_empty():
    session = _StubSession()
    probs = speech_probs(session, np.zeros(0, dtype=np.float32))
    assert probs.shape[0] == 0
    assert not session.feeds


_VAD_MODEL_ENV = "VSPEECH_VAD_MODEL"
_vad_model = os.environ.get(_VAD_MODEL_ENV)
VAD_MODEL = Path(_vad_model) if _vad_model else None

requires_vad_model = pytest.mark.skipif(
    VAD_MODEL is None or not VAD_MODEL.exists(),
    reason=f"${_VAD_MODEL_ENV} not set or model missing",
)


@requires_vad_model
def test_real_model_silence_and_noise_score_low():
    pytest.importorskip("onnxruntime")
    from vspeech.lib.vad import create_vad_session

    assert VAD_MODEL is not None
    session = create_vad_session(VAD_MODEL)
    silence = np.zeros(16000, dtype=np.float32)
    assert float(speech_probs(session, silence).max()) < 0.3
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    assert float(np.mean(speech_probs(session, noise) >= 0.5)) < 0.1


def test_create_vad_session_missing_file_fails_loudly():
    pytest.importorskip("onnxruntime")
    from vspeech.lib.vad import create_vad_session

    with pytest.raises(FileNotFoundError) as excinfo:
        create_vad_session(Path("./no-such-vad-model.onnx"))
    assert "silero" in str(excinfo.value).lower()
```

(import 行はファイル先頭の既存 import 群に 1 行 1 import で統合すること。)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vad_gate.py -v`
Expected: 新規テストが FAIL — `ImportError: cannot import name 'speech_probs'`。実モデルテストは env var 未設定なら SKIP。

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/vad.py` に追記。ファイル先頭の import 群を次のとおり拡張:

```python
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
```

関数を追加:

```python
def create_vad_session(model_file: Path) -> "InferenceSession":
    """Build a CPU onnxruntime session for the Silero VAD v5 model.

    Fails loudly on a missing file or a non-v5 model: silently passing audio
    through would mean the noise the gate exists to stop comes back unnoticed.
    CPU is deliberate -- the model is ~2MB and must not contend with RVC for
    the GPU.
    """
    from onnxruntime import GraphOptimizationLevel
    from onnxruntime import InferenceSession
    from onnxruntime import SessionOptions

    path = model_file.expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Silero VAD model not found: {path}. Download silero_vad.onnx"
            " (v5) from the snakers4/silero-vad repository and set"
            " vc.vad_model_file."
        )
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in session.get_inputs()}
    if "state" not in input_names:
        raise ValueError(
            f"{path} does not look like a Silero VAD v5 model (inputs:"
            f" {sorted(input_names)}); v4 models (h/c inputs) are unsupported."
        )
    return session


def speech_probs(
    session: Any, audio_16k: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Per-window speech probabilities for a 16kHz float32 chunk.

    Replicates the silero-vad v5 wrapper: 512-sample windows, each prefixed
    with the previous window's last 64 samples (zeros for the first), with
    the recurrent state threaded through and reset per chunk. The tail
    window is zero-padded. `session` is an onnxruntime InferenceSession
    (typed Any so tests can substitute a stub).
    """
    n = int(audio_16k.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    n_windows = ceil(n / VAD_WINDOW_SAMPLES)
    padded = np.zeros(n_windows * VAD_WINDOW_SAMPLES, dtype=np.float32)
    padded[:n] = audio_16k
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
    sr = np.array(VAD_SAMPLE_RATE, dtype=np.int64)
    probs = np.zeros(n_windows, dtype=np.float64)
    for i in range(n_windows):
        window = padded[i * VAD_WINDOW_SAMPLES : (i + 1) * VAD_WINDOW_SAMPLES]
        feed = {
            "input": np.concatenate([context, window]).reshape(1, -1),
            "state": state,
            "sr": sr,
        }
        out, state = session.run(None, feed)
        probs[i] = float(out[0, 0])
        context = window[-_CONTEXT_SAMPLES:]
    return probs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_gate.py -v`
Expected: スタブ系 + fail-loudly 系 PASS。`test_real_model_*` は `VSPEECH_VAD_MODEL` 未設定なら SKIP (設定済みなら PASS)。

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/vad.py tests/test_vad_gate.py
git commit -m "feat(vc): Silero VAD v5 onnx session + windowed speech probs"
```

---

### Task 3: `VcConfig` 拡張と `config.toml.example`

**Files:**

- Modify: `vspeech/config.py` (`class VcConfig`, 現在 234 行付近)
- Modify: `config.toml.example` (`[vc]` セクション、72–83 行付近)
- Test: `tests/test_vad_gate.py` (追記)

**Interfaces:**

- Produces (Task 4 が依存): `VcConfig.vad_gate: bool`, `vad_model_file: Path`, `vad_threshold: float`, `vad_min_speech_ratio: float`, `vad_speech_pad_ms: float`, `vad_min_gain: float`

- [ ] **Step 1: Write the failing test**

`tests/test_vad_gate.py` に追記:

```python
def test_vc_config_vad_defaults_are_off_and_sane():
    from vspeech.config import VcConfig

    config = VcConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1
    assert config.vad_speech_pad_ms == 100.0
    assert config.vad_min_gain == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vad_gate.py::test_vc_config_vad_defaults_are_off_and_sane -v`
Expected: FAIL — `AttributeError: 'VcConfig' object has no attribute 'vad_gate'`

- [ ] **Step 3: Implement**

`vspeech/config.py` の `VcConfig` を次のとおり拡張 (既存フィールドは変更しない。`Path` は既に import 済み):

```python
class VcConfig(BaseModel):
    enable: bool = False
    adjust_output_vol_to_input_voice: bool = True
    envelope_strength: float = 1.0
    min_gain: float = 0.1
    max_gain: float = 1.0
    volume_adjust_window_ms: float = 25.0
    vad_gate: bool = False
    vad_model_file: Path = Field(default=Path())
    vad_threshold: float = 0.5
    vad_min_speech_ratio: float = 0.1
    vad_speech_pad_ms: float = 100.0
    vad_min_gain: float = 0.0
```

`config.toml.example` の `[vc]` セクション末尾 (`volume_adjust_window_ms = 25.0` の後) に追記:

```toml
# Silero VAD ノイズゲート (opt-in)。非音声チャンクは RVC 推論ごとスキップし、
# 発話内の非音声区間 (ブレス・環境音) は出力をダックする。
vad_gate = false
# snakers4/silero-vad リポジトリの silero_vad.onnx (v5) を取得してパスを指定する。
# vad_model_file = "./silero_vad.onnx"
# speech 確率の閾値。窓 (32ms) ごとにこれ以上で speech 扱い。
vad_threshold = 0.5
# speech 窓の割合がこれ未満のチャンクは RVC をスキップ (playback に何も流れない)。
vad_min_speech_ratio = 0.1
# speech 区間の前後をこの ms だけ開けておく (子音の頭・語尾の食われ防止)。
vad_speech_pad_ms = 100.0
# 非音声区間の出力ゲイン。0.0 = 完全ミュート。
vad_min_gain = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_gate.py -v`
Expected: 全 PASS (実モデルテストは SKIP 可)

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py config.toml.example tests/test_vad_gate.py
git commit -m "feat(vc): VcConfig vad gate knobs + config.toml.example docs"
```

---

### Task 4: vc worker への組み込み

**Files:**

- Modify: `vspeech/worker/vc.py`
- Test: `tests/test_vc_helpers.py` (追記)

**Interfaces:**

- Consumes: Task 1–2 の `VAD_SAMPLE_RATE` / `should_skip_vc` / `speech_gate_mask` / `apply_vad_gate` / `speech_probs` / `create_vad_session`、Task 3 の `VcConfig.vad_*`
- Produces: `_input_as_float32_16k(data: bytes, sample_width: int, rate: int) -> NDArray[np.float32]` (vc.py 内ヘルパー、テスト対象)

- [ ] **Step 1: Write the failing test**

`tests/test_vc_helpers.py` に追記 (ファイル先頭の import 群に `from vspeech.worker.vc import _input_as_float32_16k` を 1 行で追加):

```python
def test_input_as_float32_16k_scales_int16_without_resample():
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    res = _input_as_float32_16k(samples.tobytes(), 2, 16000)
    assert res.dtype == np.float32
    np.testing.assert_allclose(
        res, [0.0, 0.5, -0.5, 32767 / 32768], atol=1e-6
    )
```

(rate != 16000 の分岐は torch 依存のため単体テスト対象外 — base 環境の suite を green に保つ。)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vc_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name '_input_as_float32_16k'`

- [ ] **Step 3: Implement**

`vspeech/worker/vc.py` に 3 箇所の変更を入れる。

(a) モジュール先頭の import 群に追加 (1 行 1 import。`vspeech.lib.vad` はモジュール import 時に onnxruntime を要求しないので top-level で安全):

```python
from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import apply_vad_gate
from vspeech.lib.vad import create_vad_session
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_gate_mask
from vspeech.lib.vad import speech_probs
```

(b) `apply_input_envelope` の直後にヘルパーを追加:

```python
def _input_as_float32_16k(
    data: bytes, sample_width: int, rate: int
) -> NDArray[np.float32]:
    """Decode integer PCM to float32 in [-1, 1] at the VAD rate.

    torch/torchaudio are imported only on the resample path so this stays
    usable (and testable) in environments without the rvc extra when the
    input is already 16kHz.
    """
    scale = float(2 ** (8 * sample_width - 1))
    audio = (
        np.frombuffer(data, dtype=_dtype_for_width(sample_width)).astype(np.float32)
        / scale
    )
    if rate == VAD_SAMPLE_RATE:
        return audio
    import torch

    from vspeech.lib.rvc import get_resampler

    resampler = get_resampler(rate, VAD_SAMPLE_RATE, torch.device("cpu"))
    return resampler(torch.from_numpy(audio)).numpy()
```

(c) `rvc_worker` 内、`rmvpe_session` の分岐 (現在の 158–161 行) の直後に VAD セッション構築を追加:

```python
    if vc_config.vad_gate:
        vad_session = create_vad_session(vc_config.vad_model_file)
        logger.info("vad gate enabled: %s", vc_config.vad_model_file)
    else:
        vad_session = None
```

(d) ループ本体を変更。現在の

```python
        try:
            logger.debug("voice changing...")
            input_sample_width = get_sample_size(speech.sound.format)
            vc_start_time = time.perf_counter()
            audio = await to_thread(
```

を次のとおりに (VAD 判定は RVC 推論より **前**。VAD 例外時はゲートなしで通す):

```python
        try:
            logger.debug("voice changing...")
            input_sample_width = get_sample_size(speech.sound.format)
            vad_gains: NDArray[np.float64] | None = None
            if vad_session is not None:
                try:
                    audio_16k = _input_as_float32_16k(
                        speech.sound.data, input_sample_width, speech.sound.rate
                    )
                    probs = await to_thread(speech_probs, vad_session, audio_16k)
                    skip, speech_ratio = should_skip_vc(
                        probs,
                        vc_config.vad_threshold,
                        vc_config.vad_min_speech_ratio,
                    )
                    if skip:
                        telemetry.record(
                            "vc_skip", speech_ratio, trace_id=speech.trace_id
                        )
                        logger.info(
                            "vc skipped: speech ratio %.3f < %.3f",
                            speech_ratio,
                            vc_config.vad_min_speech_ratio,
                        )
                        continue
                    vad_gains = speech_gate_mask(
                        probs,
                        vc_config.vad_threshold,
                        vc_config.vad_speech_pad_ms,
                        vc_config.vad_min_gain,
                    )
                except Exception as e:
                    logger.warning("vad gate failed; passing chunk ungated: %s", e)
                    vad_gains = None
            vc_start_time = time.perf_counter()
            audio = await to_thread(
```

さらに、既存の envelope 適用ブロック

```python
            if vc_config.adjust_output_vol_to_input_voice:
                audio = apply_input_envelope(
                    ...
                )
```

の直後・`output_data = audio.tobytes()` の直前に追加:

```python
            if vad_gains is not None:
                audio = apply_vad_gate(audio, vad_gains)
```

注意:

- `continue` でチャンクを破棄すると WorkerOutput は emit されない (playback に何も流れない) — 仕様どおり。
- `telemetry.record` の signature は `record(stage: str, seconds: float, trace_id: str = "")` — `vc_skip` は秒ではなく speech 比率を記録する流用だが、jsonl/summary は名前つき float を汎用に扱うため問題ない。
- 既存の `mul(speech.sound.data, ..., rvc_config.input_boost)` は変更しない。VAD は boost 前の raw 入力で評価する (envelope と同じ)。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vc_helpers.py tests/test_vad_gate.py tests/test_vc_telemetry.py -v`
Expected: 全 PASS (既存テストも green のまま)

- [ ] **Step 5: Commit**

```bash
git add vspeech/worker/vc.py tests/test_vc_helpers.py
git commit -m "feat(vc): VAD gate in rvc_worker (skip noise-only chunks, duck non-speech)"
```

---

### Task 5: 全体ゲートと仕上げ

**Files:**

- Modify: なし (検証のみ; 修正が出れば該当ファイル)

- [ ] **Step 1: Auto-fix mechanical issues**

Run: `uv run poe fix`
Expected: ruff format + safe lint fixes。差分が出たら内容確認。

- [ ] **Step 2: Run every health gate**

Run: `uv run poe check`
Expected: fmt-check / lint / type (ty) / test (+coverage) / lock-check すべて PASS。
既知の pre-existing 課題 (このブランチ由来でない ty 診断等) は修正対象外 — 新規コードに起因するものだけ直す。

- [ ] **Step 3: Fix anything the gates surfaced in new code, re-run**

Run: `uv run poe check`
Expected: 新規コード起因の指摘ゼロ。

- [ ] **Step 4: Commit (fixes があった場合のみ)**

```bash
git add -A
git commit -m "style(vc): apply ruff format to vad gate changes"
```

---

## 手動検証 (実機、plan 外の受け入れ確認)

1. GPU ホストで `silero_vad.onnx` (v5) を取得し、vc を動かす config に `vad_gate = true` と `vad_model_file` を設定。
2. `VSPEECH_VAD_MODEL=<モデルパス> uv run pytest tests/test_vad_gate.py -v` で実モデルテストを PASS させる。
3. パイプラインを起動し: 無言でブレス・環境音のみ → playback に何も出ない + ログに `vc skipped` / telemetry に `vc_skip`。発話 → 従来どおり変換され、語間のブレスがミュートされる。子音の頭が欠ける場合は `vad_speech_pad_ms` を増やす (または `vad_threshold` を下げる)。
