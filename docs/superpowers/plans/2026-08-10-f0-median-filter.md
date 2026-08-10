# f0 voiced-run median filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全 f0 抽出器の出力に、無声 (0) を跨がない中央値フィルタを掛け、単発のオクターブ誤りが NSF へ届かないようにする。

**Architecture:** `vspeech/lib/pitch_extract.py` に純関数 `median_filter_f0` を足し、`pitch_extract()` の抽出器分岐直後で全経路に一様に適用する。強さは `RvcConfig.f0_filter_radius` (既定 1) が唯一の出所で、`pitch_extract()` には**必須引数**として渡す。harvest 専用だった `signal.medfilt` は削除し経路を 1 本にする。バッチ VC と streaming VC は共に `_select_pitch` を通るので配線は 1 箇所で済む。

**Tech Stack:** Python 3.14 / numpy 2 / pydantic v2 / pytest (`asyncio_mode = "auto"`) / uv + poe

**ADR:** [ADR-0070](../../adr/0070-f0-voiced-run-median-filter.md) (Proposed — 実装完了時に Accepted へ昇格させる)
**Spec:** [2026-08-10-f0-median-filter-design.md](../specs/2026-08-10-f0-median-filter-design.md)

## Global Constraints

- コメント / docstring は英語。config の `description=` と例外・ログ文言は日本語 (ADR-0064)。
- import は 1 行 1 つ (`ruff` `force-single-line = true`)。
- pydantic v2 API のみ。`Field(...)` に `ge` / `le` を付ける。
- 型検査は `uv run ty check` (プロジェクト全体。ファイル単位では test 側の型エラーを取り逃す)。
- 検証コマンドの終了コードはパイプを通さず直接確認する。pytest は完全な node ID で指定する。
- `tests/test_pitch_extract.py` に日本語コメントは無い (英語化済み)。そのまま英語で書く。

## File Structure

| File | 役割 |
| --- | --- |
| `vspeech/lib/pitch_extract.py` | `median_filter_f0` の追加、`pitch_extract()` への配線、harvest 専用 `medfilt` と `scipy` import の削除 |
| `vspeech/config.py` | `RvcConfig.f0_filter_radius` (既定値の唯一の出所) |
| `vspeech/lib/rvc.py` | `_select_pitch` が config の値を `pitch_extract()` へ渡す 1 行 |
| `config.toml.example` | `[rvc]` セクションへの記載 |
| `tests/test_pitch_extract.py` | フィルタの単体テストと `pitch_extract` 経由の統合テスト。既存 6 箇所の呼び出しへ必須引数を追加 |
| `tests/test_config_f0_filter.py` | 既定値・範囲・`[stream_vc.rvc]` への波及 |

---

### Task 1: `median_filter_f0` (純関数)

**Files:**
- Modify: `vspeech/lib/pitch_extract.py`
- Test: `tests/test_pitch_extract.py`

**Interfaces:**
- Consumes: なし
- Produces: `median_filter_f0(f0: NDArray[np.floating[Any]], radius: int) -> NDArray[np.floating[Any]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_pitch_extract.py` の import 群に追加:

```python
from scipy import signal

from vspeech.lib.pitch_extract import median_filter_f0
```

ファイル末尾に追加:

```python
def test_median_filter_f0_removes_an_isolated_octave_error_in_a_voiced_run():
    f0 = np.array([220.0, 221.0, 440.0, 219.0, 220.0])
    out = median_filter_f0(f0, 1)
    # median(221, 440, 219) == 221: the single-frame outlier is replaced by a neighbour.
    assert out[2] == pytest.approx(221.0)
    # Its neighbours are not dragged up by it either.
    assert out[1] == pytest.approx(221.0)
    assert out[3] == pytest.approx(220.0)


def test_median_filter_f0_leaves_unvoiced_frames_at_zero():
    f0 = np.array([220.0, 0.0, 0.0, 218.0, 219.0, 220.0])
    out = median_filter_f0(f0, 1)
    assert out[1] == 0.0
    assert out[2] == 0.0


def test_median_filter_f0_does_not_pull_voiced_run_edges_toward_unvoiced():
    # The property that separates this from a naive medfilt over the whole array: a
    # window spanning the 0 boundary would drag the run's first frame to min().
    f0 = np.array([0.0, 300.0, 200.0, 210.0, 0.0])
    out = median_filter_f0(f0, 1)
    naive = signal.medfilt(f0, 3)
    assert out[1] == pytest.approx(300.0)
    assert naive[1] == pytest.approx(200.0)


def test_median_filter_f0_last_frame_is_identity_so_no_lookahead_is_needed():
    # Edge replication makes the final frame a no-op, which is what lets the filter run
    # with zero added latency in the streaming path.
    f0 = np.array([200.0, 200.0, 400.0])
    assert median_filter_f0(f0, 1)[-1] == pytest.approx(400.0)
    assert median_filter_f0(np.array([200.0, 200.0, 200.0, 400.0]), 2)[-1] == (
        pytest.approx(400.0)
    )


def test_median_filter_f0_radius_zero_is_identity():
    f0 = np.array([220.0, 440.0, 219.0, 0.0])
    np.testing.assert_array_equal(median_filter_f0(f0, 0), f0)


def test_median_filter_f0_runs_shorter_than_the_kernel_are_identity():
    f0 = np.array([0.0, 300.0, 0.0, 400.0, 410.0, 0.0])
    np.testing.assert_allclose(median_filter_f0(f0, 2), f0)


def test_median_filter_f0_preserves_shape_and_dtype():
    f0 = np.array([220.0, 440.0, 219.0], dtype=np.float32)
    out = median_filter_f0(f0, 1)
    assert out.shape == f0.shape
    assert out.dtype == f0.dtype
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pitch_extract.py -q`
Expected: FAIL — `ImportError: cannot import name 'median_filter_f0'`

- [ ] **Step 3: Implement**

`vspeech/lib/pitch_extract.py`、`RMVPE_THRESHOLD` の下に追加:

```python
def median_filter_f0(
    f0: NDArray[np.floating[Any]], radius: int
) -> NDArray[np.floating[Any]]:
    """Median-filter f0 inside each voiced run (RVC's `filter_radius`, ADR-0070).

    Kills the isolated single-frame octave errors rmvpe/fcpe emit. Extraction is
    block-wise with no continuity constraint across blocks, so one bad frame otherwise
    reaches the NSF unopposed and rings as a short artefact.

    Unvoiced frames (0) never enter a window and are returned untouched. A window
    spanning the 0 boundary would drag the run's first voiced frame toward min(),
    blunting voiced onsets more audibly than the artefact being removed -- so each
    maximal run of f0 > 0 is filtered on its own, with run borders padded by edge
    replication rather than zeros.

    Edge replication also makes the array's final frame an identity (the replicated
    copies are a strict majority of the window), so this filter needs no lookahead and
    adds no latency. In the streaming path the emitted region ends about 3 frames before
    the array end, so every emitted frame still gets a genuine two-sided window at
    radius <= 3; beyond that the tail degrades to unfiltered rather than wrong.
    """
    if radius <= 0:
        return f0
    kernel = 2 * radius + 1
    out = f0.copy()
    # Run boundaries from the transitions of the voiced mask, bracketed by False so a run
    # touching either end is closed. np.diff on a bool array is XOR, so the flat indices
    # come out as alternating (start, stop) pairs.
    voiced = f0 > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([False], voiced, [False]))))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        padded = np.pad(f0[start:stop], radius, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        out[start:stop] = np.median(windows, axis=1)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pitch_extract.py -q`
Expected: PASS (既存 6 テストも緑のまま — どれも定数 f0 なので中央値は恒等)

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/pitch_extract.py tests/test_pitch_extract.py
git commit -m "feat(f0): add a voiced-run median filter (ADR-0070)"
```

---

### Task 2: `pitch_extract()` への配線と harvest 特別扱いの削除

**Files:**
- Modify: `vspeech/lib/pitch_extract.py`
- Test: `tests/test_pitch_extract.py`

**Interfaces:**
- Consumes: `median_filter_f0(f0, radius)` (Task 1)
- Produces: `pitch_extract(audio, f0_up_key, sr, window, f0_extractor, f0_filter_radius, f0_session, silence_front=0)` — `f0_filter_radius: int` は既定値なしの必須引数

- [ ] **Step 1: Write the failing tests**

`tests/test_pitch_extract.py` の末尾に追加:

```python
def test_pitch_extract_applies_the_median_filter_to_the_extractor_output():
    p_len = 100
    spiked = np.full((1, p_len), 220.0, dtype=np.float32)
    spiked[0, 50] = 440.0
    session = FakeRmvpeSession(spiked)

    _coarse, f0bak = pitch_extract(
        torch.zeros(16000, dtype=torch.float32),
        f0_up_key=0,
        sr=16000,
        window=160,
        f0_extractor=F0ExtractorType.rmvpe,
        f0_filter_radius=1,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )
    assert f0bak[50] == pytest.approx(220.0)


def test_pitch_extract_radius_zero_leaves_the_extractor_output_alone():
    p_len = 100
    spiked = np.full((1, p_len), 220.0, dtype=np.float32)
    spiked[0, 50] = 440.0
    session = FakeRmvpeSession(spiked)

    _coarse, f0bak = pitch_extract(
        torch.zeros(16000, dtype=torch.float32),
        f0_up_key=0,
        sr=16000,
        window=160,
        f0_extractor=F0ExtractorType.rmvpe,
        f0_filter_radius=0,
        f0_session=cast(InferenceSession, session),
        silence_front=0,
    )
    assert f0bak[50] == pytest.approx(440.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_pitch_extract.py::test_pitch_extract_applies_the_median_filter_to_the_extractor_output -q`
Expected: FAIL — `TypeError: pitch_extract() got an unexpected keyword argument 'f0_filter_radius'`

- [ ] **Step 3: Implement**

3a. `pitch_extract` のシグネチャで `f0_extractor` の直後に必須引数を挿入する (`silence_front` に既定値があるため、その後ろには置けない):

```python
    f0_extractor: F0ExtractorType,
    f0_filter_radius: int,
    f0_session: InferenceSession | None,
    silence_front: int = 0,
```

3b. 抽出器分岐の `raise ValueError("unknown f0 extractor type")` の直後、`f0 *= pow(...)` の直前に 1 行:

```python
    # Applied here rather than after the f0_up_key scaling only for readability: the
    # median commutes with a positive scalar multiple, so the result is identical.
    f0 = median_filter_f0(f0, f0_filter_radius)
```

3c. `pitch_extract_harvest` の `return signal.medfilt(f0, 3)` を `return f0` にし、docstring 相当のコメントを添える:

```python
    # No medfilt here any more: the shared voiced-run filter in pitch_extract covers
    # every extractor uniformly (ADR-0070). At the default radius 1 the kernel is the
    # same 3 this used to apply.
    return cast(
        NDArray[np.double],
        pyworld.stonemask(audio.astype(np.double), f0_, t, sr),
    )
```

(これで `pitch_extract_harvest` の本体は `pitch_extract_dio` と同じ形になる。)

3d. 未使用になった `from scipy import signal` をファイル冒頭から削除する。

3e. 既存 6 箇所の `pitch_extract(...)` 呼び出しに `f0_filter_radius=1,` を追加する (`f0_session=` の直前)。対象テスト:
`test_pitch_extract_rmvpe_routes_to_session_and_returns_aligned_pitch` /
`test_pitch_extract_rmvpe_requires_session` /
`test_pitch_extract_fcpe_routes_to_session_waveform_only` /
`test_pitch_extract_fcpe_single_frame_does_not_collapse_to_0d` /
`test_pitch_extract_fcpe_pads_short_input_to_min_samples` /
`test_pitch_extract_fcpe_requires_session`

- [ ] **Step 4: Run the whole file**

Run: `uv run pytest tests/test_pitch_extract.py -q`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/pitch_extract.py tests/test_pitch_extract.py
git commit -m "feat(f0): filter every extractor through the shared median filter"
```

---

### Task 3: config フィールドと `_select_pitch` の配線

**Files:**
- Modify: `vspeech/config.py` (`RvcConfig`)
- Modify: `vspeech/lib/rvc.py` (`_select_pitch`)
- Modify: `config.toml.example` (`[rvc]`)
- Test: `tests/test_config_f0_filter.py` (新規)

**Interfaces:**
- Consumes: `pitch_extract(..., f0_filter_radius=...)` (Task 2)
- Produces: `RvcConfig.f0_filter_radius: int` (既定 1、`0..7`)

- [ ] **Step 1: Write the failing test**

新規 `tests/test_config_f0_filter.py`:

```python
import pytest
from pydantic import ValidationError

from vspeech.config import Config
from vspeech.config import RvcConfig


def test_f0_filter_radius_defaults_to_one():
    assert RvcConfig().f0_filter_radius == 1


def test_f0_filter_radius_rejects_out_of_range():
    with pytest.raises(ValidationError):
        RvcConfig(f0_filter_radius=-1)
    with pytest.raises(ValidationError):
        RvcConfig(f0_filter_radius=8)


def test_stream_vc_rvc_table_gets_the_same_default():
    # [stream_vc.rvc] is the same RvcConfig model, so the knob reaches the streaming
    # path without a second field. A realistic table (not an absent one) is used here
    # because that is the case the before-validator handles.
    config = Config.model_validate({"stream_vc": {"rvc": {"model_file": "x.onnx"}}})
    assert config.stream_vc.rvc.f0_filter_radius == 1


def test_stream_vc_rvc_table_honours_an_explicit_value():
    config = Config.model_validate(
        {"stream_vc": {"rvc": {"model_file": "x.onnx", "f0_filter_radius": 0}}}
    )
    assert config.stream_vc.rvc.f0_filter_radius == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config_f0_filter.py -q`
Expected: FAIL — `AttributeError` / `ValidationError: extra fields not permitted`

- [ ] **Step 3: Implement**

3a. `vspeech/config.py` の `RvcConfig`、`f0_extractor_type` の直後に:

```python
    f0_filter_radius: int = Field(
        default=1,
        ge=0,
        le=7,
        description="f0 の中央値フィルタ半径 (窓長 = 2r+1、0 で無効)。無声フレームを"
        "跨がず有声区間ごとに適用する。3 を超えると streaming の右文脈が足りず、"
        "出力へ渡る末尾フレームが実質未フィルタになる (ADR-0070)",
    )
```

3b. `vspeech/lib/rvc.py` の `_select_pitch` 内、`f0_extractor=` の次の行に:

```python
        f0_filter_radius=rvc_config.f0_filter_radius,
```

3c. `config.toml.example` の `[rvc]` セクション、`f0_extractor_type` の記載の近くに:

```toml
# f0 の中央値フィルタ半径 (窓長 = 2r+1)。0 で無効。単発のオクターブ誤り由来の
# 「キュッ」という異音を消す。無声フレームは跨がないので有声の立ち上がりは鈍らない。
# 3 を超えると streaming 経路の右文脈が足りなくなり、出力へ渡る末尾フレームが
# 実質未フィルタになる (ADR-0070)。
f0_filter_radius = 1
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config_f0_filter.py tests/test_pitch_extract.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py vspeech/lib/rvc.py config.toml.example tests/test_config_f0_filter.py
git commit -m "feat(config): expose f0_filter_radius on RvcConfig (ADR-0070)"
```

---

### Task 4: 健全性ゲートと ADR の昇格

**Files:**
- Modify: `docs/adr/0070-f0-voiced-run-median-filter.md` (Status 1 行)
- Modify: `docs/adr/README.md` (索引の Status 1 列)

- [ ] **Step 1: Run the full gate**

```bash
uv run poe check
```
終了コードをパイプを通さずそのまま確認する。既知で受容済みの指摘 (audit の torch / deadcode の vr2_config) 以外の新規指摘はゼロであること。

- [ ] **Step 2: Confirm the whole-project type check separately**

```bash
uv run ty check
```
(ファイル単位ではテスト側の型エラーを取り逃すため、必ずプロジェクト全体で。)

- [ ] **Step 3: Promote the ADR**

`docs/adr/0070-f0-voiced-run-median-filter.md` の `- Status: Proposed` を `- Status: Accepted` にし、`docs/adr/README.md` の 0070 行の Status 列も `Accepted` にする。

- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): promote ADR-0070 to Accepted"
```

---

### Task 5: 実機側の 2 手 (ユーザーへ引き渡し)

コード側では完了できない。以下をユーザーへ明示的に引き渡す。

- [ ] **Step 1: バッチ経路のゴールデン撮り直し**

既定 `f0_filter_radius = 1` は意図して f0 を変えるため、`tests/test_change_voice_golden.py` は必ず失敗する。GPU ホストで:

```bash
VSPEECH_RVC_GOLDEN_CONFIG=<path> uv run python -m scripts.capture_change_voice_golden
uv run pytest tests/test_change_voice_golden.py::test_change_voice_matches_seeded_golden
```
撮り直した `tests/assets/rvc_golden/change_voice_golden.npz` をこのブランチに commit する。

- [ ] **Step 2: 実機の耳確認**

`f0_filter_radius = 1` と `0` を切り替えて、単発オクターブ誤り由来の「キュッ」が既定で消えていること、および有声の立ち上がりが鈍っていないことを確認する。

## Self-Review

**Spec coverage:**

| 受入基準 | 実装するタスク |
| --- | --- |
| 単発の外れ値が周囲の値へ置き換わる | Task 1 / Task 2 |
| 無声フレームの値が変わらない | Task 1 |
| 有声区間の端が無声側へ引かれない | Task 1 |
| 強さを設定で指定でき無効化もできる。既定は有効 | Task 3 |
| 片道遅延が増えない | Task 1 (edge 複製＝先読み不要。末尾恒等をテストで固定) |
| 抽出器によらず同一経路、抽出器固有の平滑化が残っていない | Task 2 (3c/3d) |
| harvest の既定の強さが導入前と変わらない | Task 2 (3c、既定 radius 1 = 窓長 3) |
| 上げすぎたときの劣化が設定例と決定記録から読める | Task 3 (3a/3c) + ADR Consequences |
| バッチ経路のゴールデンが更新されている | Task 5 |
| 実機の試聴で異音が確認できなくなる | Task 5 |
| 設定例と決定記録が実装と一致 | Task 4 |

**Placeholder scan:** なし。全ステップに実コードあり。

**Type consistency:** `median_filter_f0(f0, radius)` は Task 1 で定義し Task 2 で同名同順に使用。`f0_filter_radius` の名前は config フィールド・関数引数・toml キーで一致。
