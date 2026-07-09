# RVC HuBERT fairseq-free 化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RVC の content encoder を fairseq から `transformers.HubertModel` へ置換し、runtime を fairseq 非依存にする（Python 引き上げの唯一の障害を除去する）。

**Architecture:** 手元の `hubert_base.pt`（ContentVec）を、3.11 + fairseq のオフラインツールで一度だけ transformers 形式へ変換し、`final_proj` を別テンソルとして抽出する。runtime はその変換済み資産だけを読む。正しさは「fairseq 時代に捕獲した HuBERT 特徴量」との fp32 数値等価ゲートで機械的に担保する。

**Tech Stack:** Python 3.11 / uv / PyTorch / transformers / safetensors / onnxruntime-gpu / pytest / ruff / ty

**設計書:** [docs/superpowers/specs/2026-07-09-rvc-hubert-fairseq-free-design.md](../specs/2026-07-09-rvc-hubert-fairseq-free-design.md)

## Global Constraints

- **`requires-python` は `>=3.11,<3.12` のまま。**本計画では引き上げない（後続 spec ③）。
- **ONNX 化はしない。**後続 spec ②。
- runtime（`vspeech/` 配下）に `fairseq` の import を **1 件も残さない**。
- `rvc` extra: `fairseq` を撤去、`transformers>=4.44,<5` を追加、`numpy>=1.23,<2`。
- `whisper` extra: `numpy>=1.23,<2`。
- 新設 `convert` extra: `fairseq`, `torch`, `torchaudio`, `transformers>=4.44,<5`, `numpy>=1.18,<1.24`。**`rvc` とは別の使い捨て 3.11 環境で単独使用**（fairseq 0.12.2 が `numpy<1.24` を要求するため）。
- 等価ゲート（**fp32 で判定**）: `cosine >= 0.9999` **かつ** `max-abs-diff <= 1e-4`。対象は `(emb_output_layer=9, use_final_proj=True)` と `(12, False)` の両方。
- 音声回帰ゲート: 正規化相互相関 `>= 0.999` **かつ** 全体 SNR `>= 40.0 dB`。
  全体 SNR は `waveform_snr(ref, test) = 10*log10(Σref² / Σ(ref−test)²)`。**フレーム分割も中央値も
  使わない**。完全一致（`noise == 0`）なら `inf`、参照が全無音でテストにエネルギーがあれば `-inf`。
  **非有限入力（NaN / inf）は `ValueError`**。エネルギー和が float64 で overflow した場合も `ValueError`。
  - **なぜセグメンタル中央値版を廃したか**: 中央値は外れ値に頑健なので、**少数フレームの破損に
    原理的に鈍感**であり、目的（回帰検出）に寄与しない。一方でフレーム分割 + マスク + 中央値の
    組み合わせは「壊れた信号に `inf`（完璧）を返す」経路を 5 通り生んだ（tiny 除算 overflow /
    中央値の上方飽和 / 無音参照フレームの除外 / `NaN > 0` が False / 末尾端数の破棄）。
    局所破損は相関ゲートが捕らえる。単一の式にして誤りようを無くす。
- `feature_cosine` は、**両方のノルムが 0 のフレームのみ一致 (1.0) とみなす**。片方だけが 0 の
  フレームは `0.0`（不一致）。両方 0 のフレームを一律除外すると、`feature_cosine(非ゼロ, 全ゼロ)`
  が `1.0` を返し、主ゲートがゴミに満点を出す。
- しきい値の確定規則: 実測誤差の **10 倍**を上限として採用してよい。**理由（実測値）をコメントに残さずに緩めてはならない。**
- **しきい値定数の単一情報源は `scripts/hubert_metrics.py`**（`COSINE_MIN` / `MAX_ABS_MAX` / `CORR_MIN` / `SNR_MIN_DB`）。変換ツールもテストも import して使い、値を再定義しない。
- **`load_hubert_model` は `torch.compile` を使わない。** 旧 fairseq 実装では `torch.compile` は一度も
  効いていなかった（`OptimizedModule` は `forward`/`__call__` しか包まず、旧コードが呼ぶ
  `model.extract_features(...)` は元メソッドへ素通しされる。実証済み）。transformers 版は `__call__` を
  呼ぶため初めて本当にコンパイルが走り、Triton の無い Windows/CUDA で `TritonMissing` を投げて
  **vc worker を毎チャンク落とす**。削除は旧実装の*実効*挙動への復帰であり、性能損失は無い。
  既存テストは `torch.compile` を恒等に monkeypatch していたため、この欠陥を構造的に検出できなかった。
- `HubertConfig` の値は fairseq モデルの実属性 / `saved_cfg` から導出し、**ハードコードしない**。
- import は **1 行 1 import**（ruff `force-single-line = true`）。
- 型検査は `ty`（Python 3.11）。`uv run poe check` が最終ゲート。
- 大きい派生資産（変換済み encoder・golden）は gitignore し、テストは環境変数未設定なら skip する。

## 実装順序の制約（違反すると正解データを失う）

golden 捕獲は **fairseq が動く環境でしか行えない**。必ず次の順で進める:

1. Task 1–2（純関数、資産不要）
2. **Task 3 で golden を捕獲**（この時点で `vspeech/lib/rvc.py` はまだ fairseq 版）
3. Task 4 で runtime を置換（fairseq 撤去）
4. Task 5–6 で等価を検証

## 設計書からの意図的な逸脱（1 点）

設計書 3.3(b) は「change_voice 音声 golden を**新実装で再ベースライン**する」としていたが、本計画では **fairseq 時代の golden をそのまま残し、許容誤差で照合する**。

理由: HuBERT は `eval()` + `inference_mode` 下で RNG を一切消費しないため、`seed_all()` 後の RNG ストリームは HuBERT 実装の差し替えに影響されない（RNG を引くのは RVC synthesizer の `infer` だけ）。したがって旧 golden との差分は**特徴量の値の差だけ**に由来し、この照合は移行の等価性を end-to-end で直接証明する。再ベースラインすると「将来の退行」しか守れず、移行そのものを検証できない。

## File Structure

**Create**
- `scripts/hubert_metrics.py` — 等価判定メトリクス（純関数）。変換ツールとテストが同じ数式で判定するための単一情報源。
- `scripts/hubert_keymap.py` — fairseq → transformers の state_dict キー変換（純関数、strict）。
- `scripts/convert_hubert.py` — オフライン変換ツール（3.11 + fairseq、一度だけ実行）。
- `tests/test_hubert_metrics.py`
- `tests/test_hubert_keymap.py`
- `tests/test_hubert_runtime.py` — 合成資産による runtime 単体テスト（実モデル不要）。
- `tests/test_hubert_equivalence.py` — 実資産による等価ゲート（skip-gated）。
- `tests/test_no_fairseq_import.py` — 構造ゲート。

**Modify**
- `vspeech/lib/rvc.py` — import（7,11,12 行）、`extract_features`（74-94）、`load_hubert_model`（189-209）、`_extract_hubert_feats` / `change_voice` の型注釈。
- `pyproject.toml` — `rvc` / `whisper` / 新 `convert` extra。
- `vspeech/config.py:335` — `hubert_model_file` の `description`。
- `config.toml.example:133` — 値の意味。
- `tests/test_change_voice_golden.py` — bit-exact → 許容誤差。
- `.gitignore` — 変換済み資産・golden ディレクトリ。

**変更不要（確認済み）**
- `gui/gui.py:780` — `draw_tb(config_name="rvc.hubert_model_file")` は汎用パス入力を生成するだけで、フィールド名を変えないためコード変更は不要。
- `scripts/capture_change_voice_golden.py` の `build_rvc_runtime` / `run_change_voice` — `load_hubert_model` のシグネチャを保つため変更不要（docstring のみ Task 5 で更新）。

---

### Task 1: 等価判定メトリクス（純関数）

変換ツールと等価テストが**同じ数式**で合否を出すための共有モジュール。`scripts/` に置く理由は、`pyproject.toml` の `pythonpath = "."` により `tests/` から `from scripts.x import y` で読めるため（既存 `tests/test_change_voice_golden.py` が同じ流儀）。runtime パッケージ `vspeech/` を汚さない。`scripts/__init__.py` は PR #10 以降すでに存在する（空）ので新規作成は不要。

**Files:**
- Create: `scripts/hubert_metrics.py`
- Test: `tests/test_hubert_metrics.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `feature_cosine(a: NDArray, b: NDArray) -> float` — 最終軸を特徴次元とみなしたフレーム毎 cosine の平均（両方ノルム 0 のフレームのみ 1.0、片方だけ 0 なら 0.0）
  - `feature_max_abs_diff(a: NDArray, b: NDArray) -> float`
  - `waveform_correlation(a: NDArray, b: NDArray) -> float` — 零ラグ正規化相互相関
  - `waveform_snr(reference: NDArray, test: NDArray) -> float` — 全体 SNR(dB)。`10*log10(Σref²/Σnoise²)`。完全一致なら `inf`、参照が全無音でテストに信号があれば `-inf`、非有限入力・エネルギー overflow は `ValueError`
  - **しきい値定数（単一情報源）**: `COSINE_MIN = 0.9999`, `MAX_ABS_MAX = 1e-4`, `CORR_MIN = 0.999`, `SNR_MIN_DB = 40.0`
    変換ツールもテストも**必ずここから import する**。値を二重に書かない（片方だけ書き換わる事故を防ぐ）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_hubert_metrics.py`:

```python
"""scripts/hubert_metrics の純関数テスト（資産・GPU 不要）。"""

import warnings

import numpy as np
import pytest

from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr


def test_feature_cosine_identical_is_one():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert feature_cosine(a, a) == pytest.approx(1.0)


def test_feature_cosine_orthogonal_is_zero():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert feature_cosine(a, b) == pytest.approx(0.0, abs=1e-9)


def test_feature_cosine_opposite_is_minus_one():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    assert feature_cosine(a, -a) == pytest.approx(-1.0)


def test_feature_cosine_rejects_shape_mismatch():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        feature_cosine(a, b)


def test_feature_cosine_both_zero_frames_match():
    z = np.zeros((3, 4), dtype=np.float32)
    assert feature_cosine(z, z) == pytest.approx(1.0)


def test_feature_cosine_one_side_all_zero_is_not_a_match():
    """片方だけが全ゼロなら不一致。ゼロノルムのフレームを一律除外すると 1.0 を返してしまう。"""
    a = np.ones((3, 4), dtype=np.float32)
    z = np.zeros((3, 4), dtype=np.float32)
    assert feature_cosine(a, z) == pytest.approx(0.0)


def test_feature_max_abs_diff():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.5, 2.0]], dtype=np.float32)
    assert feature_max_abs_diff(a, b) == pytest.approx(0.5)


def test_waveform_correlation_identical_and_inverted():
    x = np.sin(np.linspace(0.0, 10.0, 512)).astype(np.float32)
    assert waveform_correlation(x, x) == pytest.approx(1.0)
    assert waveform_correlation(x, -x) == pytest.approx(-1.0)


def test_waveform_snr_identical_is_inf_and_emits_no_warning():
    """完全一致 = 無限 SNR。除算しないので numpy 警告も出ない。"""
    x = np.sin(np.linspace(0.0, 10.0, 4096))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            assert waveform_snr(x, x) == float("inf")


def test_waveform_snr_decreases_with_noise():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 10.0, 4096))
    small = x + 1e-4 * rng.standard_normal(x.size)
    large = x + 1e-2 * rng.standard_normal(x.size)
    assert waveform_snr(x, small) > waveform_snr(x, large)
    assert np.isfinite(waveform_snr(x, large))


def test_waveform_snr_all_silent_is_inf():
    z = np.zeros(2048, dtype=np.float32)
    assert waveform_snr(z, z) == float("inf")


def test_waveform_snr_silent_reference_with_corrupted_test_is_minus_inf():
    """参照が無音でもテスト側にエネルギーがあれば「完璧」と報告してはならない。"""
    ref = np.zeros(2048)
    test = np.zeros(2048)
    test[1024:] = 0.1
    assert waveform_snr(ref, test) == float("-inf")


def test_waveform_snr_catches_corruption_at_any_offset():
    """フレーム分割しないので、末尾端数だろうと破損は必ず検出される。

    旧セグメンタル版は長さが frame_len の倍数でないとき末尾を捨て、そこだけが
    壊れていると inf（完璧）を返していた。全体 SNR にはその穴が無い。
    """
    ref = np.sin(np.linspace(0.0, 40.0, 2548))
    test = ref.copy()
    test[2048:] += 5.0  # 旧実装なら捨てられていた末尾端数だけを壊す
    result = waveform_snr(ref, test)
    assert np.isfinite(result), f"tail corruption was hidden: {result}"
    assert result < 40.0  # ゲートを通してはならない


def test_waveform_snr_rejects_non_finite_input():
    """NaN / inf は破損。`inf`（完璧）と誤報告せず明示的に落とすこと。"""
    x = np.sin(np.linspace(0.0, 10.0, 2048))
    with pytest.raises(ValueError, match="finite"):
        waveform_snr(x, np.full_like(x, np.nan))
    with pytest.raises(ValueError, match="finite"):
        waveform_snr(np.full_like(x, np.inf), x)


def test_waveform_snr_rejects_energy_overflow():
    """エネルギー和が float64 で overflow したら inf を返さず落とすこと。"""
    x = np.full(16, 1e200)
    test = x.copy()
    test[0] = 0.0
    with pytest.raises(ValueError, match="overflow"):
        waveform_snr(x, test)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_hubert_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.hubert_metrics'`

- [ ] **Step 3: 実装する**

`scripts/hubert_metrics.py`:

```python
"""HuBERT 置換の等価判定に使うメトリクス（純関数）。

変換ツール（scripts/convert_hubert.py）と等価テスト（tests/test_hubert_equivalence.py、
tests/test_change_voice_golden.py）が同じ数式・同じしきい値で合否を出せるよう、判定
ロジックとしきい値をここに一本化する。値をコピーせず必ず import すること。

設計方針: いずれの指標も **壊れた入力に「満点」を返してはならない**（fail-closed）。
"""

import math

import numpy as np
from numpy.typing import NDArray

# --- 合格しきい値（単一情報源） ---
# HuBERT 特徴量の等価判定（fp32）。
COSINE_MIN = 0.9999
MAX_ABS_MAX = 1e-4
# change_voice 出力音声の回帰判定。
CORR_MIN = 0.999
SNR_MIN_DB = 40.0


def _as_2d(x: NDArray) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1, arr.shape[-1])


def feature_cosine(a: NDArray, b: NDArray) -> float:
    """最終軸を特徴次元とみなしたフレーム毎 cosine 類似度の平均。

    **両方**のノルムが 0 のフレームだけを一致 (1.0) とみなす。片方だけが 0 のフレームは
    不一致 (0.0)。ゼロノルムのフレームを一律除外すると `feature_cosine(非ゼロ, 全ゼロ)` が
    1.0 を返し、主ゲートがゴミに満点を出してしまう。
    """
    if np.asarray(a).shape != np.asarray(b).shape:
        raise ValueError(f"shape mismatch: {np.asarray(a).shape} vs {np.asarray(b).shape}")
    x = _as_2d(a)
    y = _as_2d(b)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    den = norm_x * norm_y
    cosine = np.zeros(den.shape, dtype=np.float64)
    usable = den > 0.0
    cosine[usable] = (x[usable] * y[usable]).sum(axis=-1) / den[usable]
    cosine[(norm_x == 0.0) & (norm_y == 0.0)] = 1.0
    if cosine.size == 0:
        return 1.0
    return float(cosine.mean())


def feature_max_abs_diff(a: NDArray, b: NDArray) -> float:
    """要素毎の絶対差の最大値。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    return float(np.abs(x - y).max())


def waveform_correlation(a: NDArray, b: NDArray) -> float:
    """零ラグの正規化相互相関（-1..1）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den == 0.0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float((x * y).sum() / den)


def waveform_snr(reference: NDArray, test: NDArray) -> float:
    """`reference` に対する `test` の全体 SNR(dB)。`10*log10(Σref² / Σnoise²)`。

    **フレーム分割も中央値も使わない。** セグメンタル中央値版を廃した理由: 中央値は外れ値に
    頑健なので少数フレームの破損に原理的に鈍感で、回帰検出という目的に寄与しない。一方で
    フレーム分割 + マスク + 中央値の組み合わせは「壊れた信号に inf（完璧）を返す」経路を
    5 通り生んだ（tiny 除算 overflow / 中央値の上方飽和 / 無音参照フレームの除外 /
    `NaN > 0` が False / 末尾端数の破棄）。局所破損は相関ゲートが捕らえる。

    戻り値:
      - 完全一致（noise == 0）→ `inf`（両方とも無音の場合も含む）
      - 参照が全無音（signal == 0）でテストにエネルギーがある → `-inf`（破損）
      - それ以外 → 有限の dB

    非有限入力（NaN / inf）は破損なので ValueError。エネルギー和が float64 で overflow した
    場合も ValueError（inf を「完璧」と誤報告させない）。
    商を取らず log 空間で引くため、極端なダイナミックレンジでも比が overflow しない。
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    tst = np.asarray(test, dtype=np.float64).ravel()
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(tst).all():
        raise ValueError("waveform_snr: inputs must be finite (got NaN or inf)")
    # errstate: overflow は下の isfinite で ValueError にするので警告は出させない。
    with np.errstate(over="ignore"):
        signal = float((ref**2).sum())
        noise = float(((ref - tst) ** 2).sum())
    if not math.isfinite(signal) or not math.isfinite(noise):
        raise ValueError("waveform_snr: energy overflowed float64; rescale the inputs")
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * (math.log10(signal) - math.log10(noise))
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_hubert_metrics.py -v`
Expected: PASS（15 tests）

警告が出ないことも確認する（`waveform_snr` は商を取らず overflow も errstate で抑えるので numpy 警告ゼロが正）:

Run: `uv run pytest tests/test_hubert_metrics.py -W error::RuntimeWarning -q`
Expected: PASS

- [ ] **Step 5: 整形して commit**

```bash
uv run ruff format scripts/hubert_metrics.py tests/test_hubert_metrics.py
uv run ruff check scripts/hubert_metrics.py tests/test_hubert_metrics.py
git add scripts/hubert_metrics.py tests/test_hubert_metrics.py
git commit -m "feat(rvc): add HuBERT equivalence metrics (cosine, max-abs, corr, segSNR)"
```

---

### Task 2: fairseq → transformers キー変換（純関数・strict）

encoder 重みの取り違えは音質を静かに壊す。**transformers 側の全パラメータに fairseq の供給元があること**を強制し、無ければ例外にする。これが「黙って誤変換された encoder」が runtime に届くのを防ぐ唯一の砦。

**Files:**
- Create: `scripts/hubert_keymap.py`
- Test: `tests/test_hubert_keymap.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `translate_key(fairseq_key: str) -> str | None` — transformers 名。対応が無い（意図的に捨てる）なら `None`
  - `build_key_map(hf_keys: Iterable[str], fairseq_keys: Iterable[str]) -> dict[str, str]` — `{transformers_key: fairseq_key}`。未充足の transformers キーがあれば `KeyError`
  - `DROPPED_PREFIXES: tuple[str, ...]` — 意図的に捨てる fairseq キーの接頭辞

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_hubert_keymap.py`:

```python
"""fairseq -> transformers の state_dict キー変換テスト（torch 不要）。"""

import pytest

from scripts.hubert_keymap import build_key_map
from scripts.hubert_keymap import translate_key


@pytest.mark.parametrize(
    ("fairseq_key", "expected"),
    [
        ("feature_extractor.conv_layers.0.0.weight", "feature_extractor.conv_layers.0.conv.weight"),
        ("feature_extractor.conv_layers.0.2.weight", "feature_extractor.conv_layers.0.layer_norm.weight"),
        ("feature_extractor.conv_layers.4.0.weight", "feature_extractor.conv_layers.4.conv.weight"),
        ("post_extract_proj.weight", "feature_projection.projection.weight"),
        ("post_extract_proj.bias", "feature_projection.projection.bias"),
        ("layer_norm.weight", "feature_projection.layer_norm.weight"),
        ("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias"),
        ("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.weight_g"),
        ("encoder.layers.3.self_attn.q_proj.weight", "encoder.layers.3.attention.q_proj.weight"),
        ("encoder.layers.3.self_attn_layer_norm.bias", "encoder.layers.3.layer_norm.bias"),
        ("encoder.layers.11.fc1.weight", "encoder.layers.11.feed_forward.intermediate_dense.weight"),
        ("encoder.layers.11.fc2.bias", "encoder.layers.11.feed_forward.output_dense.bias"),
        ("mask_emb", "masked_spec_embed"),
        # 素通し（両者で同名）
        ("encoder.layer_norm.weight", "encoder.layer_norm.weight"),
        ("encoder.layers.7.final_layer_norm.weight", "encoder.layers.7.final_layer_norm.weight"),
    ],
)
def test_translate_key(fairseq_key, expected):
    assert translate_key(fairseq_key) == expected


@pytest.mark.parametrize("dropped", ["final_proj.weight", "final_proj.bias", "label_embs_concat"])
def test_translate_key_drops_non_encoder_params(dropped):
    assert translate_key(dropped) is None


def test_build_key_map_matches_and_drops():
    fairseq_keys = [
        "post_extract_proj.weight",
        "encoder.layer_norm.weight",
        "final_proj.weight",  # drop
        "label_embs_concat",  # drop
    ]
    hf_keys = ["feature_projection.projection.weight", "encoder.layer_norm.weight"]
    assert build_key_map(hf_keys, fairseq_keys) == {
        "feature_projection.projection.weight": "post_extract_proj.weight",
        "encoder.layer_norm.weight": "encoder.layer_norm.weight",
    }


def test_build_key_map_resolves_weight_norm_parametrization_alias():
    """新しい torch では weight_norm が parametrizations.* として現れる。"""
    fairseq_keys = ["encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"]
    hf_keys = [
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ]
    assert build_key_map(hf_keys, fairseq_keys) == {
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1": "encoder.pos_conv.0.weight_v",
    }


def test_build_key_map_raises_when_a_transformers_param_is_unsourced():
    with pytest.raises(KeyError, match="encoder.layer_norm.weight"):
        build_key_map(["encoder.layer_norm.weight"], ["post_extract_proj.weight"])


def test_build_key_map_raises_when_two_fairseq_params_collide():
    """同じ transformers パラメータに 2 つ着地したら、黙って後勝ちで上書きしないこと。

    「供給元が無い」網はこれを捕まえられない: 間違った規則の出力が別の正当なキーと
    偶然一致すると、正しい供給元が静かに捨てられて重みが壊れる。
    """
    hf_keys = ["feature_projection.projection.weight"]
    fairseq_keys = [
        "post_extract_proj.weight",  # 規則 3 で変換されて着地
        "feature_projection.projection.weight",  # 素通しで同じ名前に着地
    ]
    with pytest.raises(KeyError, match="same transformers param"):
        build_key_map(hf_keys, fairseq_keys)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_hubert_keymap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.hubert_keymap'`

- [ ] **Step 3: 実装する**

`scripts/hubert_keymap.py`:

```python
"""fairseq HuBERT/ContentVec の state_dict キーを transformers HubertModel 名へ写す。

構造的に strict: `build_key_map` は transformers 側のパラメータに供給元が
1 つでも無ければ例外を投げる。黙って誤変換された encoder が runtime に届く経路を塞ぐ。
"""

import re
from collections.abc import Iterable

# fairseq -> transformers。先頭から順に最初にマッチした 1 本だけを適用する。
_RULES: tuple[tuple[str, str], ...] = (
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"^feature_extractor\.conv_layers\.(\d+)\.2\.", r"feature_extractor.conv_layers.\1.layer_norm."),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^encoder\.pos_conv\.0\.", "encoder.pos_conv_embed.conv."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense."),
    (r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense."),
    (r"^mask_emb$", "masked_spec_embed"),
)

# transformers HubertModel に対応物が無い fairseq パラメータ。
# final_proj は別テンソルとして抽出するのでここでは捨てる。
# label_embs_concat は HuBERT の事前学習用ラベル埋め込みで、推論では使わない。
DROPPED_PREFIXES: tuple[str, ...] = ("final_proj.", "label_embs_concat")

# torch>=2.1 の weight_norm は parametrizations.* として state_dict に現れる。
_WEIGHT_NORM_ALIASES: dict[str, str] = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def translate_key(fairseq_key: str) -> str | None:
    """fairseq のキー名を transformers のキー名へ。

    `None` は「意図的に捨てるキー（DROPPED_PREFIXES）」の意味であって、「規則に一致しなかった」
    の意味ではない。規則に一致しないキーはそのまま素通しする（両者で同名のキーがあるため）。
    """
    if fairseq_key.startswith(DROPPED_PREFIXES):
        return None
    for pattern, replacement in _RULES:
        translated, hits = re.subn(pattern, replacement, fairseq_key)
        if hits:
            return translated
    # encoder.layer_norm.* / encoder.layers.N.final_layer_norm.* は両者で同名。
    return fairseq_key


def build_key_map(hf_keys: Iterable[str], fairseq_keys: Iterable[str]) -> dict[str, str]:
    """{transformers_key: fairseq_key} を返す。

    fail-closed の網は二重に張る:
      - 未充足: transformers 側のパラメータに供給元が 1 つも無ければ KeyError。
      - 衝突: 2 つの fairseq パラメータが同じ transformers パラメータに着地したら KeyError。
        素の dict 代入だと後勝ちで黙って上書きされる。「供給元が無い」網はこれを捕まえられない
        （間違った規則の出力が別の正当なキーと偶然一致するケース）ので、専用に検出する。
    """
    hf_key_set = set(hf_keys)
    mapping: dict[str, str] = {}
    for fairseq_key in fairseq_keys:
        translated = translate_key(fairseq_key)
        if translated is None:
            continue
        if translated not in hf_key_set:
            translated = _WEIGHT_NORM_ALIASES.get(translated, translated)
        if translated in hf_key_set:
            if translated in mapping:
                raise KeyError(
                    f"two fairseq params map to the same transformers param "
                    f"{translated!r}: {mapping[translated]!r} and {fairseq_key!r}"
                )
            mapping[translated] = fairseq_key
    unsourced = sorted(hf_key_set - set(mapping))
    if unsourced:
        raise KeyError(f"no fairseq source for transformers params: {unsourced}")
    return mapping
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_hubert_keymap.py -v`
Expected: PASS（22 tests = parametrize 15 + drop 3 + 単独 4）

- [ ] **Step 5: 整形して commit**

```bash
uv run ruff format scripts/hubert_keymap.py tests/test_hubert_keymap.py
uv run ruff check scripts/hubert_keymap.py tests/test_hubert_keymap.py
git add scripts/hubert_keymap.py tests/test_hubert_keymap.py
git commit -m "feat(rvc): add strict fairseq->transformers HuBERT key mapping"
```

---

### Task 3: オフライン変換ツール + golden 捕獲

**このタスクは fairseq が動く状態で完了させること。** Task 4 で `vspeech/lib/rvc.py` から fairseq を撤去すると、`capture_change_voice_golden.py` は transformers 版を呼ぶようになり、fairseq 基準の golden は作れなくなる。

**Files:**
- Create: `scripts/convert_hubert.py`
- Modify: `pyproject.toml`（`convert` extra を新設）
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `scripts.hubert_metrics` の `feature_cosine` / `feature_max_abs_diff` / `COSINE_MIN` / `MAX_ABS_MAX`（Task 1）、`scripts.hubert_keymap.build_key_map`（Task 2）
- Produces:
  - 変換済み資産ディレクトリ: `config.json`, `model.safetensors`, `final_proj.safetensors`, `mapping.json`（`{"layer_offset": int, "num_hidden_layers": int}`）
  - golden: `<golden>/hubert_golden.npz`（キー `wav`(float32 [T]), `l9_proj`(float32 [T', 256]), `l12_raw`(float32 [T', 768])）

- [ ] **Step 1: `convert` extra を追加する**

`pyproject.toml` の `[project.optional-dependencies]` に追記（`vroid2` の直前に置く）:

```toml
# HuBERT 変換ツール専用（scripts/convert_hubert.py）。runtime には含めない。
# fairseq 0.12.2 は numpy>=1.24 で撤去された旧エイリアスを使うため <1.24 を保持する。
# uv は 1 環境に numpy を 1 版しか解決しないので、rvc とは別の使い捨て 3.11 環境で使うこと。
convert = [
    "fairseq ; sys_platform == 'win32'",
    "torch ; sys_platform == 'win32'",
    "torchaudio ; sys_platform == 'win32'",
    "transformers>=4.44,<5",
    "numpy>=1.18,<1.24",
]
```

`[tool.uv.sources]` の `fairseq` エントリは**そのまま残す**（再変換可能性を保つ）。

- [ ] **Step 2: lock が通ることを確認**

Run: `uv lock`
Expected: 成功。`uv lock --check` も pass。

- [ ] **Step 3: 変換ツールを実装する**

`scripts/convert_hubert.py`:

```python
"""fairseq ContentVec (hubert_base.pt) を transformers HubertModel 資産へ変換する。

Python 3.11 + fairseq で **一度だけ** 走らせるオフライン処理。runtime には含めない。

`python scripts/convert_hubert.py` ではなく **`python -m scripts.convert_hubert`** で起動すること。
前者は sys.path[0] が scripts/ になり `from scripts.hubert_metrics import ...` を解決できない。

    uv run --extra convert python -m scripts.convert_hubert \
        --input  <path>/hubert_base.pt \
        --output <path>/hubert_contentvec \
        --golden <path>/hubert_golden

出力:
  <output>/config.json, model.safetensors  transformers HubertModel の encoder
  <output>/final_proj.safetensors          fairseq の final_proj (768->256)
  <output>/mapping.json                    fairseq output_layer -> hidden_states の対応
  <golden>/hubert_golden.npz               fairseq 側の正解特徴量（fp32）

変換の正しさはこのスクリプト自身がアサートする。通らなければ資産を書かない。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import HubertConfig
from transformers import HubertModel

from scripts.hubert_keymap import build_key_map
from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

# しきい値は scripts/hubert_metrics.py が単一情報源。ここで再定義しないこと。
# 変換時点でこれを満たさなければ資産を書き出さない。

# 検証・golden 捕獲に使う代表音声（決定論的。RNG を使わない）。
GOLDEN_SAMPLE_RATE = 16000
GOLDEN_SECONDS = 1.0


def make_fixed_audio() -> np.ndarray:
    """220Hz + 440Hz の決定論的な mono float32 波形（[-1, 1]）。"""
    n = int(GOLDEN_SAMPLE_RATE * GOLDEN_SECONDS)
    t = np.arange(n, dtype=np.float64) / GOLDEN_SAMPLE_RATE
    wave = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    return wave.astype(np.float32)


def load_fairseq_model(checkpoint: Path):
    """fairseq 側の ContentVec を eval モードで読む（現行 rvc.py と同一手順）。

    saved_cfg も返す。HubertConfig の一部（活性化関数名）はモジュール属性からは
    綺麗に取れないため、チェックポイントが保持している設定値から読む。
    """
    import fairseq.data.dictionary
    from fairseq import checkpoint_utils

    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
    models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint.expanduser())], suffix=""
    )
    model = models[0]
    model.eval()
    return model, saved_cfg


def derive_hf_config(fs_model, saved_cfg) -> HubertConfig:
    """fairseq モデルの実属性 / 保存済み設定から HubertConfig を組む（ハードコードしない）。"""
    conv_dim: list[int] = []
    conv_kernel: list[int] = []
    conv_stride: list[int] = []
    for conv_layer in fs_model.feature_extractor.conv_layers:
        conv = conv_layer[0]
        conv_dim.append(int(conv.out_channels))
        conv_kernel.append(int(conv.kernel_size[0]))
        conv_stride.append(int(conv.stride[0]))

    first_layer = fs_model.feature_extractor.conv_layers[0]
    has_group_norm = any(isinstance(m, torch.nn.GroupNorm) for m in first_layer)
    conv_bias = first_layer[0].bias is not None

    pos_conv = fs_model.encoder.pos_conv[0]
    first_block = fs_model.encoder.layers[0]

    return HubertConfig(
        hidden_size=int(fs_model.encoder.embedding_dim),
        num_hidden_layers=len(fs_model.encoder.layers),
        num_attention_heads=int(first_block.self_attn.num_heads),
        intermediate_size=int(first_block.fc1.out_features),
        hidden_act=str(saved_cfg.model.activation_fn),
        conv_dim=tuple(conv_dim),
        conv_kernel=tuple(conv_kernel),
        conv_stride=tuple(conv_stride),
        conv_bias=conv_bias,
        feat_extract_norm="group" if has_group_norm else "layer",
        feat_extract_activation="gelu",
        do_stable_layer_norm=bool(getattr(fs_model.encoder, "layer_norm_first", False)),
        num_conv_pos_embeddings=int(pos_conv.kernel_size[0]),
        num_conv_pos_embedding_groups=int(pos_conv.groups),
        # 推論専用。dropout は全て 0 にして eval と一致させる。
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
    )


def convert_encoder(fs_model, hf_config: HubertConfig) -> HubertModel:
    """fairseq の重みを transformers HubertModel へ strict に流し込む。"""
    hf_model = HubertModel(hf_config)
    hf_model.eval()

    fairseq_sd = fs_model.state_dict()
    key_map = build_key_map(hf_model.state_dict().keys(), fairseq_sd.keys())
    hf_model.load_state_dict({hf: fairseq_sd[fs] for hf, fs in key_map.items()}, strict=True)
    return hf_model


def fairseq_features(fs_model, source: torch.Tensor, layer: int, use_final_proj: bool) -> np.ndarray:
    """現行 rvc.py の extract_features と同一の呼び出しで fp32 特徴量を得る。"""
    padding_mask = torch.zeros(source.shape, dtype=torch.bool)
    with torch.inference_mode():
        logits = fs_model.extract_features(
            source=source, padding_mask=padding_mask, output_layer=layer
        )
        feats = fs_model.final_proj(logits[0]) if use_final_proj else logits[0]
    return feats.squeeze(0).float().cpu().numpy()


def hf_hidden_states(hf_model: HubertModel, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        # fairseq の padding_mask (True=パディング) と transformers の attention_mask
        # (1=有効) は意味が反転している。パディング無しなので何も渡さない = 全有効。
        outputs = hf_model(input_values=source, output_hidden_states=True)
    return outputs.hidden_states


def resolve_layer_offset(fs_model, hf_model: HubertModel, source: torch.Tensor) -> int:
    """fairseq output_layer=N が transformers hidden_states[N + offset] に対応する offset。

    候補を総当たりして誤差最小を選び、複数の層で一致することを要求する（off-by-one 対策）。
    """
    hidden_states = hf_hidden_states(hf_model, source)
    resolved: set[int] = set()
    for layer in (9, 12):
        reference = fairseq_features(fs_model, source, layer, use_final_proj=False)
        best_offset = None
        best_diff = float("inf")
        for offset in (-1, 0, 1):
            index = layer + offset
            if not 0 <= index < len(hidden_states):
                continue
            candidate = hidden_states[index].squeeze(0).float().cpu().numpy()
            if candidate.shape != reference.shape:
                continue
            diff = feature_max_abs_diff(candidate, reference)
            if diff < best_diff:
                best_diff = diff
                best_offset = offset
        if best_offset is None or best_diff > MAX_ABS_MAX:
            raise SystemExit(
                f"layer {layer}: no transformers hidden_states index matches fairseq "
                f"(best max-abs-diff={best_diff:.3e} > {MAX_ABS_MAX:.1e}). 変換が壊れている。"
            )
        resolved.add(best_offset)
    if len(resolved) != 1:
        raise SystemExit(f"layer offset が層ごとに矛盾している: {sorted(resolved)}")
    return resolved.pop()


def verify(fs_model, hf_model: HubertModel, source: torch.Tensor, layer_offset: int) -> dict[str, np.ndarray]:
    """(9,True) と (12,False) で等価をアサートし、fairseq 側の golden を返す。"""
    final_proj = fs_model.final_proj
    hidden_states = hf_hidden_states(hf_model, source)
    golden: dict[str, np.ndarray] = {}

    for layer, use_final_proj, key in ((9, True, "l9_proj"), (12, False, "l12_raw")):
        reference = fairseq_features(fs_model, source, layer, use_final_proj)
        hidden = hidden_states[layer + layer_offset]
        with torch.inference_mode():
            candidate_t = final_proj(hidden) if use_final_proj else hidden
        candidate = candidate_t.squeeze(0).float().cpu().numpy()

        cosine = feature_cosine(candidate, reference)
        max_abs = feature_max_abs_diff(candidate, reference)
        print(f"{key}: cosine={cosine:.8f} max_abs={max_abs:.3e} shape={reference.shape}")
        if cosine < COSINE_MIN or max_abs > MAX_ABS_MAX:
            raise SystemExit(
                f"{key}: equivalence FAILED (cosine={cosine:.8f} < {COSINE_MIN} "
                f"or max_abs={max_abs:.3e} > {MAX_ABS_MAX:.1e})。資産は書き出さない。"
            )
        golden[key] = reference.astype(np.float32)
    return golden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="hubert_base.pt")
    parser.add_argument("--output", required=True, type=Path, help="変換済み資産ディレクトリ")
    parser.add_argument("--golden", required=True, type=Path, help="golden 出力ディレクトリ")
    args = parser.parse_args()

    fs_model, saved_cfg = load_fairseq_model(args.input)
    hf_config = derive_hf_config(fs_model, saved_cfg)
    hf_model = convert_encoder(fs_model, hf_config)

    wav = make_fixed_audio()
    source = torch.from_numpy(wav).unsqueeze(0)

    layer_offset = resolve_layer_offset(fs_model, hf_model, source)
    print(f"resolved layer_offset={layer_offset}")

    golden = verify(fs_model, hf_model, source, layer_offset)

    args.output.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.output)
    save_file(
        {
            "weight": fs_model.final_proj.weight.detach().cpu().contiguous(),
            "bias": fs_model.final_proj.bias.detach().cpu().contiguous(),
        },
        str(args.output / "final_proj.safetensors"),
    )
    with open(args.output / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {"layer_offset": layer_offset, "num_hidden_layers": hf_config.num_hidden_layers},
            f,
            indent=2,
        )

    args.golden.mkdir(parents=True, exist_ok=True)
    np.savez(args.golden / "hubert_golden.npz", wav=wav, **golden)

    print(f"wrote asset -> {args.output}")
    print(f"wrote golden -> {args.golden / 'hubert_golden.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 資産を gitignore する**

`.gitignore` の末尾に追記:

```gitignore
# HuBERT 変換済み資産と golden 捕獲（派生物・大容量）
hubert_contentvec/
hubert_golden/
tests/assets/hubert_golden/
```

- [ ] **Step 5: lint と型検査**

```bash
uv run ruff format scripts/convert_hubert.py
uv run ruff check scripts/convert_hubert.py
```
Expected: pass。`ty` は `convert` extra 未同期だと `fairseq`/`transformers` 未解決を報告しうる。その場合は `uv sync --extra convert` してから再実行する（環境不足による偽陽性を repo 側で抑制しない）。

- [ ] **Step 6: 【オペレータ手順】change_voice golden を fairseq のまま捕獲する**

> **エージェントへ: このステップは実物の `hubert_base.pt` / RVC モデル / GPU を要する。実行環境が無ければここで停止し、ユーザに実行を依頼すること。**

`tests/assets/rvc_golden/change_voice_golden.npz` が既に存在すればスキップしてよい（PR #14 で捕獲済み）。無ければ **今のうちに**（rvc.py がまだ fairseq 版のうちに）捕獲する:

```bash
uv run --extra rvc python scripts/capture_change_voice_golden.py --config <path>/your-rvc-config.toml
```
Expected: `seeded self-noise max=0 (expected 0 -> reproducible)` と `wrote .../change_voice_golden.npz`

- [ ] **Step 7: 【オペレータ手順】変換を実行して資産と golden を作る**

```bash
uv run --extra convert python -m scripts.convert_hubert \
    --input  <path>/hubert_base.pt \
    --output <path>/hubert_contentvec \
    --golden <path>/hubert_golden
```
Expected: 標準出力に `resolved layer_offset=<N>`、`l9_proj: cosine=0.999... max_abs=...`、`l12_raw: ...` が出て、`wrote asset` / `wrote golden` で終わる。**等価アサートに落ちた場合は資産が書かれない**ので、Task 4 に進んではならない。

実測した `cosine` / `max_abs` を控えておく。Task 5 のしきい値確定（実測の 10 倍まで）に使う。

- [ ] **Step 8: commit**

```bash
git add pyproject.toml uv.lock .gitignore scripts/convert_hubert.py
git commit -m "feat(rvc): add offline fairseq->transformers HuBERT converter + convert extra"
```

---

### Task 4: runtime 置換（fairseq 撤去）

`extract_features` の**戻り値契約（shape / dtype / device）を保つ**ため、後段の `functional.interpolate` と `infer()` の io_binding は無改修で済む。

**Files:**
- Modify: `vspeech/lib/rvc.py`
- Modify: `pyproject.toml`（`rvc` / `whisper` extra）
- Modify: `vspeech/config.py:335`
- Modify: `config.toml.example:133`
- Modify: `tests/test_change_voice_golden.py`
- Modify: `scripts/capture_change_voice_golden.py`（docstring のみ）
- Create: `tests/test_hubert_runtime.py`
- Create: `tests/test_no_fairseq_import.py`

**Interfaces:**
- Consumes: Task 3 が書き出した資産レイアウト（`config.json` / `model.safetensors` / `final_proj.safetensors` / `mapping.json`）
- Produces:
  - `HubertBundle`（dataclass: `model: HubertModel`, `final_proj: torch.nn.Linear | None`, `layer_offset: int`）
  - `load_hubert_model(file_name: Path, device: torch.device, is_half: bool) -> HubertBundle`（シグネチャ不変、`file_name` は**ディレクトリ**）
  - `extract_features(model: HubertBundle, feats: torch.Tensor, dev: torch.device, emb_output_layer: int = 9, use_final_proj: bool = True) -> torch.Tensor`（シグネチャ不変）

- [ ] **Step 1: 失敗するテストを書く（合成資産・実モデル不要）**

`tests/test_hubert_runtime.py`:

```python
"""transformers ベース HuBERT runtime の単体テスト。

実物の hubert_base.pt を使わず、ごく小さな HubertConfig でランダム初期化した
資産をその場で作って検証する。層選択・final_proj 適用・エラー経路を固定する。
"""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import HubertConfig
from transformers import HubertModel

HIDDEN = 32
PROJ_OUT = 8
NUM_LAYERS = 4


def _tiny_config() -> HubertConfig:
    return HubertConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=(HIDDEN, HIDDEN),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
    )


@pytest.fixture
def asset_dir(tmp_path):
    """Task 3 の変換ツールが書き出すのと同じレイアウトの合成資産。"""
    torch.manual_seed(0)
    model = HubertModel(_tiny_config())
    model.eval()
    model.save_pretrained(tmp_path)
    save_file(
        {
            "weight": torch.randn(PROJ_OUT, HIDDEN).contiguous(),
            "bias": torch.randn(PROJ_OUT).contiguous(),
        },
        str(tmp_path / "final_proj.safetensors"),
    )
    with open(tmp_path / "mapping.json", "w", encoding="utf-8") as f:
        json.dump({"layer_offset": 0, "num_hidden_layers": NUM_LAYERS}, f)
    return tmp_path


def _wav() -> torch.Tensor:
    t = np.arange(4000, dtype=np.float32) / 16000.0
    return torch.from_numpy(np.sin(2 * np.pi * 220.0 * t).astype(np.float32)).unsqueeze(0)


def test_load_hubert_model_returns_bundle(asset_dir):
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    assert bundle.final_proj is not None
    assert bundle.layer_offset == 0


def test_extract_features_applies_final_proj(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=True)
    assert out.shape[0] == 1
    assert out.shape[2] == PROJ_OUT


def test_extract_features_without_final_proj_returns_hidden(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=False)
    assert out.shape[2] == HIDDEN


def test_extract_features_selects_the_requested_layer(asset_dir):
    """layer_offset=0 なら hidden_states[N] がそのまま返ること。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    wav = _wav()
    with torch.inference_mode():
        expected = bundle.model(input_values=wav, output_hidden_states=True).hidden_states[2]
    out = extract_features(bundle, wav, torch.device("cpu"), emb_output_layer=2, use_final_proj=False)
    assert torch.allclose(out, expected, atol=1e-6)


def test_extract_features_raises_when_final_proj_missing(asset_dir):
    """useFinalProj=True を要求されたのに資産に final_proj が無い場合。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    (asset_dir / "final_proj.safetensors").unlink()
    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    assert bundle.final_proj is None
    with pytest.raises(RuntimeError, match="final_proj"):
        extract_features(bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=True)
```

`tests/test_no_fairseq_import.py`:

```python
"""runtime に fairseq を二度と入れないための構造ゲート。

fairseq は requires-python 引き上げの唯一の障害（上流は 0.12.2 で凍結、
リポジトリは 2026-03-20 に archived）。import が復活したら即座に落とす。
"""

import ast
from pathlib import Path

VSPEECH_DIR = Path(__file__).resolve().parents[1] / "vspeech"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_vspeech_never_imports_fairseq():
    offenders = []
    for py_file in sorted(VSPEECH_DIR.rglob("*.py")):
        for module in _imported_modules(py_file):
            if module == "fairseq" or module.startswith("fairseq."):
                offenders.append(f"{py_file.relative_to(VSPEECH_DIR.parent)}: {module}")
    assert not offenders, "fairseq import leaked back into the runtime:\n" + "\n".join(offenders)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_hubert_runtime.py tests/test_no_fairseq_import.py -v`
Expected: FAIL — `test_hubert_runtime` は `ImportError: cannot import name 'HubertBundle'` 相当、`test_no_fairseq_import` は `vspeech/lib/rvc.py: fairseq...` で 3 件の offender。

- [ ] **Step 3: `vspeech/lib/rvc.py` の import を差し替える**

1–17 行目を次に置換（1 行 1 import、`fairseq` 3 本を削除）:

```python
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import torch
import torchaudio.transforms as T
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from safetensors.torch import load_file
from torch.nn import functional
from transformers import HubertModel
```

- [ ] **Step 4: `HubertBundle` を追加する**

`HUBERT_SAMPLE_RATE = 16000` の直後に挿入:

```python
@dataclass
class HubertBundle:
    """変換済み ContentVec 資産の runtime 表現。

    `final_proj` は資産に含まれていれば読む。RVC モデルのメタデータ `useFinalProj`
    が真のときだけ使われるため、ロード時点では必須にできない（不在は正当な構成）。
    """

    model: HubertModel
    final_proj: torch.nn.Linear | None
    layer_offset: int
```

- [ ] **Step 5: `extract_features` を差し替える（現 74-94 行）**

```python
def extract_features(
    model: HubertBundle,
    feats: torch.Tensor,
    dev: torch.device,
    emb_output_layer: int = 9,
    use_final_proj: bool = True,
) -> torch.Tensor:
    with torch.inference_mode():
        # fairseq の padding_mask (True=パディング) と transformers の attention_mask
        # (1=有効) は意味が反転している。ここは常にパディング無しなので何も渡さない
        # (= 全フレーム有効) のが等価。
        outputs = model.model(input_values=feats.to(dev), output_hidden_states=True)
        hidden = outputs.hidden_states[emb_output_layer + model.layer_offset]
        if use_final_proj:
            if model.final_proj is None:
                raise RuntimeError(
                    "RVC モデルが useFinalProj=True を要求していますが、変換済み資産に "
                    "final_proj.safetensors がありません"
                )
            hidden = model.final_proj(hidden)
    return hidden
```

- [ ] **Step 6: `load_hubert_model` を差し替える（現 189-209 行）**

```python
def load_hubert_model(
    file_name: Path, device: torch.device, is_half: bool
) -> HubertBundle:
    """変換済み ContentVec 資産ディレクトリを読む（scripts/convert_hubert.py の出力）。"""
    asset_dir = file_name.expanduser()

    # torch.compile は使わない。旧 fairseq 実装では OptimizedModule が forward/__call__ しか
    # 包まないため `model.extract_features(...)` が素通しされ、コンパイルは一度も走っていなかった。
    # transformers 版は __call__ を呼ぶので本当にコンパイルされ、Triton の無い Windows/CUDA では
    # TritonMissing で落ちる。旧実装の実効挙動（eager）に合わせる。
    model = HubertModel.from_pretrained(asset_dir)
    model.eval()
    model = model.to(device)
    if is_half:
        model = model.half()

    with open(asset_dir / "mapping.json", encoding="utf-8") as f:
        layer_offset = int(json.load(f)["layer_offset"])

    final_proj: torch.nn.Linear | None = None
    final_proj_path = asset_dir / "final_proj.safetensors"
    if final_proj_path.exists():
        tensors = load_file(str(final_proj_path))
        weight = tensors["weight"]
        bias = tensors["bias"]
        final_proj = torch.nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            final_proj.weight.copy_(weight)
            final_proj.bias.copy_(bias)
        final_proj.eval()
        final_proj = final_proj.to(device)
        if is_half:
            final_proj = final_proj.half()

    return HubertBundle(
        model=cast(HubertModel, model),
        final_proj=final_proj,
        layer_offset=layer_offset,
    )
```

- [ ] **Step 7: 残りの型注釈を `HubertBundle` に直し、warmup コメントを実態に合わせる**

`_extract_hubert_feats` の `hubert_model: HubertModel` → `hubert_model: HubertBundle`（現 239 行）。
`change_voice` の `hubert_model: HubertModel` → `hubert_model: HubertBundle`（現 333 行）。

`vspeech/worker/vc.py` の warmup コメント（現 201-203 行）は `torch.compile` に言及しているが、
`load_hubert_model` から撤去したので実態と合わない。次に差し替える:

```python
    # Warm up: pay the onnxruntime graph-build / CUDA kernel autotune cost at
    # startup. The first real inference would otherwise stall for seconds
    # (observed up to ~145s) while these build lazily.
```

- [ ] **Step 8: テストが通ることを確認**

Run: `uv run pytest tests/test_hubert_runtime.py tests/test_no_fairseq_import.py -v`
Expected: PASS（6 tests）

- [ ] **Step 9: 依存を入れ替える**

`pyproject.toml` の `whisper` / `rvc` extra を次に置換:

```toml
whisper = [
    "faster-whisper==1.2.1",
    "torch ; sys_platform == 'win32'",
    "onnxruntime-gpu>=1.24.4,<2",
    "numpy>=1.23,<2",
]
rvc = [
    "torch ; sys_platform == 'win32'",
    "transformers>=4.44,<5",
    "pyworld>=0.3.3,<0.4",
    "faiss-cpu>=1.7.2,<2 ; sys_platform == 'win32'",
    "onnxruntime-gpu>=1.24.4,<2",
    "torchaudio ; sys_platform == 'win32'",
    "numpy>=1.23,<2",
    "scipy>=1.10.1,<2",
]
```

（`rvc` から `fairseq` の行が消え、`transformers` が入り、numpy 上限が上がる。`fairseq` は Task 3 で作った `convert` extra にのみ残る。）

- [ ] **Step 10: lock と同期を確認（非破壊で）**

```bash
uv lock
uv lock --check
uv sync --all-extras
uv export --no-default-groups --extra rvc --no-hashes 2>/dev/null | grep -ci fairseq
grep -n "fairseq" pyproject.toml
```
Expected:
- `uv lock --check` が pass。
- `uv export ... --extra rvc | grep -ci fairseq` が **`0`**（rvc 経路に fairseq が居ない）。
- `pyproject.toml` の `fairseq` 出現箇所が **`convert` extra と `[tool.uv.sources]` のみ**（`rvc` extra には無い）。

> **`uv sync --extra rvc` を使わないこと。** 他 extra のパッケージを venv から削除してスイートを壊す。
> **numpy が `1.23.5` のままでも正しい。** `convert` extra が fairseq のため `<1.24` を保持しており、
> 両方が入りうる単一ロックでは交差集合 `[1.23, 1.24)` が選ばれる。`rvc`/`whisper` 側の上限撤廃は
> 後続 spec ③（Python 3.12 化）で効く — そこでは 3.11 専用の `convert` を入れないため。

- [ ] **Step 11: config の意味を更新する**

`vspeech/config.py:335`:

```python
    hubert_model_file: Path = Field(
        default=Path(),
        description="scripts/convert_hubert.py が出力した変換済み ContentVec 資産ディレクトリ",
    )
```

`config.toml.example:133`:

```toml
# scripts/convert_hubert.py が出力した変換済み ContentVec 資産ディレクトリ（旧: hubert_base.pt）
hubert_model_file = "./hubert_contentvec"
```

- [ ] **Step 12: change_voice golden を許容誤差化する（同じ commit で行うこと）**

実装が fairseq → transformers に変わった以上、bit-exact のままでは資産のある環境で
`test_change_voice_golden` がこの commit から落ちる。**同じ commit で緩めて赤を出さない。**

`tests/test_change_voice_golden.py` の docstring を差し替え:

```python
"""change_voice の音声回帰テスト（fairseq -> transformers 移行の end-to-end 検証）。

golden は **fairseq 版で捕獲したまま** 据え置く。HuBERT は eval + inference_mode 下で
RNG を一切消費しないため、seed_all() 後の RNG ストリームは実装差し替えの影響を受けない
(RNG を引くのは RVC synthesizer の infer だけ)。したがって golden との差は特徴量の値の
差だけに由来し、この照合は移行の等価性をそのまま証明する。

実装が変わった以上 bit-exact にはならないので、判定は許容誤差（相関 + セグメンタル SNR）。

golden npz / CUDA / RVC worker config ($VSPEECH_RVC_GOLDEN_CONFIG) が揃わなければ skip。
"""
```

import 群に追記（**しきい値は再定義せず import する**。単一情報源は `scripts/hubert_metrics.py`）:

```python
from scripts.hubert_metrics import CORR_MIN
from scripts.hubert_metrics import SNR_MIN_DB
from scripts.hubert_metrics import waveform_correlation
from scripts.hubert_metrics import waveform_snr
```

`test_change_voice_matches_seeded_golden` の末尾（現 55-61 行）を差し替え:

```python
    assert out.shape == golden.shape, f"length changed: {out.shape} vs {golden.shape}"
    # 実装が fairseq -> transformers に変わったので bit-exact ではない。
    # 緩めるときは実測値をこのコメントに残すこと（実測の 10 倍まで）。
    correlation = waveform_correlation(out, golden)
    snr_db = waveform_snr(golden, out)
    assert correlation >= CORR_MIN, f"correlation {correlation:.6f} < {CORR_MIN}"
    assert snr_db >= SNR_MIN_DB, f"waveform SNR {snr_db:.2f} dB < {SNR_MIN_DB} dB"
```

- [ ] **Step 13: 捕獲スクリプトの docstring を現実に合わせる**

`scripts/capture_change_voice_golden.py` の docstring 中の
「the golden test re-seeds identically and asserts exact equality」を次に差し替える:

```
seeded output as the golden. The golden test re-seeds identically; it asserted exact
equality while HuBERT ran under fairseq, and now asserts a tight tolerance
(correlation + waveform SNR) because the content encoder moved to transformers.
This npz must be captured while `vspeech/lib/rvc.py` still uses fairseq -- it is the
fairseq-side reference the migration is validated against.
```

- [ ] **Step 14: 全体ゲート**

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
Expected: 全て pass。資産の無い環境では `test_change_voice_golden` は skip。`tests/test_hubert_equivalence.py` はまだ存在しない。

- [ ] **Step 15: commit**

```bash
git add vspeech/lib/rvc.py vspeech/config.py config.toml.example pyproject.toml uv.lock \
        tests/conftest.py tests/test_hubert_runtime.py tests/test_no_fairseq_import.py \
        tests/test_change_voice_golden.py scripts/capture_change_voice_golden.py
git commit -m "feat(rvc)!: replace fairseq HuBERT with transformers HubertModel

runtime no longer imports fairseq. hubert_model_file now points at the
converted asset directory produced by scripts/convert_hubert.py. The
change_voice golden stays fairseq-captured but is asserted with a tolerance,
so it now validates the migration end-to-end instead of bit-exactness."
```

---

### Task 5: HuBERT 特徴量の等価ゲート

**Files:**
- Create: `tests/test_hubert_equivalence.py`

**Interfaces:**
- Consumes: `scripts.hubert_metrics.feature_cosine` / `feature_max_abs_diff`（Task 1）、`vspeech.lib.rvc.load_hubert_model` / `extract_features`（Task 4）、Task 3 が書き出した資産と golden
- Produces: なし（終端ゲート）

- [ ] **Step 1: 等価テストを書く**

`tests/test_hubert_equivalence.py`:

```python
"""transformers 版 HuBERT が fairseq 版と数値等価であることの主ゲート。

fairseq 時代に scripts/convert_hubert.py が捕獲した特徴量（fp32）を正解とし、
(9, use_final_proj=True) と (12, use_final_proj=False) の両方で照合する。

資産と golden は派生物なので gitignore してある。環境変数が未設定なら skip し、
CPU/CI のスイートを壊さない（tests/test_change_voice_golden.py と同じ流儀）。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

_ASSET_ENV = "VSPEECH_HUBERT_ASSET_DIR"
_GOLDEN_ENV = "VSPEECH_HUBERT_GOLDEN_DIR"

_asset = os.environ.get(_ASSET_ENV)
_golden = os.environ.get(_GOLDEN_ENV)
ASSET_DIR = Path(_asset) if _asset else None
GOLDEN_NPZ = Path(_golden) / "hubert_golden.npz" if _golden else None

# しきい値 (COSINE_MIN / MAX_ABS_MAX) の単一情報源は scripts/hubert_metrics.py。
# 緩めるときはそこで変更し、実測値を根拠としてコメントに残すこと（実測の 10 倍まで）。
pytestmark = pytest.mark.skipif(
    ASSET_DIR is None
    or not ASSET_DIR.exists()
    or GOLDEN_NPZ is None
    or not GOLDEN_NPZ.exists(),
    reason=f"${_ASSET_ENV} / ${_GOLDEN_ENV} not available",
)


@pytest.mark.parametrize(
    ("emb_output_layer", "use_final_proj", "golden_key"),
    [(9, True, "l9_proj"), (12, False, "l12_raw")],
)
def test_features_match_fairseq_golden(emb_output_layer, use_final_proj, golden_key):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    assert ASSET_DIR is not None and GOLDEN_NPZ is not None  # skipif guarantees; narrows for ty

    data = np.load(GOLDEN_NPZ)
    wav = data["wav"].astype(np.float32)
    reference = data[golden_key].astype(np.float32)

    device = torch.device("cpu")
    bundle = load_hubert_model(ASSET_DIR, device, is_half=False)
    out = extract_features(
        bundle,
        torch.from_numpy(wav).unsqueeze(0),
        device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    candidate = out.squeeze(0).float().cpu().numpy()

    assert candidate.shape == reference.shape, f"{candidate.shape} vs {reference.shape}"
    cosine = feature_cosine(candidate, reference)
    max_abs = feature_max_abs_diff(candidate, reference)
    assert cosine >= COSINE_MIN, f"cosine {cosine:.8f} < {COSINE_MIN}"
    assert max_abs <= MAX_ABS_MAX, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX:.1e}"
```

- [ ] **Step 2: 資産なしで skip されることを確認**

Run: `uv run pytest tests/test_hubert_equivalence.py -v`
Expected: `2 skipped`（環境変数未設定時）

- [ ] **Step 3: 資産なしのスイートが緑であることを確認**

Run: `uv run pytest -v`
Expected: 全 pass。`test_change_voice_golden` / `test_hubert_equivalence` は skip。

- [ ] **Step 4: commit**

```bash
uv run ruff format tests/test_hubert_equivalence.py
uv run ruff check tests/test_hubert_equivalence.py
git add tests/test_hubert_equivalence.py
git commit -m "test(rvc): add HuBERT feature-equivalence gate against the fairseq golden"
```

---

### Task 6: 実資産での検証（オペレータ）

> **エージェントへ: このタスクは実物の資産・RVC モデル・GPU を要する。実行環境が無ければユーザに依頼すること。数値を捏造しないこと。**

**Files:** なし（検証のみ。しきい値を緩める必要が出た場合のみ Task 5 のファイルを修正）

**Interfaces:**
- Consumes: Task 3 の資産・golden、Task 4 の runtime、Task 5 のゲート
- Produces: なし

- [ ] **Step 1: 等価ゲートを実資産で走らせる**

```bash
VSPEECH_HUBERT_ASSET_DIR=<path>/hubert_contentvec \
VSPEECH_HUBERT_GOLDEN_DIR=<path>/hubert_golden \
uv run --extra rvc pytest tests/test_hubert_equivalence.py -v
```
Expected: `2 passed`（`(9,True)` と `(12,False)`）

落ちた場合: **しきい値を緩める前に**、`resolve_layer_offset` が選んだ offset と `derive_hf_config` が導いた config を疑うこと。緩めてよいのは Task 3 Step 7 で実測した誤差の 10 倍までで、変更先は `scripts/hubert_metrics.py` の定数（単一情報源）。実測値を根拠としてそこにコメントで残すこと。

- [ ] **Step 2: 音声回帰を実資産で走らせる**

```bash
VSPEECH_RVC_GOLDEN_CONFIG=<path>/your-rvc-config.toml \
uv run --extra rvc pytest tests/test_change_voice_golden.py -v
```
Expected: `1 passed`（相関 >= 0.999、セグメンタル SNR >= 40 dB）

`[rvc] hubert_model_file` を変換済み資産ディレクトリに書き換えてから実行すること。

- [ ] **Step 3: 健全性ゲート一式**

```bash
uv run --extra rvc poe check
```
Expected: ruff format/lint、ty、pytest、lock-check、audit、security、deadcode が全て pass。

- [ ] **Step 4: fairseq が runtime から消えたことを確認**

```bash
uv run pytest tests/test_no_fairseq_import.py -v
uv tree --package voicerecog | grep -i fairseq || echo "OK: fairseq absent from rvc runtime"
```
Expected: `1 passed` と `OK: fairseq absent from rvc runtime`

- [ ] **Step 5: 実機の耳チェック**

実際に vc worker を起動し、変換前と聴感上の差が無いことを確認する（自動ゲートには含めない。VAD v6 移行と同じ運用）。

```bash
uv run --extra rvc python -m vspeech --config <path>/your-rvc-config.toml
```

- [ ] **Step 6: 結果を記録して commit**

しきい値を緩めた場合のみ:

```bash
git add scripts/hubert_metrics.py
git commit -m "test(rvc): record measured equivalence error and adjust thresholds"
```

緩めなかった場合は commit 不要。ブランチを PR に出す準備が整う。

---

## 完了条件

1. `tests/test_hubert_equivalence.py` が実資産で `(9,True)` / `(12,False)` の両方 pass。
2. `tests/test_change_voice_golden.py` が許容誤差内で pass。
3. `tests/test_no_fairseq_import.py` が pass（`vspeech/` に fairseq import 0 件）。
4. `uv run --extra rvc poe check` が pass。
5. `uv tree` の `rvc` 経路に fairseq が居ない。
6. 実機の耳チェックで差が無い。

これを満たした時点で、後続 spec ②（ONNX 化）と ③（`requires-python` 引き上げ）が着手可能になる。
