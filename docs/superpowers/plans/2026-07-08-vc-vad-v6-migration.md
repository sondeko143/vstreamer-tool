# VC VAD モデル v5→v6 移行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** VAD ノイズゲートの Silero VAD モデルを v5.1.2 から v6.2.1 に差し替える。

**Architecture:** v6.2.1 は v5 と同一の ONNX I/O 契約 (`input`/`state(2,·,128)`/`sr` → `output`/`stateN`) を共有する実測済みのドロップイン互換モデル。推論ラッパー ([vspeech/lib/vad.py](../../../vspeech/lib/vad.py)) と `"state"` 入力検証は無改修で v6 を受け付ける。よって作業は (1) コード/設定の "v5" 表記更新、(2) モデルファイルの取得・pin・sha 照合、(3) 品質ゲート + メモリ更新、の 3 タスク。挙動変更はモデル差し替えのみ。

**Tech Stack:** Python 3.11, numpy, onnxruntime (rvc extra, CPUExecutionProvider), Silero VAD v6.2.1 ONNX, pydantic v2, uv + poethepoet, pytest。

**Spec:** [docs/superpowers/specs/2026-07-08-vc-vad-v6-migration-design.md](../specs/2026-07-08-vc-vad-v6-migration-design.md)

## Global Constraints

- Python **3.11 only** (`>=3.11,<3.12`)。フロアを下げない。
- pin 対象: Silero VAD **v6.2.1** の `src/silero_vad/data/silero_vad.onnx`。
  - sha256 = `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
  - サイズ = 2327524 bytes
  - 取得 URL = `https://github.com/snakers4/silero-vad/raw/refs/tags/v6.2.1/src/silero_vad/data/silero_vad.onnx`
  - 参考 (退避する現行 v5.1.2 の sha256) = `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f`
- モデルファイルは **gitignore・非ベンダリング** (`rvc.rmvpe_model_file` と同じ運用)。repo にコミットしない。ローカルの `~/.config/vstreamer/` に配置する。
- **推論コードの挙動は変更しない。** `create_vad_session` の `"state" in input_names` 検証はそのまま (v5/v6 accept, v4 reject)。変更は docstring/コメント/エラー文/設定コメントの文言のみ。
- `VcConfig` の VAD 既定値は **据置**: `vad_gate=false`, `vad_threshold=0.5`, `vad_min_speech_ratio=0.1`, `vad_speech_pad_ms=100.0`, `vad_min_gain=0.0`。
- ruff `force-single-line = true` (import 1 行 1 個)。型チェックは ty。
- 歴史的記録 (`docs/superpowers/{specs,plans}/2026-07-08-vc-vad-noise-gate*`) は **改変しない**。
- コミットメッセージ末尾に必ず付ける:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- 作業ブランチ: `feat/vad-v6-migration` (spec は既にこのブランチにコミット済み)。

---

### Task 1: コード/設定の "v5"→"v5/v6" 表記更新 (挙動不変)

純粋に文言のみの変更。既存テストがグリーンのまま維持されることで正しさを担保する
(新しい挙動はないので TDD の「先に失敗するテスト」ではなく、既存のガードテストが変更
前後で緑であることを確認するサイクル)。特にエラー文は
`test_create_vad_session_missing_file_fails_loudly` が
`"silero" in str(...).lower()` を検証しているため、`Silero` の語を残す必要がある。

**Files:**
- Modify: `vspeech/lib/vad.py` (5 箇所)
- Modify: `config.toml.example:87`
- Test: `tests/test_vad_gate.py` (既存・変更なし。ガードとして実行)

**Interfaces:**
- Consumes: なし (単独タスク)
- Produces: 公開 API・シグネチャ・挙動に変更なし。`create_vad_session(model_file: Path) -> InferenceSession` と `speech_probs(session, audio_16k) -> NDArray[np.float64]` は不変。

- [ ] **Step 1: 変更前のガードテストがグリーンであることを確認 (baseline)**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra rvc pytest tests/test_vad_gate.py -v
```
Expected: PASS 13 + SKIP 1 (`test_real_model_silence_and_noise_score_low` は `VSPEECH_VAD_MODEL` 未設定でスキップ)。特に `test_create_vad_session_missing_file_fails_loudly` が PASS。

- [ ] **Step 2: `vspeech/lib/vad.py` の 5 箇所を編集**

編集 A — 契約コメント (現行 L19-20):
```python
# Silero VAD v5/v6 share this contract: 16kHz mono, 512-sample (32ms) windows,
# with a 64-sample context carried between windows and a (2, 1, 128) recurrent
# state. This module pins the v6.2.1 model.
```
(置換対象の現行 2 行:
```python
# Silero VAD v5 operates on 16kHz mono, 512-sample (32ms) windows, with a
# 64-sample context carried between windows and a (2, 1, 128) recurrent state.
```
)

編集 B — `create_vad_session` docstring (現行 L80-86):
```python
    """Build a CPU onnxruntime session for the Silero VAD model (v6.2.1).

    Fails loudly on a missing file or a model lacking the shared v5/v6
    state-input contract: silently passing audio through would mean the noise
    the gate exists to stop comes back unnoticed. CPU is deliberate -- the
    model is ~2MB and must not contend with RVC for the GPU.
    """
```

編集 C — `FileNotFoundError` メッセージ (現行 L93-97、`(v5)` → `(v6.2.1)` のみ):
```python
        raise FileNotFoundError(
            f"Silero VAD model not found: {path}. Download silero_vad.onnx"
            " (v6.2.1) from the snakers4/silero-vad repository and set"
            " vc.vad_model_file."
        )
```

編集 D — `ValueError` メッセージ (現行 L107-110、`v5 model` → `v5/v6 model`):
```python
        raise ValueError(
            f"{path} does not look like a Silero VAD v5/v6 model (inputs:"
            f" {sorted(input_names)}); v4 models (h/c inputs) are unsupported."
        )
```

編集 E — `speech_probs` docstring 1 行 (現行 L117):
```python
    Replicates the silero-vad v5/v6 wrapper: 512-sample windows, each prefixed
```
(置換対象:
```python
    Replicates the silero-vad v5 wrapper: 512-sample windows, each prefixed
```
)

- [ ] **Step 3: `config.toml.example:87` を編集**

新 (置換後):
```toml
# snakers4/silero-vad リポジトリの silero_vad.onnx (v6.2.1) を取得してパスを指定する。
```
(置換対象の現行行:
```toml
# snakers4/silero-vad リポジトリの silero_vad.onnx (v5) を取得してパスを指定する。
```
)

- [ ] **Step 4: フォーマット + 型チェック + テストがグリーンのまま**

Run:
```bash
uv run poe fmt
uv run poe lint
uv run poe type
uv run --extra rvc pytest tests/test_vad_gate.py -v
```
Expected: fmt/lint 変更なしまたは安全な整形のみ、ty はこのファイルについて新規エラーなし、pytest は Step 1 と同じ PASS 13 + SKIP 1。`test_create_vad_session_missing_file_fails_loudly` が引き続き PASS ("Silero" 語が残っているため)。

- [ ] **Step 5: コミット**

```bash
git add vspeech/lib/vad.py config.toml.example
git commit -m "docs(vc): retarget VAD wording to Silero v5/v6 shared contract (pin v6.2.1)

Wording-only: docstrings/comments/error messages in vad.py and the
config.toml.example VAD note now describe the shared v5/v6 state-input
contract and name v6.2.1 as the pinned model. No behavior change; the
'state'-input validation still accepts v5/v6 and rejects v4.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: v6.2.1 モデルの取得・pin・sha 照合 + real-model テスト注記

実際の挙動変更 (モデル差し替え)。モデルは repo 非管理なので、この差し替え自体は
ローカル手順で、検証は env-gated real-model テストが v6 で PASS することで担保する。
唯一の repo 変更は real-model テストへの v6 注記コメント。

**Files:**
- Local (非 repo): `~/.config/vstreamer/silero_vad.onnx` (v6 で上書き), `~/.config/vstreamer/silero_vad_v5.onnx` (退避)
- Modify: `tests/test_vad_gate.py` (`test_real_model_silence_and_noise_score_low` にコメント追加)

**Interfaces:**
- Consumes: Task 1 の `create_vad_session` / `speech_probs` (シグネチャ不変)。
- Produces: `~/.config/vstreamer/silero_vad.onnx` が v6.2.1 (sha256 `1a153a22…`)。config_vc.toml の `vad_model_file` は不変 (このパスを指し続ける)。

- [ ] **Step 1: 現行モデルが v5.1.2 であることを確認してから退避**

Run:
```bash
sha256sum "$HOME/.config/vstreamer/silero_vad.onnx"
# 期待 (v5.1.2): 2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f
cp "$HOME/.config/vstreamer/silero_vad.onnx" "$HOME/.config/vstreamer/silero_vad_v5.onnx"
```
Expected: sha256 が v5.1.2 の値と一致。退避コピー作成。
(万一 sha が一致しない場合は現行モデルが想定と違う。停止してユーザーに確認する。)

- [ ] **Step 2: v6.2.1 を一時ファイルへ取得し sha 照合**

Run (GitHub raw のレート制限に備え curl 内部リトライを使う):
```bash
curl -sSL --retry 8 --retry-delay 8 --retry-all-errors \
  -o "$HOME/.config/vstreamer/silero_vad_v6.2.1.onnx" \
  "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.2.1/src/silero_vad/data/silero_vad.onnx"
stat -c%s "$HOME/.config/vstreamer/silero_vad_v6.2.1.onnx"
sha256sum "$HOME/.config/vstreamer/silero_vad_v6.2.1.onnx"
```
Expected: size = `2327524`、sha256 = `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`。
(size が 199 前後や sha 不一致なら「429 Too Many Requests」等の HTML を掴んでいる。数分待って再取得する。)

- [ ] **Step 3: 照合済み v6 を config パスへ配置**

Run:
```bash
cp "$HOME/.config/vstreamer/silero_vad_v6.2.1.onnx" "$HOME/.config/vstreamer/silero_vad.onnx"
sha256sum "$HOME/.config/vstreamer/silero_vad.onnx"
```
Expected: config パスの sha256 = `1a153a22…` (v6.2.1)。

- [ ] **Step 4: env-gated real-model テストを v6 で実行 (差し替えの検証)**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
VSPEECH_VAD_MODEL="$HOME/.config/vstreamer/silero_vad.onnx" \
  uv run --extra rvc pytest tests/test_vad_gate.py -v
```
Expected: 全 14 テスト PASS (SKIP なし)。`test_real_model_silence_and_noise_score_low` が PASS = v6 モデルが読み込め、無音・ノイズで低スコアを返す (v6 実測: silence≈0.0017, noise 割合≈0 で閾値 0.3/0.1 を大きく下回る)。

- [ ] **Step 5: real-model テストに v6 注記コメントを追加**

`tests/test_vad_gate.py` の `test_real_model_silence_and_noise_score_low` 関数本体の
先頭 (`pytest.importorskip("onnxruntime")` の直前) に以下 2 行を挿入:
```python
    # Targets Silero VAD v6.2.1 (shares the v5 state-input contract). v6 scores
    # silence/noise even lower than v5, so these loose upper bounds still hold.
```
閾値 (`< 0.3`, `< 0.1`) は **変更しない** (過剰 fit を避ける)。

- [ ] **Step 6: テスト再実行 + コミット (repo 変更はコメントのみ)**

Run:
```bash
uv run poe fmt
VSPEECH_VAD_MODEL="$HOME/.config/vstreamer/silero_vad.onnx" \
  uv run --extra rvc pytest tests/test_vad_gate.py::test_real_model_silence_and_noise_score_low -v
git add tests/test_vad_gate.py
git commit -m "test(vc): note VAD real-model test now targets Silero v6.2.1

Model swapped locally (v5.1.2 -> v6.2.1, sha256 1a153a22...); the env-gated
real-model test passes unchanged since v6 shares the v5 I/O contract and
scores silence/noise even lower. Comment-only; thresholds unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: 対象テスト PASS、コメント差分のみコミット。

---

### Task 3: フル品質ゲート + メモリ更新

repo 全体のゲートを通し、アシスタントの永続メモリを v6 の事実に更新して締める。

**Files:**
- Test: リポジトリ全体 (`poe check`)
- Local (非 repo): `~/.claude/projects/<this-project>/memory/vc-vad-noise-gate.md` と `MEMORY.md`

**Interfaces:**
- Consumes: Task 1・2 の変更。
- Produces: なし (検証 + メモリ)。

- [ ] **Step 1: フル品質ゲート**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run poe check
```
Expected: `fmt-check`, `lint`, `type`, `test` (pytest+coverage), `lock-check` すべて PASS。
(`poe check` の `test` は `VSPEECH_VAD_MODEL` 未設定なので real-model テストは SKIP、それ以外の VAD テストと全既存テストが緑。)

- [ ] **Step 2: メモリ `vc-vad-noise-gate.md` を v6 事実に更新**

`~/.claude/projects/<this-project>/memory/vc-vad-noise-gate.md` の「Model fetched & verified」段落を、v6.2.1 が **契約互換で pin 対象**である事実に訂正する。要点 (誤りの「v6 do NOT use」を撤回):
  - pin は Silero VAD **v6.2.1** (`~/.config/vstreamer/silero_vad.onnx`, sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, 2327524B)。v5.1.2 (`2623a295…`) は `silero_vad_v5.onnx` に退避。
  - 実測: v5.1.2 と v6.2.1 の ONNX I/O は完全一致、v5 スタイル feed が v6 で動作、v6 はノイズ/無音でより低スコア。よって以前の「master は v6、契約を壊すから使うな」は **未検証の誤り**だった。
  - 推論コード・`"state"` 検証は無改修で v6 対応。移行 = モデル差し替え + 文言更新 (spec/plan: `docs/superpowers/{specs,plans}/2026-07-08-vc-vad-v6-migration*`)。
  - ロールバックは退避した `silero_vad_v5.onnx` を元パスへ戻すだけ (コード変更不要)。

`MEMORY.md` の該当インデックス行を、pin が v6.2.1 に上がった旨に 1 行更新する。

- [ ] **Step 3: 実機検証の申し送り (ユーザー手動)**

以下をユーザーへ申し送る (自動化しない — GPU ホスト + マイクが要る):
  - `vad_gate=true` + v6 モデルで vc を動かし、ブレス・環境音の録音チャンクがスキップされ (`vc_skip` ログ)、発話部分の音質・音量が v5 と同等以上であることを耳で確認する。
  - もし v6 が耳で劣る/取りこぼす場合は、`silero_vad_v5.onnx` を `silero_vad.onnx` に戻せば即ロールバック。必要なら `vad_threshold`/`vad_min_speech_ratio` を実機で微調整。

---

## Self-Review

**1. Spec coverage:**
- モデル取得・pin・sha 照合 → Task 2 (Step 1-4)。✓
- vad.py 文言更新 → Task 1 (Step 2)。✓
- config.toml.example 更新 → Task 1 (Step 3)。✓
- 既定値据置 → Global Constraints + Task では触れない (据置)。`test_vc_config_vad_defaults_are_off_and_sane` は poe check で緑。✓
- real-model テスト維持 + v6 注記 → Task 2 (Step 4-5)。✓
- 検証 (real-model / フルゲート / 実機) → Task 2 Step 4, Task 3 Step 1, Task 3 Step 3。✓
- メモリ更新 → Task 3 Step 2。✓
- スコープ外 (A/B なし等) → Global Constraints / 各タスクで新規依存や切替を導入せず。✓
- ロールバック → Task 2 で v5 退避、Task 3 Step 3 で手順明記。✓

**2. Placeholder scan:** "TBD"/"TODO"/曖昧指示なし。各編集は現行行と新行を明示。コマンドは期待値付き。✓

**3. Type consistency:** 公開シグネチャ (`create_vad_session`, `speech_probs`) は不変。文言変更のみでシンボル名の齟齬なし。sha256/サイズ/URL は Global Constraints と各 Step で同一値。✓
