# FCPE f0 Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rmvpe と同じ「16k 波形 + threshold → f0」契約の FCPE を第二の波形入力 ONNX f0 抽出器として追加し、rmvpe より低い f0 抽出レイテンシで RVC(vc) を動かせるようにする。

**Architecture:** まず捨て可能なスパイクで「FCPE が ONNX 化でき、かつ rmvpe.onnx よりこの環境で有意に速い」を go/no-go 判定する。緑のときだけ本実装へ進み、hubert のオフライン export 方式と rmvpe の抽出器統合方式を踏襲する。ONNX 資産は gitignore＋オフライン生成、torchfcpe は runtime 依存に入れず `uv run --with` overlay 専用。

**Tech Stack:** Python 3.14 / uv / torch(+cu130) / onnxruntime-gpu / torchfcpe(overlay) / onnxruntime InferenceSession / pytest。

**Spec:** `docs/superpowers/specs/2026-07-21-fcpe-f0-extractor-design.md`
**ADR:** `docs/adr/0049-fcpe-baked-waveform-onnx-f0-extractor.md` (Proposed → スパイク緑で Accepted 昇格)

## Global Constraints

- Python は `>=3.14,<3.15` のみ。floor を下げない。
- torchfcpe / onnx / onnxscript は **pyproject にも uv.lock にも入れない**。`uv run --with` overlay 専用（convert/export-hubert と同手口）。`tests/test_forbidden_imports.py` の対象を増やさない（`vspeech/` は torchfcpe を import しない）。
- 生成物 `fcpe.onnx` は **gitignore**＋オフライン生成のみ。配布物へ同梱しない。
- 全 GPU-capable onnxruntime セッションは `vspeech/lib/onnx_session.py` の `create_session` 経由（ADR-0024）。`InferenceSession(...)` を新規に別の場所で構築しない（`tests/test_onnx_session.py` が2ファイル限定を強制）。
- rmvpe が既定。FCPE はオプトイン。dio/harvest/rmvpe の既存挙動を変えない。
- import は1行1つ（ruff `force-single-line`）。型検査は `ty`。テストは pytest（`asyncio_mode=auto`）。
- **go/no-go バー**: fcpe.onnx の median f0 抽出レイテンシが同一条件の rmvpe.onnx より **≥30% 低い**、かつ export 成功、かつ f0 が torch FCPE と概ね一致。

---

# Phase 0 — スパイク（go/no-go、捨て可・製品コードへコミットしない）

スパイクは**スクラッチパッド**（`$SCRATCH` = このセッションの scratchpad ディレクトリ）に置く。緑なら Phase 1 で `scripts/export_fcpe_onnx.py` に昇格させる。Phase 0 は git にコミットしない。

### Task S1: torchfcpe の bundled model を調べ、export 可能な forward を特定する

**Files:**
- Create: `$SCRATCH/fcpe_spike/inspect.py`

**Interfaces:**
- Produces: メモ（bundled model の型、mel 計算の実体＝torchaudio か torch.stft か自前 conv か、net の forward、decode(local_argmax)の実体）。Phase 0 の後続と Phase 1 の export ツールがこの構造に依存する。

- [ ] **Step 1: overlay で torchfcpe を入れ、bundled model の構造を出力する**

```python
# $SCRATCH/fcpe_spike/inspect.py
import torch
import torchfcpe

m = torchfcpe.spawn_bundled_infer_model(torch.device("cpu"))
print("top:", type(m))
# 内部モジュールを列挙して mel / net / decoder の実体を掴む
for name, mod in m.named_modules():
    print(name, type(mod).__name__)
# infer の実装を読むためのヒント
print("has infer:", hasattr(m, "infer"))
```

Run: `uv run --isolated --no-project --with torchfcpe --with torch python $SCRATCH/fcpe_spike/inspect.py`
Expected: モジュール木が出る。mel が `torch.stft` 由来か torchaudio MelSpectrogram か自前 conv かを特定する。

- [ ] **Step 2: infer() の合成（mel→net→decode）をソースで確認する**

Run: torchfcpe のインストール先 `infer` / `mel_extractor` / `decoder` のソースを読む（`uv run --with torchfcpe python -c "import torchfcpe, inspect, os; print(os.path.dirname(torchfcpe.__file__))"` でパスを得て Read）。
Expected: forward を `def forward(self, waveform[1,N], threshold[1]) -> f0[1,T,1]` に再構成するのに必要な部品（mel 変換・net・local_argmax デコード・f0_min/f0_max のセント換算）が判明する。判明内容を `$SCRATCH/fcpe_spike/NOTES.md` に書く。

### Task S2: 波形入力 ONNX を export する（mel/STFT の export 可否を潰す）

**Files:**
- Create: `$SCRATCH/fcpe_spike/export.py`

**Interfaces:**
- Consumes: S1 の NOTES（mel/net/decode の実体）。
- Produces: `$SCRATCH/fcpe_spike/fcpe.onnx`（入力 `waveform (1,N)` fp32 + `threshold (1,)` fp32、出力 `f0 (Hz)` を index 0）。または「export 不可」の記録。

- [ ] **Step 1: export ラッパを書く（S1 で判明した部品で forward を合成）**

```python
# $SCRATCH/fcpe_spike/export.py  （骨子。S1 の実体名で埋める）
import torch
import torchfcpe

F0_MIN, F0_MAX, THRESHOLD = 65.0, 800.0, 0.006  # S1/torchfcpe 既定に合わせて確定

class FcpeWave(torch.nn.Module):
    def __init__(self, bundled):
        super().__init__()
        self.mel = bundled.mel_extractor        # ← S1 で実属性名に置換
        self.net = bundled.net                  # ← 同上
    def forward(self, waveform, threshold):
        mel = self.mel(waveform)                # mel/STFT がここで graph に入る
        f0 = self.net.infer_from_mel(mel, threshold=threshold, decoder="local_argmax",
                                     f0_min=F0_MIN, f0_max=F0_MAX)  # ← S1 で実 API に置換
        return f0  # (1, T, 1) Hz

bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
wrap = FcpeWave(bundled).eval()
wav = torch.zeros(1, 16000)
thr = torch.tensor([THRESHOLD], dtype=torch.float32)
torch.onnx.export(
    wrap, (wav, thr), "$SCRATCH/fcpe_spike/fcpe.onnx",
    input_names=["waveform", "threshold"], output_names=["f0"],
    dynamic_axes={"waveform": {1: "N"}, "f0": {1: "T"}},
    opset_version=17,  # STFT は opset>=17
)
```

- [ ] **Step 2: export を実行。STFT が通らなければ conv-mel に差し替える**

Run: `uv run --isolated --no-project --with torchfcpe --with torch --with onnx --with onnxscript python $SCRATCH/fcpe_spike/export.py`
Expected: `fcpe.onnx` が生成される。`torch.stft` が opset17 で export できない/実行時に落ちる場合は、mel を rmvpe export と同じ **conv1d ベースの mel**（STFT フィルタバンクを Conv1d の重みに焼く）に差し替えて再試行。ここで詰まったら S1 のメモに沿って原因を切り分け、**それでも不可なら go/no-go=NO-GO として Phase 0 を終了**（否定結果を NOTES.md に記録）。

### Task S3: レイテンシと f0 一致を計測して go/no-go を判定する

**Files:**
- Create: `$SCRATCH/fcpe_spike/bench.py`

**Interfaces:**
- Consumes: `$SCRATCH/fcpe_spike/fcpe.onnx`、既存 `rvc.rmvpe_model_file` が指す `rmvpe.onnx`。
- Produces: go/no-go 判定（レイテンシ比・f0 一致）を NOTES.md に記録。

- [ ] **Step 1: 同一音声で fcpe.onnx と rmvpe.onnx を GPU 実行しレイテンシ比較**

```python
# $SCRATCH/fcpe_spike/bench.py （骨子）
import time
import numpy as np
import torch
import torchfcpe
from vspeech.lib.onnx_session import create_session  # 一元ファクトリを使う（ADR-0024）

dev = torch.device("cuda")
# 代表的な VC チャンク: 16k mono, 例として 0.5s〜1.0s の実音声 or ノイズ
wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1

fcpe = create_session("$SCRATCH/fcpe_spike/fcpe.onnx", dev)
rmvpe = create_session("<rmvpe_model_file>", dev)  # 実 config の値を入れる

def run(sess, thr):
    x = np.expand_dims(wav, 0)
    for _ in range(5):  # warm
        sess.run(None, {"waveform": x, "threshold": np.array([thr], np.float32)})
    ts = []
    for _ in range(50):
        t = time.perf_counter(); sess.run(None, {"waveform": x, "threshold": np.array([thr], np.float32)}); ts.append(time.perf_counter()-t)
    return float(np.median(ts))

m_fcpe = run(fcpe, 0.006); m_rmvpe = run(rmvpe, 0.3)
print(f"fcpe={m_fcpe*1e3:.2f}ms rmvpe={m_rmvpe*1e3:.2f}ms speedup={(1-m_fcpe/m_rmvpe)*100:.1f}%")
```

Run: `uv run --extra rvc --with torchfcpe python $SCRATCH/fcpe_spike/bench.py`
Expected: 速度比が出る。**≥30% 高速でなければ go/no-go=NO-GO**。

- [ ] **Step 2: f0 一致を確認（torch FCPE を正解に）**

Run: 同スクリプトに、torch `spawn_bundled_infer_model().infer(wav, sr=16000, decoder_mode="local_argmax", threshold=0.006)` の f0 と、fcpe.onnx 出力の f0 を、有声フレームで相対誤差比較する処理を足して実行。
Expected: 有声フレームで概ね一致（大きく外れないこと）。外れる場合は S2 のデコード/レンジ焼き込みを見直す。

- [ ] **Step 3: go/no-go を記録して合流**

`$SCRATCH/fcpe_spike/NOTES.md` に「export 可否 / speedup% / f0 一致」を書き、**GO なら Phase 1 へ、NO-GO なら ADR-0049 を Deprecated にして終了**（`docs/adr/0049-...md` の Status を1行更新、README 索引も更新、否定結果を ADR Consequences 末尾に追記してコミット）。

---

# Phase 1 — 本実装（Phase 0 が GO のときだけ）

以降は `feat/fcpe-f0-extractor` ブランチにコミットする。

### Task 1: config に FCPE を足す

**Files:**
- Modify: `vspeech/config.py:360-380`（`F0ExtractorType`, `RvcConfig`）
- Test: `tests/test_config.py`（既存があればそこへ、無ければ新規）

**Interfaces:**
- Produces: `F0ExtractorType.fcpe`（値 `"fcpe"`）、`RvcConfig.fcpe_model_file: Path`（既定 `Path()`）。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_config.py
from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig

def test_fcpe_extractor_and_model_file():
    assert F0ExtractorType("fcpe") is F0ExtractorType.fcpe
    c = RvcConfig.model_validate({"f0_extractor_type": "fcpe", "fcpe_model_file": "x.onnx"})
    assert c.f0_extractor_type is F0ExtractorType.fcpe
    assert str(c.fcpe_model_file) == "x.onnx"
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run pytest tests/test_config.py::test_fcpe_extractor_and_model_file -v`
Expected: FAIL（`fcpe` が無い / `fcpe_model_file` が無い）

- [ ] **Step 3: 実装**

```python
# vspeech/config.py  F0ExtractorType に1行、RvcConfig に1フィールド
class F0ExtractorType(Enum):
    dio = "dio"
    harvest = "harvest"
    rmvpe = "rmvpe"
    fcpe = "fcpe"
# RvcConfig 内、rmvpe_model_file の直後に:
    fcpe_model_file: Path = Field(default=Path())
```

- [ ] **Step 4: 合格確認**

Run: `uv run pytest tests/test_config.py::test_fcpe_extractor_and_model_file -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add vspeech/config.py tests/test_config.py
git commit -m "feat(rvc): F0ExtractorType.fcpe と fcpe_model_file を追加 (ADR-0049)"
```

### Task 2: pitch_extract に fcpe 分岐 ＋ セッション名を汎用化

**Files:**
- Modify: `vspeech/lib/pitch_extract.py`（`pitch_extract_fcpe` 追加、`pitch_extract` の引数 `rmvpe_session` → `f0_session` に改名し fcpe 分岐追加）
- Modify: `vspeech/lib/rvc.py`（`change_voice` / `_select_pitch` の `rmvpe_session` → `f0_session`）
- Test: `tests/test_pitch_extract.py`（新規）

**Interfaces:**
- Consumes: `F0ExtractorType.fcpe`（Task 1）。
- Produces: `pitch_extract_fcpe(audio: Tensor, session: InferenceSession, threshold: float=0.006) -> NDArray`。`pitch_extract(...)` の f0 セッション引数名は `f0_session`（rmvpe/fcpe 共用）。`change_voice`/`_select_pitch` も同名に統一。

- [ ] **Step 1: 失敗するテストを書く（fake onnx セッションで分岐を検証）**

```python
# tests/test_pitch_extract.py
import numpy as np
import torch
from vspeech.config import F0ExtractorType
from vspeech.lib.pitch_extract import pitch_extract

class FakeSession:
    def __init__(self, f0): self._f0 = f0
    def run(self, out, feed):
        assert "waveform" in feed and "threshold" in feed  # fcpe/rmvpe 共通契約
        return [self._f0]

def test_pitch_extract_dispatches_to_fcpe():
    f0 = np.full(50, 220.0, dtype=np.float32)
    audio = torch.zeros(1, 16000)
    coarse, f0bak = pitch_extract(
        audio[0], f0_up_key=0, sr=16000, window=160,
        f0_extractor=F0ExtractorType.fcpe, f0_session=FakeSession(f0),
    )
    assert f0bak.shape[0] == 50
    assert np.allclose(f0bak, 220.0)
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run pytest tests/test_pitch_extract.py::test_pitch_extract_dispatches_to_fcpe -v`
Expected: FAIL（`f0_session` 引数が無い / fcpe 分岐が無い）

- [ ] **Step 3: 実装（fcpe は rmvpe と同一 I/O 契約なので session.run を共有）**

```python
# vspeech/lib/pitch_extract.py
FCPE_THRESHOLD = 0.006

def pitch_extract_fcpe(audio, session, threshold: float = FCPE_THRESHOLD):
    """FCPE onnx（waveform (1,N)+threshold→f0 Hz を index0）から f0 を取る。
    rmvpe.onnx と同一契約なので pitch_extract_rmvpe と入出力は同型。"""
    audio_num = np.expand_dims(audio.detach().cpu().numpy().astype(np.float32), axis=0)
    onnx_f0 = cast(NDArray[np.float32], session.run(
        None, {"waveform": audio_num, "threshold": np.array([threshold], dtype=np.float32)})[0])
    return cast(NDArray[np.double], onnx_f0.squeeze())

# pitch_extract(...) の引数名 rmvpe_session -> f0_session に改名し、分岐追加:
#   elif f0_extractor == F0ExtractorType.rmvpe:
#       if not f0_session: raise ValueError("f0 onnx session is not provided.")
#       f0 = pitch_extract_rmvpe(audio, session=f0_session)
#   elif f0_extractor == F0ExtractorType.fcpe:
#       if not f0_session: raise ValueError("f0 onnx session is not provided.")
#       f0 = pitch_extract_fcpe(audio, session=f0_session)
```

`vspeech/lib/rvc.py` の `_select_pitch` と `change_voice` の引数 `rmvpe_session` を `f0_session` に改名し、`pitch_extract(..., f0_session=f0_session)` に渡す（`rmvpe_session=` の呼び出しも同様に改名）。

- [ ] **Step 4: 合格確認＋既存 golden 不変**

Run: `uv run pytest tests/test_pitch_extract.py -v` → PASS
Run: `uv run --extra rvc pytest tests/test_rvc*.py -v`（既存 change_voice/rmvpe テスト）→ PASS（改名で壊れていない）

- [ ] **Step 5: コミット**

```bash
git add vspeech/lib/pitch_extract.py vspeech/lib/rvc.py tests/test_pitch_extract.py
git commit -m "feat(rvc): pitch_extract に fcpe 分岐を追加し f0 セッション名を汎用化 (ADR-0049)"
```

### Task 3: vc worker で fcpe セッションをロードする

**Files:**
- Modify: `vspeech/worker/vc.py:187-217`（セッション load 分岐、`change_voice(..., f0_session=...)`）
- Test: 手動起動確認（worker は統合点なのでユニットは Task 2 で担保。ここは smoke）

**Interfaces:**
- Consumes: `F0ExtractorType.fcpe`（Task1）、`change_voice(f0_session=...)`（Task2）。

- [ ] **Step 1: セッション load を rmvpe/fcpe 両対応にする**

```python
# vspeech/worker/vc.py （187-190 を置換）
        if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
            f0_session = create_session(rvc_config.rmvpe_model_file, device)
        elif rvc_config.f0_extractor_type == F0ExtractorType.fcpe:
            f0_session = create_session(rvc_config.fcpe_model_file, device)
        else:
            f0_session = None
```

warmup とループ内の `change_voice(..., rmvpe_session=rmvpe_session)` を `f0_session=f0_session` に置換（216 行と後段 275 行相当の2箇所）。

- [ ] **Step 2: 型検査＋起動 smoke（entry point を実際に走らせる）**

Run: `uv run --extra rvc ty check vspeech/worker/vc.py vspeech/lib/rvc.py vspeech/lib/pitch_extract.py`
Expected: 追加エラー無し
Run: fcpe を指す最小 config で `uv run --extra rvc python -m vspeech --config <fcpe.toml>` を起動し、`vc worker started` まで到達することを確認（**テストだけでなく entry point を走らせる**）。

- [ ] **Step 3: コミット**

```bash
git add vspeech/worker/vc.py
git commit -m "feat(rvc): vc worker で fcpe の f0 セッションをロード (ADR-0049)"
```

### Task 4: preflight で fcpe_model_file を検証する

**Files:**
- Modify: `vspeech/preflight.py:197-205`（rmvpe チェックの隣に fcpe チェック）
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `F0ExtractorType.fcpe`, `RvcConfig.fcpe_model_file`（Task1）。
- Produces: fcpe 選択かつ `fcpe_model_file` 不在時に `ConfigProblem(field="rvc.fcpe_model_file")`。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_preflight.py
from pathlib import Path
from vspeech.config import F0ExtractorType
# 既存の preflight テストのヘルパに倣って rvc を組む
def test_fcpe_missing_model_file_flagged(make_rvc_config):
    cfg = make_rvc_config(f0_extractor_type=F0ExtractorType.fcpe, fcpe_model_file=Path("nope.onnx"))
    problems = collect_problems(cfg)  # 既存テストの呼び方に合わせる
    assert any(p.field == "rvc.fcpe_model_file" for p in problems)
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run pytest tests/test_preflight.py::test_fcpe_missing_model_file_flagged -v`
Expected: FAIL

- [ ] **Step 3: 実装**

```python
# vspeech/preflight.py  rmvpe チェックの直後に:
    if rvc.f0_extractor_type == F0ExtractorType.fcpe:
        if not rvc.fcpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    w,
                    f"rvc.fcpe_model_file '{rvc.fcpe_model_file}' が存在しません",
                    field="rvc.fcpe_model_file",
                )
            )
```

- [ ] **Step 4: 合格確認**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: PASS（GUI は ADR-0045 で自動追従、コード追加不要）

- [ ] **Step 5: コミット**

```bash
git add vspeech/preflight.py tests/test_preflight.py
git commit -m "feat(rvc): preflight で fcpe_model_file を検証 (ADR-0049, ADR-0045)"
```

### Task 5: オフライン export ツール `poe export-fcpe-onnx`

**Files:**
- Create: `scripts/export_fcpe_onnx.py`（Phase 0 の `export.py` を昇格＋golden 自己検証）
- Modify: `poe_tasks.toml`（`export-fcpe-onnx` タスク、`convert-hubert` の隣）
- Modify: `pyproject.toml`（`[tool.ty.overrides]` の `include` に `scripts/export_fcpe_onnx.py` を追加）
- Modify: `.gitignore`（`fcpe.onnx` を除外、既に hubert 資産を除外している近辺）

**Interfaces:**
- Produces: `fcpe.onnx`（Task2/3 が `create_session` で開く波形入力契約）。torch FCPE を正解に数値等価を自己検証。

- [ ] **Step 1: export＋golden 自己検証スクリプトを書く**

Phase 0 `export.py` の FcpeWave を移植し、export 後に torch FCPE と fcpe.onnx の f0 を有声フレームで比較して閾値超過なら非0 exit する（`scripts/export_hubert_onnx.py` の golden 検証と同型）。CLI は `--output fcpe.onnx --golden ./fcpe_golden` を受ける。

- [ ] **Step 2: poe タスクを追加**

```toml
# poe_tasks.toml, convert-hubert の隣
# FCPE を波形入力 ONNX に export (offline)。torchfcpe は overlay 専用 (pyproject/uv.lock に載せない)。
export-fcpe-onnx = { cmd = "uv run --with torchfcpe --with onnx --with onnxscript python -m scripts.export_fcpe_onnx", help = "FCPE -> waveform-input ONNX (offline, project env + overlay)" }
```

- [ ] **Step 3: ty override と gitignore を更新**

`pyproject.toml` の `[[tool.ty.overrides]]` `include` に `scripts/export_fcpe_onnx.py` を足す（torchfcpe 未解決 import を許容、hubert スクリプトと同扱い）。`.gitignore` に `fcpe.onnx`（および `fcpe_golden/`）を追加。

- [ ] **Step 4: 実行して資産生成＋自己検証が通ることを確認**

Run: `uv run poe export-fcpe-onnx --output fcpe.onnx --golden ./fcpe_golden`
Expected: `fcpe.onnx` 生成、golden 等価 OK（非0 exit しない）。`poe --help` が cp1252 で落ちないこと（help は ASCII）。

- [ ] **Step 5: コミット**

```bash
git add scripts/export_fcpe_onnx.py poe_tasks.toml pyproject.toml .gitignore
git commit -m "feat(rvc): poe export-fcpe-onnx で FCPE をオフライン ONNX 化 (ADR-0049)"
```

### Task 6: golden 数値等価テスト

**Files:**
- Create: `tests/test_fcpe_onnx.py`
- Test 用 env: `VSPEECH_FCPE_GOLDEN_*`（`test_rvc` の golden env 方式に倣う。資産が無ければ skip）

**Interfaces:**
- Consumes: `fcpe.onnx`（Task5）、torch FCPE 参照。

- [ ] **Step 1: golden 等価テストを書く（資産が無ければ skip）**

```python
# tests/test_fcpe_onnx.py
import os
import numpy as np
import pytest

_ASSET = os.environ.get("VSPEECH_FCPE_ONNX")

@pytest.mark.skipif(not _ASSET, reason="fcpe.onnx が無い (uv run poe export-fcpe-onnx で生成)")
def test_fcpe_onnx_matches_torch():
    import torch
    import torchfcpe
    from vspeech.lib.onnx_session import create_session
    wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    ref = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).infer(
        torch.from_numpy(wav).unsqueeze(0), sr=16000, decoder_mode="local_argmax", threshold=0.006)
    sess = create_session(_ASSET, torch.device("cpu"))
    got = sess.run(None, {"waveform": wav[None], "threshold": np.array([0.006], np.float32)})[0].squeeze()
    ref_np = ref.squeeze().cpu().numpy()
    voiced = ref_np > 0
    assert np.allclose(got[voiced], ref_np[voiced], rtol=0.05, atol=1.0)
```

- [ ] **Step 2: 実行**

Run: `VSPEECH_FCPE_ONNX=./fcpe.onnx uv run --with torchfcpe pytest tests/test_fcpe_onnx.py -v`
Expected: PASS（資産未指定なら skip）

- [ ] **Step 3: コミット**

```bash
git add tests/test_fcpe_onnx.py
git commit -m "test(rvc): FCPE onnx と torch の f0 数値等価 golden (ADR-0049)"
```

### Task 7: ドキュメント＋ADR 昇格

**Files:**
- Modify: `config.toml.example`（rvc セクションに `f0_extractor_type = "fcpe"` と `fcpe_model_file` の例＋生成コマンド）
- Modify: `CLAUDE.md`（f0 抽出器の記述に fcpe を追記、オフライン生成手順）
- Modify: `docs/adr/0049-fcpe-baked-waveform-onnx-f0-extractor.md`（Status を Accepted に）、`docs/adr/README.md`（索引 Status）

**Interfaces:** なし（ドキュメント）。

- [ ] **Step 1: config.toml.example と CLAUDE.md を更新**

fcpe の選び方、`fcpe_model_file` の意味、`uv run poe export-fcpe-onnx` での生成手順、rmvpe との違い（速いが精度は落ちる）を追記。

- [ ] **Step 2: ADR-0049 を Accepted に昇格（実装が裏づけた）**

`docs/adr/0049-...md` の `Status: Proposed` を `Status: Accepted` に1行更新、`README.md` 索引の Status も更新。スパイクで測った実 speedup% を Consequences 末尾に1行追記してよい。

- [ ] **Step 3: 全ゲート＋コミット**

Run: `uv run --all-extras poe check`
Expected: 既存の accepted 失敗（torch CVE / deadcode）以外に新規失敗が無い。

```bash
git add config.toml.example CLAUDE.md docs/adr/0049-fcpe-baked-waveform-onnx-f0-extractor.md docs/adr/README.md
git commit -m "docs(rvc): FCPE の使い方を記載し ADR-0049 を Accepted に昇格"
```

---

## Self-Review（spec 突合）

- **受入: レイテンシ < rmvpe 実測** → Phase0 Task S3（≥30% ゲート）。GO の記録が受入証跡。✅
- **受入: fcpe を選べる** → Task 1（config）。✅
- **受入: 動く・妥当な f0** → Task 3（worker 起動 smoke）＋ Task 6（f0 一致）。✅
- **受入: アセット不正で preflight が field 付きで fail** → Task 4。✅
- **受入: torch と許容誤差内一致をテスト** → Task 6。✅
- **受入: オフライン生成手順の文書化** → Task 5（poe タスク）＋ Task 7（docs）。✅
- **受入: 既存抽出器・既定挙動が不変** → Task 2 Step4（既存 golden 不変）＋ rmvpe 既定のまま。✅
- **非ゴール（onnx 同梱しない / 全パラメータ可変化しない / 既定置換しない）** → Task5 で gitignore、threshold/レンジは export 固定、rmvpe 既定維持。✅

型整合：f0 セッション引数は全経路 `f0_session` に統一（Task2 で pitch_extract/rvc、Task3 で vc worker）。`pitch_extract_fcpe(audio, session, threshold)` は Task2 で定義し Task5/6 が同 I/O を使う。プレースホルダ無し（Phase0 の骨子は spike の性質上 S1 で実属性名を確定する旨を明記）。
