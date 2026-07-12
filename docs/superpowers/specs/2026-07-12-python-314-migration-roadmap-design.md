# Python 3.14 移行ロードマップ 設計書

- 日付: 2026-07-12
- ステータス: 承認済み（実装計画へ）
- ブランチ: **フェーズ1** `feat/numpy2-on-312`（本 spec が詳細化する範囲）。フェーズ2/3 は別ブランチ・別 spec。
- 対象: `pyproject.toml` の依存制約 / `uv.lock` / `requirements-pod.txt`、および numpy 2・CUDA 13 が波及する GPU・音声経路（[rvc.py](../../../vspeech/lib/rvc.py), [pitch_extract.py](../../../vspeech/lib/pitch_extract.py), [vc.py](../../../vspeech/worker/vc.py), [recording.py](../../../vspeech/worker/recording.py), [playback.py](../../../vspeech/worker/playback.py)）

## 1. 背景と目的

`requires-python` は現在 `>=3.12,<3.13`（[pyproject.toml:6](../../../pyproject.toml)）。本 spec は **3.14 への引き上げを 3 フェーズに分解**し、その最初のフェーズ（numpy 2 化）を実装可能なところまで詳細化する。

**なぜ 3.14 か（3.13 ではなく）。** 3.12 は bugfix フェーズ終了済みで security-only は 2028-10 まで（PEP 693）。3.13 は **runtime 利得ゼロ**——incremental GC は 3.13.0 final 直前に revert され、free-threading は experimental——かつ 3.14（EOL 2030-10、free-threading 正式サポート）に劣る。3.13 を経由する理由が無い。

**正直な費用便益。** ただし本ワークロードでは **3.14 も runtime 利得はほぼゼロ**である。レイテンシは GPU 演算ではなく音声長とバッファリングに律速され（telemetry: transc ~0.38s / RVC vc ~0.55s、処理ボトルネック無し）、free-threading は無意味（torch / onnxruntime-gpu / grpcio / ctranslate2 はいずれも `Py_mod_gil` 未対応で、1 つでも import すれば GIL がプロセス全体で復活する。voicevox-core には free-threaded ABI が無い）。**得られるのはサポート窓の延長のみ**であり、緊急性は無い。これは性能移行ではなく保守性移行である。

**3.14 の唯一の wheel ギャップは `pyworld`**（cp314 wheel 不在、後述 1.1）。それ以外の重い依存はすべて cp314 wheel を持つ。

**併せて CUDA 13 化を検討する理由。** onnxruntime-gpu は **1.27.0 で CUDA 13 へ移行**した（後述 1.2 の地雷）。CUDA 12 に留まると onnxruntime-gpu を `<1.27` で**恒久的に cap** することになり、いずれ修正も新 opset も来ない版に取り残される。CUDA 13 化はこの縛りを解く。runtime 利得はほぼ無いが、価値は **onnxruntime の更新可能性を維持すること**にある。コストは全 GPU ホストの **NVIDIA ドライバ R580+ 化**（リポジトリ外）。

### 1.1 検証済み wheel 状況（2026-07-12、PyPI / download.pytorch.org 実確認）

| 依存 | cp314 wheel（Windows / Linux） | 備考 |
|---|---|---|
| torch / torchaudio 2.10.0**+cu128** | ✅ cp314 + cp314t 両プラットフォーム | 現行ピンそのものが cp314 を配布済み |
| torch / torchaudio 2.10.0**+cu130** | ✅ cp312 / cp313 / cp314 win + linux | CUDA 13 経路（フェーズ2以降） |
| onnxruntime-gpu **1.26.0**（CUDA 12） | ✅ cp314 win + linux | `nvidia-*-cu12` 依存。CUDA 12 のまま 3.14 に届く |
| onnxruntime-gpu **1.27.0**（CUDA 13） | ✅ cp314 win + linux | `nvidia-*-cu13` 依存。torch cu128 と衝突（1.2） |
| numpy ≥2.1 | ✅ cp314 | numpy 1.26.4 は cp314 wheel 無し → `<2` cap 撤廃が必須 |
| voicevox-core 0.16.4 | ✅ cp310-abi3 | 3.10 以降どこでも import 可 |
| pyvcroid2 | ✅ 純 Python | ABI 非依存 |
| **pyworld 0.3.5** | ❌ **cp314 wheel 無し** | 最終リリース 2025-01。`Requires-Python` null → 3.14 では sdist ビルドに落ちる |

`numpy<2` cap が撤廃を強制される論理は前 spec で確認済み: numpy 1.26.4 に cp313/cp314 wheel は無く、`Requires-Python` が上限なし（`>=3.9`）のためリゾルバは候補から外さず sdist ビルドを試みて失敗する。そして **cp314 wheel を配る依存は numpy 1.26 でビルドできない以上必然的に numpy 2 ビルド**であり、cap を上げれば numpy 2 は自動的に随伴する。自コードは NEP 50 安全（後述 3.3）。

### 1.2 発見した地雷: onnxruntime-gpu の CUDA 13 化

現在 `uv.lock` は onnxruntime-gpu **1.26.0**（CUDA 12、`nvidia-*-cu12`）に解決している（[uv.lock:663](../../../uv.lock)）。制約は `>=1.24.4,<2`（[pyproject.toml:37,44](../../../pyproject.toml)）で、**1.27.0（CUDA 13）を許してしまう**。

`uv lock` を再生成すると onnxruntime-gpu が 1.27.0 に上がり、**torch `+cu128`（CUDA 12.8）と 1 プロセス内で CUDA 12 / CUDA 13 の 2 ランタイムが同居**する。これは GPU 経路（RVC decoder / HuBERT / RMVPE / whisper）を静かに壊しうる。

**対処は明示 cap。** フェーズ1では `<1.27` で CUDA 12 を凍結し、CUDA 13 の採否をフェーズ2に切り出す。cap にはコメントを添える——添えないと次の `uv lock` が黙って 1.27 を戻す。

## 2. 3 フェーズ・ロードマップ（1 フェーズ = 1 変数）

移行を貫く原則は **「CI で検証できない実機変更を、版 churn と混ぜない」**（[docs/follow-ups.md](../../follow-ups.md) の spec ③ 教訓）。numpy 1→2・CUDA 12.8→13・Python 3.12→3.14 は**いずれも GPU / 音声の実機経路に触れる独立変数**であり、同時に動かすと切り分け不能になる。よって各フェーズが動かす変数は 1 つだけとする。

| フェーズ | 動かす唯一の変数 | 完了後の依存状態 | ブランチ / spec |
|---|---|---|---|
| **1. numpy 2**（3.12 のまま） | numpy 1.26 → 2.x | torch **cu128**、onnxruntime **1.26**（CUDA 12、`<1.27` で cap） | `feat/numpy2-on-312`（**本 spec**） |
| **2. CUDA 13**（3.12・numpy 2） | CUDA 12.8 → 13 | torch **cu128→cu130**、torchaudio cu130、onnxruntime **1.26→1.27**（cap を `>=1.27` へ） | 別 spec |
| **3. Python 3.14**（CUDA 13・numpy 2） | Python 3.12 → 3.14 | torch/onnxruntime の cp312→cp314 URL、+ audioop-lts、pyworld 遅延 import + extra 撤去、ruff/ty target、Docker 3.14-slim | 別 spec |

onnxruntime の cap が**フェーズをまたいで進化**する点に注目——フェーズ1で `<1.27`（CUDA 12 凍結）、フェーズ2で `>=1.27`（CUDA 13 採用）。どのフェーズも「numpy 変更 × CUDA 変更 × Python 変更」を混ぜない。CUDA 13 化を先に済ませることで、3.14 到達時には onnxruntime 1.27/CUDA 13 が**検証済みの土台**になっており、cap と戦わずに済む。

## 3. フェーズ1 設計（本 spec の実装範囲）: numpy 2 化（3.12 のまま）

危険で実機依存な半分を最初に切り出す。Python を 3.12 に固定することで **numpy 1→2 を唯一の可変変数**にする。RVC 出力が変わる・C 拡張が誤動作すれば、原因は numpy に確定できる。

### 3.1 スコープ / 非目標

- **スコープ**:
  - `whisper` / `rvc` extra の `numpy>=1.23,<2` → `numpy>=2,<3` 緩和（2 箇所）。
  - onnxruntime-gpu を `>=1.24.4,<2` → `>=1.24.4,<1.27` に cap（CUDA 12 凍結、1.2）。**この cap はフェーズ2ではなくここで入れる**——numpy 移行中の lock 再生成が 1.27/CUDA 13 を引き込むと numpy と CUDA の 2 障害が重なるため。
  - `uv lock` 再生成 + `make` による `requirements-pod.txt` 再生成。
  - **実機での numpy 2 再検証**（3.4。CI では検証不能）。

- **非目標（後続フェーズ / YAGNI）**:
  - **`requires-python` の変更**・torch/onnxruntime の版や CUDA の変更 — フェーズ2/3。フェーズ1は **3.12 / cu128 / onnxruntime 1.26 のまま**完結する。
  - audioop → audioop-lts、pyworld の遅延 import・extra 撤去、ruff/ty target、Docker、classifiers — すべてフェーズ3。
  - `scipy` / `faiss-cpu` / `ctranslate2` の制約変更 — 不要。lock 再生成が numpy 2 対応ビルドへ自動追随する（3.2）。
  - アプリケーションコードの変更 — 不要（3.3）。

### 3.2 依存変更

**`pyproject.toml`**

- `whisper` extra（[pyproject.toml:38](../../../pyproject.toml)）: `numpy>=1.23,<2` → `numpy>=2,<3`。
- `rvc` extra（[pyproject.toml:46](../../../pyproject.toml)）: `numpy>=1.23,<2` → `numpy>=2,<3`。
- `whisper` / `rvc` extra の onnxruntime-gpu（[pyproject.toml:37,44](../../../pyproject.toml)）: `>=1.24.4,<2` → `>=1.24.4,<1.27`。**コメント必須**——「1.27.0 は CUDA 13（`nvidia-*-cu13`）へ移行。torch `+cu128` と衝突するため CUDA 12 を凍結。CUDA 13 化は別 spec」。

**lock の随伴（明示変更ではなく、`uv lock` の結果として起こること）**

- `faiss-cpu` は現在 1.8.0.post1（[uv.lock:254](../../../uv.lock)、numpy 1 ビルドの疑い）。numpy 2 制約下で lock を再生成すると numpy 2 対応版（1.9+ 系）へ上がる。`>=1.7.2,<2` の範囲内。
- `scipy`（現 1.15.3）・`ctranslate2`（現 4.8.0）はいずれも numpy 2 対応済みで、現制約のまま解決される見込み。上がっても既存レンジ内。
- これらは**意図された随伴**であり、検証（3.4）で実際に解決されることを確認する。

**判定**: `uv lock` が通り、`uv sync --all-extras` で numpy が 2.x に解決され、rvc / whisper 経路が import できること。

### 3.3 コード変更が不要な根拠（NEP 50 安全性）

numpy 2 の主な破壊面は (a) 値ベースのスカラー昇格（NEP 50）と (b) C 拡張の ABI。本 repo はいずれも安全:

- **整数 PCM 経路は numpy 非依存**。dBFS 算出（[recording.py:66](../../../vspeech/worker/recording.py) `audioop.rms`）と再生音量スケール（[playback.py:127](../../../vspeech/worker/playback.py) `audioop.mul`）は `bytes` に対する audioop 演算で、numpy を通らない。numpy 2 は録音の開始/停止判定にも音量にも触れない。
- **float↔int16 変換はすべて明示クランプ済み**。RVC 出力（[rvc.py:410](../../../vspeech/lib/rvc.py) `torch.clamp(... ).to(int16)`）、VC のゲイン適用（[vc.py:139](../../../vspeech/worker/vc.py) `np.clip(...).astype(np.int16)`）はいずれも `astype(float)` → `np.clip` → キャストの形で、弱スカラーの値ベース昇格に依存しない。
- **C 拡張の ABI** は cap を上げれば numpy 2 ビルドが随伴する（1.1 の論理）。onnxruntime-gpu 1.26.0 / scipy / faiss / ctranslate2 / pyworld はいずれも numpy 2 対応 wheel を持つ。

したがってフェーズ1は**書き換えではなく検証**が本体である。

### 3.4 検証プロトコル（実機。この環境には GPU / 音声デバイスが無い）

1. `uv sync --all-extras` が numpy 2.x で解決すること（`uv tree | grep numpy` で 2.x を確認）。
2. `uv run --all-extras poe check` が green（ruff format/lint、ty、pytest、lock-check、audit、security、deadcode）。
3. **seeded RVC 数値 golden**（環境変数 `VSPEECH_RVC_GOLDEN_CONFIG`、`--extra rvc`）が既存のシード下で再現すること。これは `change_voice` 経路の numpy 2 退行を**耳を使わずに**検知する主資産（seeded bit-exact、PR #14 由来）。
4. 実機耳チェック（最終確認、自動ゲート外）: 実機 config で RVC ボイチェン + whisper 文字起こしを走らせ、アーティファクト・誤出力が無いこと。VAD v6 移行と同じ運用。

## 4. フェーズ2 素描（別 spec）: CUDA 13 化（3.12・numpy 2 のまま）

CUDA 12.8 → 13 を唯一の変数として動かす。

- **依存**: torch/torchaudio の `[tool.uv.sources]` URL を `+cu128` → `+cu130`（cp312 win、[検証済み 1.1]）。onnxruntime-gpu cap を `>=1.27,<2` へ（CUDA 13 採用）。
- **前提（リポジトリ外・要確認）**: 全 GPU ホストの NVIDIA ドライバ **R580+**（CUDA 13.0 の最低要件。CUDA 12.8 は R570+）。CUDA 13 は Maxwell/Pascal/Volta（sm_70 以下）を drop——Ada 4060 / Blackwell は無関係だが、旧 GPU が経路に無いこと。
- **要検証（実機）**: **混在 CUDA の同一プロセス動作**。ctranslate2（faster-whisper）は自前の CUDA 12 ランタイムを同梱し 13 へ移らないため、フェーズ2完了後は CUDA 12（ctranslate2）と CUDA 13（torch / onnxruntime）の DLL が同居する。R580+ ドライバは両方を走らせるが、耳チェック前に実挙動を確認する。
- **判定**: `uv sync --all-extras` 解決、`poe check` green、RVC/whisper の実機動作、seeded RVC golden 再現。

## 5. フェーズ3 素描（別 spec）: Python 3.14 化（CUDA 13・numpy 2 のまま）

numpy 2・CUDA 13 が実機で確認済みの土台の上で、版 churn のみを行う。

- **`requires-python`** `>=3.12,<3.13` → `>=3.14,<3.15`。classifiers を 3.14 へ。`[tool.ty.environment].python-version` を `3.14` へ（ruff は `requires-python` から target を自動決定）。
- **torch/torchaudio** `[tool.uv.sources]` URL を cp312 → **cp314**（cu130、[検証済み 1.1]）。onnxruntime-gpu は既に 1.27（cp314・CUDA 13）で cap 変更不要。
- **audioop → audioop-lts**: base `dependencies` に `audioop-lts ; python_version >= '3.13'` を追加（audioop は [recording.py](../../../vspeech/worker/recording.py) / [playback.py](../../../vspeech/worker/playback.py) の core 経路で使う。audioop-lts の `Requires-Python` が `>=3.13` のためマーカー必須。C ポートで飽和セマンティクスを保存し、[follow-ups](../../follow-ups.md) の wrap 罠を回避）。**3 箇所のコードは無変更**。
- **pyworld**（cp314 wheel 不在への対処、承認済み方針）: [pitch_extract.py:5](../../../vspeech/lib/pitch_extract.py) の `import pyworld` を `pitch_extract_dio` / `pitch_extract_harvest` の**関数内へ遅延化**し、既定 `f0_extractor_type` を `harvest` → **`rmvpe`**（[config.py:343](../../../vspeech/config.py)）へ変更。`rvc` extra から pyworld を撤去（or 別サブ extra へ）。実機は rmvpe なので影響なし。dio/harvest は pyworld 手動導入時のみ。
- **ruff target 昇格の随伴**: `requires-python` を上げると ruff の target-version が上がり、新規 `UP` ルール（3.12 化で UP040/UP046 が PEP 695 `type X =` を強制したのと同型）が発火しうる。`ruff check --fix` の機械適用 + 残りの手当て。
- **Docker**: base image を `python:3.14-slim` へ。`make` で `requirements-pod.txt` 再生成。**初回ビルドは人間が確認**（この環境に docker が無い、follow-ups と同じ運用）。

## 6. テスト

| 種別 | 対象 | ゲート | フェーズ |
|---|---|---|---|
| 解決 | numpy 版 / onnxruntime cap | `uv lock` 通過・`uv sync --all-extras` で numpy 2.x・onnxruntime <1.27 | 1 |
| 健全性 | 全ゲート | `uv run --all-extras poe check` green | 1/2/3 |
| 回帰（数値） | `change_voice` 出力 vs seeded golden | 既存シード下で再現 | 1/2 |
| 実機（耳） | RVC ボイチェン + whisper | アーティファクト・誤出力なし | 1/2 |
| 混在 CUDA | ctranslate2(cu12) + torch/ort(cu13) 同居 | 実機で GPU 経路が動く | 2 |
| 構造 | pyworld 遅延 import | rmvpe 経路が pyworld 不在で import 可 | 3 |

数値 golden・実機テストは GPU / 資産を要するため環境変数で gating し、CI・通常の `poe check` では skip される。資産のある開発機で実行する（既存の `VSPEECH_RVC_GOLDEN_CONFIG` / `voicevox_e2e` と同じ流儀）。

## 7. 検証（フェーズ1）

- `uv lock` 通過、`uv tree` で numpy 2.x / onnxruntime-gpu <1.27 を確認。
- `uv run --all-extras poe check` green。
- seeded RVC golden（`VSPEECH_RVC_GOLDEN_CONFIG`、`--extra rvc`）再現。
- 実機耳チェック（最終確認、自動ゲート外）。
- `make` 後 `requirements-pod.txt` に numpy 2.x が反映されること（Docker イメージの初回ビルド確認はフェーズ3へ繰り延べ——フェーズ1は Windows 開発機で完結）。

## 8. リスクと緩和

| リスク | 影響 | 緩和 | フェーズ |
|---|---|---|---|
| lock 再生成が onnxruntime 1.27（CUDA 13）を引く | torch cu128 と CUDA 12/13 混在で GPU 経路が壊れる | `<1.27` cap を**フェーズ1で**入れる（1.2, 3.2） | 1 |
| numpy 2 で C 拡張が ABI 不整合 | rvc/whisper が実行時に壊れる | cap を上げれば numpy 2 ビルドが随伴（1.1）。import スモーク + seeded golden + 耳チェック（3.4） | 1 |
| numpy 2 の NEP 50 昇格で数値が変わる | 音質劣化・クリック | コードは既に明示クランプ済み（3.3）。seeded golden が即検知 | 1 |
| CUDA 13 ドライバ未達 | GPU ホストで起動不能 | フェーズ2前提として R580+ を明示・事前確認（4） | 2 |
| 混在 CUDA（ctranslate2 cu12 + torch/ort cu13） | whisper か RVC の一方が落ちる | R580+ が両対応。実機で耳チェック前に確認（4） | 2 |
| pyworld sdist ビルド強制 | 3.14 で rvc extra が解決しない | 遅延 import + extra 撤去で pyworld を必須依存から外す（5） | 3 |
| ruff target 昇格で新 lint 発火 | `poe check` が赤 | `ruff check --fix` 機械適用 + 手当て（3.12 化と同型、5） | 3 |
| Docker 初回ビルド未検証 | Linux イメージが起動しない | 人間が初回ビルド確認（follow-ups と同じ運用、5） | 3 |

## 9. 後続 spec への引き継ぎ

本 spec（フェーズ1）が凍結する成果物は **numpy 2 で検証済みの土台**（3.12 / cu128 / onnxruntime 1.26 のまま）。これがフェーズ2の入力になる。

- **フェーズ2（CUDA 13）**: numpy を固定したまま torch cu128→cu130・onnxruntime 1.26→1.27 を動かす。混在 CUDA と R580+ ドライバが焦点。
- **フェーズ3（Python 3.14）**: numpy 2・CUDA 13 の土台の上で版 churn のみ。audioop-lts・pyworld 遅延化・ruff/ty target・Docker が焦点。torch/onnxruntime の cp314 wheel は検証済み（1.1）のため機械的作業に近い。

各フェーズは実機検証が green になってから次へ進む。フェーズ2/3 はそれぞれ着手時に本 spec の素描（4/5）を出発点として独立に brainstorm・詳細化する。
