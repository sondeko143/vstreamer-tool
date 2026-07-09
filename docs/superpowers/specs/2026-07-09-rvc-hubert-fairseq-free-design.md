# RVC HuBERT の fairseq-free 化 設計書

- 日付: 2026-07-09
- ステータス: 承認済み（実装計画へ）
- ブランチ: `feat/rvc-hubert-fairseq-free`（vstreamer-tool）
- 対象: `vspeech/lib/rvc.py` の HuBERT ロード / 特徴量抽出（[rvc.py:7-12](../../../vspeech/lib/rvc.py), [rvc.py:74-94](../../../vspeech/lib/rvc.py), [rvc.py:189-209](../../../vspeech/lib/rvc.py)）

## 1. 背景と目的

`rvc` extra は `fairseq` に依存しており、これが **Python バージョン引き上げの唯一にして最大の障害** になっている。

**上流の状況（2026-07-09 実データ確認）:**

- 公式 `facebookresearch/fairseq` の最終リリースは **0.12.2（2022-06-27）** で以降ゼロ。PyPI の classifier は Python 3.6/3.7/3.8 止まり。
- リポジトリは **2026-03-20 に archived（read-only）**。Python 3.11 対応の追跡 Issue [#5191](https://github.com/facebookresearch/fairseq/issues/5191) は `needs triage` のまま未対応でアーカイブ入りし、3.12 の [#5634](https://github.com/facebookresearch/fairseq/issues/5634) も同様。**公式修正は永久に来ない。**
- 3.11 以降では import 時点で `ValueError: mutable default ... use default_factory`（dataclass の可変デフォルト規制）で落ちる。加えて `omegaconf<2.1` / hydra-core の依存衝突がある。
- 後継の `fairseq2` は API 非互換の全面リライトで、本 repo が使う `checkpoint_utils` / `fairseq.models.hubert.HubertModel` とは互換がない。**ドロップイン代替にならない。**

このため本 repo は自前ビルドの `sondeko143/fairseq-311`（`cp311-cp311-win_amd64` wheel、[pyproject.toml:95](../../../pyproject.toml)）をピンしている。この cp311 固定 wheel が、`torch`/`torchaudio` の cp311 URL ピンと `numpy<1.24`（1.23.5 は cp312 wheel が存在しない）を連鎖的に縛っている。**3 つのピンはすべて fairseq 由来**であり、fairseq を外せば結び目ごと解ける。

**目的**: RVC の content encoder を fairseq から `transformers.HubertModel` に置換し、**runtime を fairseq 非依存**にする。置換の正しさは **HuBERT 特徴量の数値等価**で機械的に担保する。

### 1.1 fairseq の実使用範囲（調査結果）

本 repo における fairseq の footprint は極めて狭く、`vspeech/lib/rvc.py` の 2 箇所に閉じている:

1. `load_hubert_model()`（[rvc.py:189-209](../../../vspeech/lib/rvc.py)）— `checkpoint_utils.load_model_ensemble_and_task()` で ContentVec の `.pt` を読む。
2. `extract_features()`（[rvc.py:74-94](../../../vspeech/lib/rvc.py)）— `model.extract_features(source, padding_mask, output_layer=N)` を呼び、必要なら `model.final_proj(...)` を適用。

`emb_output_layer` と `use_final_proj` は **RVC モデル自身の ONNX メタデータ**（`embOutputLayer` / `useFinalProj`、既定 `9` / `True`）から供給される（[vc.py:213-214](../../../vspeech/worker/vc.py), [vc.py:273-274](../../../vspeech/worker/vc.py)）。したがって特徴量仕様は固定ではなく**ロードするモデルごとに決まる**。代替実装は任意の層 + optional final_proj の**両対応が必須**。

### 1.2 なぜ transformers か

RVC-Project 本家も「fairseq は衝突が多すぎる」として transformers への置換を追跡している（Issue [#2264](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2264)、**未マージ**）。`transformers` は純 Python + torch で 3.12/3.13 の wheel があり、可変長入力をネイティブに扱え、eager 実行なので層ごとの数値検証が容易。ONNX 化（後続 spec）に進む前段として、まず「正しい特徴量とは何か」を eager で凍結するのに適する。

### 1.3 重要な前提: final_proj は既存 `.pt` の中にある

RVC の `hubert_base.pt`（ContentVec）は**全 RVC モデルで共有される単一ファイル**であり、`final_proj.weight` / `final_proj.bias`（768→256 の Linear）も**同じ state_dict に含まれる**。各 RVC モデルは「どの層を使うか」「その共有 final_proj を当てるか」をメタデータで宣言しているにすぎない。

したがって final_proj は**手元の `hubert_base.pt` から抽出**でき、新規アセットの入手は不要で、**fairseq が使っていた重みとの厳密一致が保証される**。

## 2. スコープ / 非目標

- **スコープ**:
  - オフライン変換ツールの新設（`hubert_base.pt` → transformers 形式 + final_proj 抽出）。
  - `vspeech/lib/rvc.py` の `load_hubert_model` / `extract_features` の置換と fairseq import の撤去。
  - HuBERT 特徴量の数値等価ゲート（新規テスト）と、既存 change_voice 音声 golden の許容誤差化。
  - `pyproject.toml` の `rvc` extra から fairseq を撤去し `transformers` を追加。fairseq は変換専用 extra へ隔離。
  - `numpy` 上限の緩和（transformers を解決するために必要な範囲）。
  - `RvcConfig.hubert_model_file` の意味変更（ファイル → 変換済み資産ディレクトリ）と `config.toml.example` / GUI の追随。

- **非目標（YAGNI / 後続 spec）**:
  - **ONNX 化**（content encoder を `onnxruntime` で走らせる）— 後続 spec ②。本 spec で凍結した特徴量定義を基準に export する。
  - **`requires-python` の 3.12 化**、`torch` / `torchaudio` の cp312 URL 差し替え、classifiers / `[tool.ty.environment]` の更新 — 後続 spec ③。本 spec は **3.11 のまま**完結する。
  - `torch` / `torchaudio` の除去（リサンプリング・`interpolate`・io_binding で引き続き必要。かつ cp312 wheel があるため障害ではない）。
  - RVC 推論本体（`infer` / `change_voice`）、f0 抽出、VAD ゲートの変更。
  - `index_rate` / faiss index 周りの変更。
  - config の後方互換 alias — **クリーン入れ替え**（config は個人利用・gitignore のため破壊的変更可）。

## 3. 設計

責務を 3 コンポーネントに分離する。

### 3.1 オフライン変換ツール `scripts/convert_hubert.py`

**実行環境**: Python 3.11 + fairseq（後述の `convert` extra）。**一度だけ**実行するオフライン処理で、runtime パスには含まれない。

**入出力**:

```
uv run --extra convert python scripts/convert_hubert.py \
    --input  <path>/hubert_base.pt \
    --output <path>/hubert_contentvec/     # 変換済み資産ディレクトリ
    --golden <path>/hubert_golden/         # 等価テスト用の golden 捕獲
```

**処理**:

1. fairseq の `checkpoint_utils.load_model_ensemble_and_task()` で `hubert_base.pt` を読む（現行 [rvc.py:189-196](../../../vspeech/lib/rvc.py) と同一手順）。
2. ContentVec に対応する `transformers.HubertConfig` を構築する。ContentVec は HuBERT-base 系であり `do_stable_layer_norm=False` / `feat_extract_norm="group"` / 12 層 / hidden 768。値は fairseq 側モデルの実属性から読み取って設定し、ハードコードしない。
3. encoder の state_dict を fairseq 命名 → transformers 命名へマッピングし、`model.safetensors` + `config.json` として `--output` に保存。
4. `final_proj.weight` / `final_proj.bias` を抽出し、`final_proj.safetensors` として同ディレクトリに保存。
5. **層インデックス対応の確定**: fairseq の `extract_features(output_layer=N)` と transformers の `output_hidden_states=True` が返す `hidden_states[i]` の対応（off-by-one の危険がある）を、実際に両方を走らせて誤差最小となる `i` を同定し、`config.json` と並置する `mapping.json` に**明示的に記録**する。runtime はこの記録を読むだけで推測しない。
6. **自己検証**: 固定の代表音声を両実装に通し、`(emb_output_layer=9, use_final_proj=True)` と `(12, False)` の両方で特徴量が許容誤差内であることをその場でアサートする。失敗したら変換を成功扱いにしない。
7. **golden 捕獲**: 上記の fairseq 側出力（fp32）を `--golden` に保存する。これが 3.3 のテストの正解データになる。

**なぜオフラインか**: 変換に必要な fairseq を runtime から完全に隔離できる。ネットワーク取得（HF hub）にも依存しない。何より**手元の実ファイルと bit 一致した重み**を使えるため、3.3 の等価ゲートが通りやすい。

**実装順序の制約**: golden 捕獲（手順 7）は fairseq が動く環境でしか行えない。したがって実装は必ず **「変換ツール + golden 捕獲」→「runtime 置換」→「等価ゲートで検証」** の順に進める。逆順（先に fairseq を撤去）にすると正解データを失う。なお fairseq は `convert` extra に残るため、後からでも再捕獲は可能。

### 3.2 runtime 置換 `vspeech/lib/rvc.py`

**外部契約は不変**に保つ。`extract_features` の戻り値（shape / dtype / device）が現状と同じであれば、後段の `functional.interpolate`（[rvc.py:262-263](../../../vspeech/lib/rvc.py)）と `infer()` の io_binding（[rvc.py:97-186](../../../vspeech/lib/rvc.py)）は**無改修**で済む。

**新しい束ね型**:

```python
@dataclass
class HubertBundle:
    model: transformers.HubertModel
    final_proj: torch.nn.Linear | None   # useFinalProj=False のモデルしか使わない場合は None 可
    layer_offset: int                    # mapping.json から読む（3.1-5）
```

**`load_hubert_model(file_name: Path, device, is_half) -> HubertBundle`**

- `file_name` は**変換済み資産ディレクトリ**を指す（意味変更、3.4 参照）。
- `HubertModel.from_pretrained(file_name)` で encoder を読み、`final_proj.safetensors` から `nn.Linear` を復元、`mapping.json` から `layer_offset` を読む。
- `.eval()` / `.to(device)` / `is_half` なら `.half()` — 現行と同じ。`final_proj` も同じ device / dtype に揃える。
- `torch.compile(..., dynamic=True, mode="max-autotune")` の try/except は**現行のまま維持**（[rvc.py:204-207](../../../vspeech/lib/rvc.py)）。compile 対象は `HubertBundle` ではなく**内部の `HubertModel` インスタンス**（`bundle.model`）。`final_proj` は単層 Linear のため compile しない。
- `torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])` は不要になるため削除。

**`extract_features(model: HubertBundle, feats, dev, emb_output_layer=9, use_final_proj=True) -> torch.Tensor`**

- `model.model(input_values=feats.to(dev), output_hidden_states=True)` を `torch.inference_mode()` 下で実行。
- `hidden_states[emb_output_layer + model.layer_offset]` を取得。
- `use_final_proj` が真なら `model.final_proj(...)` を適用（偽ならそのまま）。
- **padding_mask の扱い**: 現行は全ゼロ（= パディング無し）の `padding_mask` を渡している（[rvc.py:81](../../../vspeech/lib/rvc.py)）。transformers の `attention_mask` は**意味が反転**している（1 = 有効、fairseq の padding_mask は True = パディング）。パディング無しなので `attention_mask` は渡さない（`None` = 全有効）。これは等価。
- fairseq import（[rvc.py:7-12](../../../vspeech/lib/rvc.py)）を撤去。

`use_final_proj=True` かつ `final_proj is None` の場合は、資産が不完全なので `load_hubert_model` 時点ではなく **`extract_features` 到達時に明示的に例外**を投げる（メタデータはモデル依存なので、ロード時には必要性が判断できない）。

### 3.3 等価 + 回帰ハーネス

**(a) 主ゲート — HuBERT 特徴量の数値等価（新規 `tests/test_hubert_equivalence.py`）**

- 3.1-7 で捕獲した fairseq 側 golden 特徴量（fp32）を読み、新実装の `extract_features` 出力と比較する。
- 対象は `(9, True)`（v1 / 256 次元）と `(12, False)`（v2 / 768 次元）の**両方**。
- 判定（fp32、初期値）: `cosine ≥ 0.9999` **かつ** `max-abs-diff ≤ 1e-4`。
- しきい値の確定規則: 3.1-6 の自己検証で実測した誤差の **10 倍**を上限として採用する。実測が上記初期値より大きい場合のみ緩め、その根拠（実測値）をテストのコメントに残す。**理由なく緩めない。**

**(b) 回帰 — change_voice 音声 golden の許容誤差化**

- 既存の seeded bit-exact golden（`scripts/capture_change_voice_golden.py` 由来）は、HuBERT 実装が変わるため **bit-exact では通らなくなる**。これは仕様であって退行ではない（従来の bit-golden は「数式を変えないリファクタ」の検証用だった）。
- 判定を **許容誤差ベース**へ変更する: 正規化相互相関 `≥ 0.999` **かつ** セグメンタル SNR `≥ 40 dB`。しきい値は (a) と同じ確定規則に従う。
- golden 音声は新実装で**再ベースライン**する。

**(c) fairseq 不在ゲート**

- `vspeech/` 配下に `fairseq` の import が 1 件も無いことを検査する軽量テストを追加する（`ast` で import 文を走査、または `poe check` に grep ステップを追加）。**これが本 spec の成果を守る回帰ゲート。**

**資産の扱い**: 変換済み encoder（約 95MB）と golden 捕獲は派生物であり、他のモデル資産と同様に gitignore する。テストは環境変数 `VSPEECH_HUBERT_ASSET_DIR`（変換済み資産）と `VSPEECH_HUBERT_GOLDEN_DIR`（golden）が未設定なら **skip** する。これは既存の `VSPEECH_RVC_GOLDEN_CONFIG` / `voicevox_e2e` マーカーと同じ流儀。

### 3.4 依存 / 設定の変更

**`pyproject.toml`**

- `rvc` extra: `fairseq` を**撤去**、`transformers>=4.44,<5` を追加。
- 新設 `convert` extra（3.11 限定、変換ツール専用）: `fairseq`、`torch`、`torchaudio`、`numpy>=1.18,<1.24`。`[tool.uv.sources]` の fairseq URL エントリはここへ残す（再変換可能性を保つ）。
- `numpy`: `rvc` / `whisper` extra の `>=1.18,<1.24` を **`>=1.23,<2`** に緩和する。上限 `<1.24` は fairseq 0.12.2 の旧 numpy API 都合だったため撤廃できる。`<2` に留めるのは `onnxruntime-gpu` / `faiss-cpu` / `pyworld` の numpy 2.x 対応が未確認のため（`<3` への引き上げは後続 spec で扱う）。
- **`convert` と `rvc` の numpy 制約の関係（重要）**: fairseq 0.12.2 は `numpy>=1.24` で撤去された旧エイリアスを使うため、`convert` extra は `<1.24` を自前で保持する必要がある。uv は 1 環境につき numpy を 1 版しか解決しないので、**`convert` は `rvc` とは別の使い捨て 3.11 環境で単独インストールして使う**（`uv run --extra convert`）。両方を同時に入れた場合は numpy 1.23.5 に落ちるだけで壊れはしない（`transformers` は `numpy>=1.24` を要求しない）が、**runtime 環境に `convert` を含めない**ことを運用ルールとする。
- 判定: `uv lock` が通り、`uv sync --extra rvc` 後に rvc 経路が import できること（このとき numpy は `>=1.24` に解決され、fairseq は不在）。

**`vspeech/config.py`**

- `RvcConfig.hubert_model_file`（[config.py:335](../../../vspeech/config.py)）: 型は `Path` のまま、**意味を「`.pt` ファイル」→「変換済み資産ディレクトリ」へ変更**。`Field(description=...)` を付けて明示する。
- フィールド名は据え置く（`hubert_model_dir` へのリネームは config / GUI / golden スクリプトに波及するだけで機能価値が無い = YAGNI）。

**追随が必要な箇所**

- `config.toml.example` の `[rvc]` セクション。
- GUI のラベル（[gui.py:780](../../../gui/gui.py)）。
- `scripts/capture_change_voice_golden.py`（`load_hubert_model` の呼び出し）。

## 4. テスト

| 種別 | 対象 | ゲート |
|---|---|---|
| 等価（主） | `extract_features` の特徴量 vs fairseq golden、`(9,True)` / `(12,False)` | cosine ≥ 0.9999 かつ max-abs ≤ 1e-4（fp32） |
| 回帰 | `change_voice` の出力音声 vs 再ベースライン golden | 相関 ≥ 0.999 かつ SNR ≥ 40 dB |
| 構造 | `vspeech/` 配下の fairseq import 数 | 0 件 |
| 変換ツール | `scripts/convert_hubert.py` の自己検証（3.1-6） | 変換時に自らアサート |

等価テスト / 回帰テストは資産（変換済み encoder・golden）を要するため環境変数で gating し、未設定時は skip する。CI・通常の `uv run poe check` では skip される想定で、資産のある開発機で実行する。

## 5. 検証

- `uv run pytest tests/test_hubert_equivalence.py`（資産あり環境）が green。
- `uv run poe check`（ruff format/lint、ty、pytest、lock-check、audit、security、deadcode）が green。
- `uv sync --extra rvc` に fairseq が含まれないこと（`uv tree` で確認）。
- `vspeech/` に fairseq import が無いこと（3.3-c のテスト）。
- 実機の耳チェックは最終確認として別途行う（自動ゲートには含めない。VAD v6 移行と同じ運用）。

## 6. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| encoder 重みマッピングの取り違え | 特徴量が別物になり音質が壊れる | 3.1-6 の変換時自己検証 + 3.3-a の等価ゲートが即検知。**両方通らない限り採用しない。** |
| `output_layer` ↔ `hidden_states` の off-by-one | 隣接層の特徴量を使ってしまい微妙に劣化 | 3.1-5 で実測により同定し `mapping.json` に固定。runtime は推測しない。 |
| layernorm / feat_extract 設定の不一致 | 誤差が許容外 | config を fairseq モデルの実属性から読み取り、ハードコードしない（3.1-2）。 |
| `transformers` の版依存 | 将来の API 変更で壊れる | `>=4.44,<5` のレンジピン。 |
| numpy 緩和が pyworld / faiss / onnxruntime へ波及 | rvc extra が解決しない / 実行時に壊れる | `<2` に留める。`uv lock` + import スモークで確認（3.4）。 |
| fp16 経路での誤差増大 | 半精度で等価ゲートを満たせない | 等価ゲートは **fp32 で判定**する。fp16 は既存の音声回帰（3.3-b）でカバーする。 |

## 7. 後続 spec への引き継ぎ

本 spec が凍結する成果物は次の 2 つで、これが後続作業の入力になる:

1. **特徴量の正解定義**（どの層 / final_proj 適用有無 / 層インデックス対応）と、その golden 捕獲。
2. **fairseq 非依存の runtime**。

これにより後続は次の 2 本に分解される:

- **spec ②（ONNX 化）**: 本 spec の golden をそのまま基準にして content encoder を ONNX へ export し、既存の `onnxruntime-gpu` セッション基盤へ載せ替える。export の正しさを同じ等価ゲートで即検証できる。
- **spec ③（Python 引き上げ）**: `requires-python` を 3.12 系へ、`torch` / `torchaudio` を cp312 URL へ、classifiers と `[tool.ty.environment]` を更新。fairseq が消えているため機械的作業になる。
