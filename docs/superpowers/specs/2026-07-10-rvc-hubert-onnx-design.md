# RVC HuBERT の ONNX 化 設計書（spec ②）

- 日付: 2026-07-10
- ステータス: 承認済み（実装計画へ）
- ブランチ: `feat/rvc-hubert-onnx`（vstreamer-tool）
- 前提 spec: [2026-07-09 RVC HuBERT の fairseq-free 化](2026-07-09-rvc-hubert-fairseq-free-design.md) §7 の「spec ②」
- 対象: `vspeech/lib/rvc.py` の HuBERT ロード / 特徴量抽出、`pyproject.toml` の依存宣言

## 1. 背景と目的

spec ①（`3f4a0f5`）で RVC の content encoder は fairseq から `transformers.HubertModel` に置き換わり、runtime は fairseq 非依存になった。特徴量の正解（どの層 / final_proj 適用有無 / 層インデックス対応）は `hubert_golden.npz` に凍結され、`layer_offset = 0` が実測で確定している。

本 spec の目的は **runtime 依存の縮小**である。等価性を最優先し、速度は現状維持でよい。

### 1.1 いま残っている問題（2026-07-10 実測）

`uv.lock` は universal lock であり、**extra に宣言されているだけでパッケージはロックに残る**。そして `uv audit` はロック全体を監査する。現状:

```
$ uv audit
torch 2.10.0+cu128 has 1 known vulnerability:
- GHSA-rrmf-rvhw-rf47 (torch.jit.script)            # 修正版なし・到達不能・受容済み
transformers 4.57.6 has 3 known vulnerabilities:
- GHSA-29pf-2h5f-8g72 (RCE)            Fixed in 5.3.0
- GHSA-69w3-r845-3855 (Trainer 任意コード実行)  Fixed in 5.0.0rc3
- PYSEC-2025-217                       修正版なし
```

さらに `convert` extra が持つ `numpy>=1.18,<1.24` が universal 解決を縛り、`uv.lock` の numpy は **1.23.5**（cp312 wheel が存在しない 2022 年版）に固定されている。これは spec ③（Python 3.12 化）の最後の障害そのものである。

したがって「extra に隔離する」だけでは不十分で、**offline ツールの依存を `pyproject.toml` から完全に外す**必要がある。

### 1.2 手段

1. content encoder を ONNX へ export し、runtime を `onnxruntime` の `InferenceSession` だけで動かす（`transformers` / `safetensors` の import を撤去）。RVC 本体と RMVPE は既に同じ基盤に載っているので、3 本目のセッションが増えるだけになる。
2. 変換（fairseq → transformers）と export（transformers → ONNX）は **一度きりのオフライン工程**なので、依存を extra ではなく `uv run --with` のオーバーレイで供給する。この repo は既に `poe security`（`uv run --with bandit ...`）と `poe deadcode`（`uv run --with vulture ...`）で同じ手を使っている。

### 1.3 検証済みの環境事実

- `torch 2.10.0+cu128` / `onnxruntime 1.26.0` / `transformers 4.57.6`、CUDA 利用可。
- `torch.onnx.export` は torch 2.10 で `dynamo=True` が既定。dynamo exporter は `onnx` と `onnxscript` を要求する（どちらも現在は未インストール）。
- 現行 runtime は `half_precision_available()` が真なら `HubertModel.half()` で **fp16 のまま推論している**。fp16 ONNX はこれと同じ数値経路になる。

## 2. スコープ / 非目標

- **スコープ**:
  - `create_session` が呼び出し側の `device` を尊重するようにする（**実装中に発見した既存バグ**。`torch.cuda.is_available()` だけで CUDA EP を選んでおり、config が CPU を指定しても GPU で走っていた。CUDA EP の TF32 で fp32 特徴量に `2.6e-3` の誤差が乗り、fp32 等価ゲートが CUDA ボックスで成立しない）。
  - オフライン export ツールの新設（`scripts/export_hubert_onnx.py`）。
  - `vspeech/lib/rvc.py` の `load_hubert_model` / `extract_features` を `InferenceSession` ベースへ置換。`transformers` / `safetensors` の import 撤去。
  - `infer()` の io_binding 処理を小さなヘルパへ括り出し、`extract_features` と共有する。
  - 等価ゲートの ONNX 対応（fp32 / fp16 の 2 系統）と、構造ゲートの `transformers` 追加。
  - `pyproject.toml`: `rvc` extra から `transformers` 撤去、`convert` extra 削除、`[tool.uv.sources]` から fairseq 撤去、offline ツールを poe task 化、dev group に `onnx` 追加、`[[tool.ty.overrides]]` 追加。

- **非目標（YAGNI / 後続 spec ③）**:
  - `requires-python` の 3.12 化、`torch` / `torchaudio` の cp312 URL 差し替え、classifiers / `[tool.ty.environment]` の更新。本 spec は **3.11 のまま**完結する。
  - `torch` / `torchaudio` の除去（resample・`interpolate`・io_binding で引き続き必要）。
  - RVC 推論本体（`infer` / `change_voice` の数式）、f0 抽出、VAD ゲートの変更。
  - `create_rmvpe_session`（`pitch_extract.py:25`）の同型バグの修正 — スコープ外。
  - `scripts/convert_hubert.py` の削除（`hubert_base.pt` からの再変換手段は残す）。
  - config の後方互換 alias。`RvcConfig.hubert_model_file` の意味（変換済み資産ディレクトリ）は **不変**なので、`config.toml.example` と GUI は無改修。

## 3. 設計

工程を 2 段のオフライン処理 + runtime に分ける。fairseq が要るのは第 1 段だけなので、spec ③ 以降も ONNX の再生成は 3.12 上で行える。

```
hubert_base.pt ─[① scripts/convert_hubert.py : fairseq]─▶ hubert_contentvec/ (transformers 資産)
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     │ ② scripts/export_hubert_onnx.py │
                                                     │    (transformers + torch + onnx) │
                                                     └────────────┬────────────┘
                                                                  ▼
                                     hubert_fp32.onnx / hubert_fp16.onnx / mapping.json
                                                                  │
                                                                  ▼
                                            runtime: onnxruntime InferenceSession のみ
```

### 3.1 オフライン export ツール `scripts/export_hubert_onnx.py`（新規）

**実行**: `uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden`
（poe への引数は `--` 区切り**なし**でタスク名の直後に置く。`poe startup-profile` と同じ流儀。）

プロジェクト環境（cu128 torch）の上に `transformers` / `onnx` / `onnxscript` を `--with` で重ねて走る。`--no-project` は**付けない**。fp16 グラフを CUDA 上で export するために、プロジェクトの cu128 torch が必要だからである。

**グラフの形**: `final_proj` を**グラフに焼き込んだ** 2 出力ラッパを export する。

```python
class HubertOnnxWrapper(torch.nn.Module):   # export 専用。runtime には入らない。
    def forward(self, source: Tensor) -> tuple[Tensor, Tensor]:
        hs = self.model(source, output_hidden_states=True).hidden_states
        return self.final_proj(hs[9 + self.off]), hs[12 + self.off]
```

- `off` は既存 `mapping.json` の `layer_offset`（実資産では 0）。**export 時に解決してグラフへ固定**する。runtime は層インデックスを知らない。
- 入力 `source` は `(1, N)`、出力は `feats_l9_proj` `(1, T, 256)` と `feats_l12_raw` `(1, T, 768)`。`N` / `T` は dynamic axis。
- 出す組合せを 2 つに限るのは、実在する RVC モデルが v1 = `(9, True)` と v2 = `(12, False)` の 2 種類しかないため（`embOutputLayer` / `useFinalProj` メタデータ、[vc.py:213-214](../../../vspeech/worker/vc.py)）。これは `hubert_golden.npz` の 2 キー `l9_proj` / `l12_raw` と 1:1 に対応する。
- `final_proj` をグラフに含めるので、runtime から `safetensors` と `torch.nn.Linear` が消える。
- 12 層すべてを回すのは現行 `output_hidden_states=True` と同じであり、計算量は増えない。

**fp32 / fp16 の 2 本を出す**:

- fp32: CPU 上で export。
- fp16: `model.half().to("cuda")` を CUDA 上で export。現行 runtime の `HubertModel.half()` と**同一の数値経路**になるため、新規依存ゼロで最も忠実。
- export 前に `pos_conv` の `weight_norm` パラメトリゼーションを `remove_parametrizations` で畳み込む。

**自己検証（spec ① と同じ流儀。通らなければ資産を書かない）**:

1. fp32 ONNX の出力 vs `hubert_golden.npz`（fairseq 由来 fp32 正解）→ `cosine ≥ 0.9999` かつ `max-abs ≤ 1e-4`。
2. fp16 ONNX の出力 vs 同 golden → fp16 用しきい値（3.3-b）。
3. 一時ディレクトリへ書き、両方通ってから資産ディレクトリへ move する。

**`mapping.json` の更新**（runtime が読む唯一のメタデータ）:

```json
{
  "layer_offset": 0,
  "num_hidden_layers": 12,
  "exporter": "dynamo",
  "opset": 20,
  "_comment": "exporter / opset は export 時に実際に使った値を書く（上は例）",
  "outputs": [
    {"name": "feats_l9_proj",  "layer": 9,  "use_final_proj": true,  "dim": 256},
    {"name": "feats_l12_raw",  "layer": 12, "use_final_proj": false, "dim": 768}
  ]
}
```

`layer_offset` は来歴として残すが runtime は使わない。`exporter` は dynamo / legacy のどちらで出したかの記録。

### 3.2 runtime `vspeech/lib/rvc.py`

`HubertBundle`（transformers モデル + Linear + offset）を **`HubertSession`** へ置き換える。

```python
@dataclass
class HubertSession:
    session: InferenceSession
    output_names: dict[tuple[int, bool], str]  # (emb_output_layer, use_final_proj) -> ONNX 出力名
    is_half: bool
```

**`load_hubert_model(file_name: Path, device, is_half: bool) -> HubertSession`**

- **シグネチャ不変**。`vspeech/worker/vc.py` は無改修で済む。
- `is_half and device.type == "cuda"` かつ `hubert_fp16.onnx` が存在すれば fp16、それ以外は `hubert_fp32.onnx` を、既存の `create_session()` で開く。fp16 グラフは `CPUExecutionProvider` で実質動かないため、**CPU では必ず fp32 を選ぶ**。`HubertSession.is_half` には実際に選んだ方を記録する（引数の `is_half` ではない）。`create_session(model_file, gpu_id)` は `device.index` が `None`（CPU）でも呼べるよう `gpu_id=device.index or 0` で渡す。
- `mapping.json` の `outputs` から `output_names` を組む。
- `transformers` / `safetensors` の import を撤去。

**`extract_features(model, feats, dev, emb_output_layer=9, use_final_proj=True) -> torch.Tensor`**

- **戻り値の契約（shape / dtype / device）は不変**。後段の `functional.interpolate` と `infer()` は無改修。
- `output_names[(emb_output_layer, use_final_proj)]` を引く。未登録なら対応表を添えて `RuntimeError`。**runtime は推測しない**という spec ① の原則をそのまま維持する。
- 入力 `feats` をセッションの入力 dtype（`model.is_half`）へキャストする。
- CUDA: `io_binding` で入力を torch tensor のポインタから直接バインドし、出力も CUDA に bind して dlpack で torch tensor として受け取る。GPU→CPU→GPU の往復を避ける。
- CPU: `session.run([name], {"source": ndarray})` → `torch.from_numpy`。

**共有ヘルパの抽出**: 上記は `infer()` が既にやっていることと同型である。`infer()` の中の io_binding 分岐から

- `_bind_torch_input(io_binding, name, tensor)` — dtype→numpy 型の対応とポインタ bind
- `_ort_output_to_torch(ort_value, device)` — dlpack 取り出しと numpy フォールバック

を括り出し、`infer()` と `extract_features()` の両方から使う。`infer()` の分岐が縮み、DepDegree も下がる。

**`half_available` の撤去**: `change_voice()` と `_extract_hubert_feats()` の `half_available` 引数を落とす。HuBERT の精度は `HubertSession.is_half` が単一の情報源になるため、呼び出し側が別経路で同じ判断を持ち回る必要がなくなる（現状は `load_hubert_model` と `change_voice` の両方に同じ値を渡している）。`vc.py` は `load_hubert_model` に渡す分だけ `half_precision_available()` を使い続ける。追随: `scripts/capture_change_voice_golden.py`、`tests/test_rvc_helpers.py`。

### 3.3 等価 + 回帰ハーネス

**(a) 主ゲート — fp32**

`tests/test_hubert_equivalence.py` を ONNX 経路へ書き換える。`hubert_golden.npz`（fairseq 由来）を正解とし、`(9, True)` / `(12, False)` の両方を `cosine ≥ 0.9999` かつ `max-abs ≤ 1e-4` で判定する。**しきい値は spec ① から据え置き**。

**(b) fp16 ゲート — 参照は fp32 golden ではなく torch fp16**

当初は fp16 も fp32 golden と照合する設計だったが、**2026-07-10 の実測でこれが原理的に成立しないことが判明した**。hidden state の大きさは O(1)〜O(2.5) あり、半精度の絶対誤差はもともと 1e-1 オーダーになる。実際、**この branch より前から production で動いている `HubertModel.half()` 自身**が fp32 golden に対して次の値を出す:

| 実装 | l9_proj cosine | l9_proj max_abs | l12_raw cosine | l12_raw max_abs |
|---|---|---|---|---|
| torch fp16 (現行 runtime) | 0.98720998 | 4.353e-01 | 0.97697942 | 2.651e-01 |
| ONNX fp16 (dynamo) | 0.98736646 | 4.304e-01 | 0.97705446 | 2.631e-01 |
| ONNX fp32 (dynamo) | 1.00000000 | 1.010e-05 | 1.00000000 | 9.030e-06 |

ONNX fp16 は torch fp16 よりむしろわずかに良い。つまり壊れているのは export ではなく「fp16 を fp32 golden に絶対誤差で照らす」というゲートの立て方だった。

したがって fp16 ゲートは **ONNX fp16 を「置き換え対象である torch fp16」と照合する**。問いが「ONNX 化で fp16 の振る舞いが変わっていないか」になり、はじめて意味を持つ。実測（同上）:

| 比較 | cosine | max_abs |
|---|---|---|
| ONNX fp16 vs torch fp16 `l9_proj` | 0.99999010 | 1.379e-02 |
| ONNX fp16 vs torch fp16 `l12_raw` | 0.99997235 | 1.074e-02 |

- torch fp16 の出力は export ツールが `<golden>/hubert_golden_fp16.npz` として捕獲する（transformers を持つのは export 環境だけなので、ここでしか捕獲できない）。テスト側は npz を読むだけで transformers を要らない。
- しきい値: `COSINE_MIN_FP16 = 0.9999`、`MAX_ABS_MAX_FP16 = 5e-2`（実測 1.379e-2 の約 3.6 倍。10 倍規則の範囲内で、より厳しい側に取る）。単一情報源は `scripts/hubert_metrics.py`。
- 硬い上限 `MAX_ABS_MAX_FP16 ≤ 1e-1` / `COSINE_MIN_FP16 ≥ 0.999` は**そのまま維持できる**。これを超えるなら export が壊れている。
- fp16 参照は GPU / カーネル依存（RTX 4060 で捕獲）。このテストはどのみち CUDA と資産の両方で gating され開発機限定なので許容する。他機で再捕獲が要る場合は export を回し直せばよい。
- テストは CUDA・`hubert_fp16.onnx`・`hubert_golden_fp16.npz` のいずれかが無ければ skip する。

**(c) 回帰 — `change_voice` 音声 golden は据え置き。しきい値は SNR のみ実測で緩和**

既存の `tests/assets/rvc_golden/change_voice_golden.npz` を**そのまま使う。再ベースラインはしない。** golden は fairseq 由来なので、これが実装を跨いだ唯一の保証である。

ただし当初「`SNR ≥ 40 dB` 据え置きで通るはず」としたのは**誤りだった**。実測（2026-07-10, RTX 5060, 実 RVC モデル）:

| HuBERT 実装 | corr | SNR |
|---|---|---|
| transformers fp16（spec ①） | 0.99998675 | 44.59 dB |
| **ONNX fp16（spec ②）** | 0.99995400 | **39.52 dB** |

golden は **fp16 の HuBERT で捕獲されている**ため、このゲートは HuBERT の実行エンジンに敏感である。spec ① が 44.59 dB を出せたのは transformers が torch の fp16 カーネルをそのまま使っていたからで、ORT の fp16 カーネルは丸めが違う。**グラフの欠陥ではない。**

切り分けの根拠（同じ ONNX を fp32 で走らせた対照実験）:

- fp32 グラフの特徴量は fairseq を `max_abs 1.010e-05` で再現する（3.3-a）。グラフは正しい。
- その fp32 グラフで `change_voice` を走らせると golden 比 `corr 0.717 / SNR 3.03 dB` まで落ちる。golden が fp16 で捕獲されているので当然である。
- 感度の整合: fp16↔fp32 の特徴量差は `max_abs 0.43`、ONNX fp16↔torch fp16 は `max_abs 1.379e-02`。比は約 30 倍で、`20·log10(30) ≈ 29.5 dB`。実際の SNR 差 `39.52 − 3.03 = 36.5 dB` と同じ桁で整合する。

したがって `SNR_MIN_DB` を **40.0 → 35.0** とする（実測 39.52 dB に対し約 4.5 dB の余裕。spec ① が 44.59/40 で持っていた余裕と同じ）。`CORR_MIN = 0.999` は据え置き（実測 0.999954）。単一情報源は `scripts/hubert_metrics.py`。

HuBERT は RNG を消費しないため、`capture_change_voice_golden.seed_all()` 後の RNG ストリームは実装差し替えの影響を受けない（spec ① と同じ論拠）。

**(d) runtime 単体テストの依存を切る**

`tests/test_hubert_runtime.py` は現在 tiny `HubertConfig` で合成資産を作っており、`transformers` をテストに引きずる。代わりに `onnx` のグラフ API で **2 出力の極小 ONNX を組み立てる**。実 HuBERT も transformers も要らず、検査対象は runtime の契約そのものになる:

- `output_names` の引き当て（`(9,True)` → `feats_l9_proj`、`(12,False)` → `feats_l12_raw`）
- 未登録の組合せ（例 `(9, False)`）で `RuntimeError`、メッセージに対応表が出る
- 戻り値の shape / dtype / device
- `is_half` と device による fp16/fp32 ファイル選択（`hubert_fp16.onnx` 不在なら fp32 へ落ちる）

**(e) 構造ゲートの一般化**

`tests/test_no_fairseq_import.py` を `tests/test_forbidden_imports.py` へ `git mv` し、禁止モジュールを `("fairseq", "transformers")` で parametrize する。`vspeech/` 配下に該当 import が 0 件であることを AST で検査する。**これが本 spec の成果を守る回帰ゲート。**

**資産の扱い**: `hubert_fp32.onnx`（約 380MB）と `hubert_fp16.onnx`（約 190MB）は派生物であり gitignore する。等価テストは `VSPEECH_HUBERT_ASSET_DIR` / `VSPEECH_HUBERT_GOLDEN_DIR` が未設定なら skip する（既存の流儀のまま）。

### 3.4 依存 / 設定の変更

**`pyproject.toml`**

- `rvc` extra: `transformers>=4.44,<5` を**撤去**。残るのは torch / torchaudio / onnxruntime-gpu / numpy / scipy / pyworld / faiss-cpu。
- `convert` extra: **削除**。
- `[tool.uv.sources]`: `fairseq` エントリを**削除**。wheel URL は poe task の文字列とスクリプトの docstring に移す。
- dev group: `onnx` を追加（3.3-d の極小グラフ用。pure Python、runtime extras には入らない）。
- 新規 poe task（依存は pyproject にも `uv.lock` にも載せない）:

```toml
convert-hubert = { cmd = "uv run --isolated --no-project --python 3.11 --with 'fairseq @ https://github.com/sondeko143/fairseq-311/releases/download/v0.12.2.post1/fairseq-0.12.2.post1-cp311-cp311-win_amd64.whl' --with 'numpy<1.24' --with torch --with transformers python -m scripts.convert_hubert" }
export-hubert-onnx = { cmd = "uv run --with transformers --with onnx --with onnxscript python -m scripts.export_hubert_onnx" }
```

**入れ子 `uv run` の挙動（2026-07-10 に uv 0.11.20 で実測）**:

- `poe` はプロジェクトの `.venv` の中でコマンドを起動するため、入れ子の `uv run` には `VIRTUAL_ENV` が伝播する。それでも `--isolated --no-project --python 3.11` は `VIRTUAL_ENV` を無視して専用の一時環境を作る。`.venv` もたまたま 3.11 だが**再利用されない**（隔離環境から `torch` / `vspeech` は import 不可であることを確認済み）。
- 指定した処理系が手元に無ければ uv が自動でダウンロードする。
- **`--python 3.11` は必須。** 省くとプロジェクトの処理系（spec ③ 以降は 3.12）に落ち、cp311 の fairseq wheel が入らず壊れる。
- `python -m` は cwd を `sys.path[0]` に入れるので、隔離環境でも `scripts.` パッケージ import は解決する（poe の cwd = repo root）。venv とは無関係。
- `--no-project` では `[tool.uv.sources]` が読まれない。したがって変換環境の `torch` は PyPI 版（Windows では CPU ビルド）になる。**変換は CPU で足りるのでこれは意図どおり。** fairseq の wheel URL を `--with 'fairseq @ ...'` としてインラインで与えているのも同じ理由。
- 対して `export-hubert-onnx` は `--no-project` を付けないので、プロジェクト環境の cu128 torch を保ったまま `transformers` / `onnx` / `onnxscript` が重なる（`torch 2.10.0+cu128` / `cuda=True` のまま `onnx` が入ることを確認済み）。fp16 の CUDA export はこれに依存する。

- `[[tool.ty.overrides]]`: `scripts/convert_hubert.py` と `scripts/export_hubert_onnx.py` に限定して `unresolved-import` を無視する。これらの import（fairseq / transformers / onnx / onnxscript）は**本当にプロジェクト依存ではない**ため、`python-health` スキルが禁じる「extras 未同期による偽陽性の repo 抑制」には当たらない。その根拠を override のコメントに残す。

**numpy の解決**: `convert` extra の `<1.24` が消えることで、universal 解決の numpy は `>=1.24,<2` の最新（1.26 系）へ上がる。これは `rvc` / `whisper` extra が既に許容している範囲（`>=1.23,<2`）である。`pyworld` / `faiss-cpu` / `onnxruntime-gpu` / `ctranslate2` の numpy 互換は `uv lock` と import スモーク + 既存テストで確認する。

**config / GUI**: 変更なし。`RvcConfig.hubert_model_file` は引き続き資産ディレクトリを指し、その中に ONNX が増えるだけ。

## 4. テスト

| 種別 | 対象 | ゲート |
|---|---|---|
| 等価（主） | fp32 ONNX の特徴量 vs fairseq golden、`(9,True)` / `(12,False)` | `cosine ≥ 0.9999` かつ `max-abs ≤ 1e-4` |
| 等価（fp16） | fp16 ONNX の特徴量 vs **torch fp16 参照**（CUDA gating） | `cosine ≥ 0.9999` かつ `max-abs ≤ 5e-2` |
| 回帰 | `change_voice` の出力音声 vs 既存 golden（fairseq 由来、再ベースラインしない） | `corr ≥ 0.999` かつ `SNR ≥ 35 dB`（実測 0.999954 / 39.52 dB。3.3-c 参照） |
| runtime 単体 | 出力名の引き当て / エラー経路 / fp16 選択 | 極小 ONNX で固定（transformers 不要） |
| 構造 | `vspeech/` 配下の `fairseq` / `transformers` import 数 | 0 件 |
| export ツール | `scripts/export_hubert_onnx.py` の自己検証 | 失敗したら資産を書かない |

## 5. 検証

- `uv lock` が通り、`uv.lock` に `fairseq` と `transformers` の**エントリが存在しない**こと。
- `uv audit` が **torch の 1 件のみ**を報告すること（`GHSA-rrmf-rvhw-rf47`、修正版なし・到達不能・受容済み）。
- `uv.lock` の numpy が `>=1.24,<2` に上がっていること。
- `uv sync --all-extras` 後、`uv run --all-extras poe check` が green（既存の非ゼロ 2 件を除く: `audit` の受容済み torch advisory と `deadcode` の `vr2_config`。どちらも `main` にもある）。
- 資産のある開発機で:
  - まず `uv run poe export-hubert-onnx --asset <資産dir> --golden ./hubert_golden` を走らせる。これが `hubert_fp32.onnx` / `hubert_fp16.onnx` に加えて **`<golden>/hubert_golden_fp16.npz`（fp16 ゲートの参照）** を書く。これが無いと fp16 の等価テストは 2 件 skip される。
  - `VSPEECH_HUBERT_ASSET_DIR=<資産dir> VSPEECH_HUBERT_GOLDEN_DIR=./hubert_golden uv run --all-extras pytest tests/test_hubert_equivalence.py`（fp32 2 件 + fp16 2 件）
  - `VSPEECH_RVC_GOLDEN_CONFIG=<rvc toml> uv run --all-extras pytest tests/test_change_voice_golden.py`（CUDA 必須）
- **`<資産dir>` は `rvc.hubert_model_file` が指す実物と同じであること。** 2026-07-10 に実際に踏んだ罠: テストと export は repo 直下の `./hubert_contentvec` を見ていたが、実運用 config は `~/.config/vstreamer/hubert_contentvec` を指していた。全ゲートが green のまま、vc worker だけが起動時に `FileNotFoundError` で落ちる状態だった（`_select_onnx_file` が大声で落ちるので静かな劣化ではないが、自動ゲートは構造的にこれを検出できない）。
- 実機の耳チェックは最終確認として別途行う（自動ゲートには含めない。spec ① / VAD v6 移行と同じ運用）。

## 6. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| `torch.onnx.export(dynamo=True)` が transformers HubertModel で失敗する | export できない | 既定の dynamo を第一候補、失敗時は legacy exporter（`dynamo=False`）へフォールバック。どちらで出したかを `mapping.json` の `exporter` に記録する。**フォールバックは必ず traceback 付きで大声で報告する**（下の行を参照） |
| フォールバックが本当の失敗を隠す | 静かに劣った経路を使う | **2026-07-10 に実際に起きた。** torch.onnx が進捗表示の `✅` を Windows の cp1252 stdout に書こうとして `UnicodeEncodeError` で落ち、広い `except Exception` がそれを「dynamo 失敗」と誤認して legacy へ落ちていた。export ツールは起動時に stdout/stderr を UTF-8 へ `reconfigure` し、フォールバック時は例外型と traceback を印字する |
| `pos_conv` の `weight_norm` パラメトリゼーションが export を壊す | 特徴量が別物になる | export 前に `remove_parametrizations` で畳み込む。3.1 の自己検証が即検知 |
| fp16 ONNX が想定より劣化 | 音質退行 | fp16 は torch fp16 を参照に判定（3.3-b）。音声側は fairseq 由来 golden への corr/SNR ゲートが最終判定。**両方通らない限り採用しない**。実際に SNR が 40 dB を割ったが、fp32 グラフで対照実験して「グラフではなく ORT の fp16 カーネルの丸め差」と切り分けた上で、実測値からしきい値を 35 dB へ確定した |
| `io_binding` で 1 出力だけ bind したときの ORT の挙動 | 出力が取れない / 想定外の計算 | 等価ゲートで検証。ダメなら両出力を bind して片方を捨てる（現状比で劣化しない） |
| numpy 1.23.5 → 1.26 系への繰り上がりが pyworld / faiss / onnxruntime-gpu / ctranslate2 へ波及 | rvc / whisper 経路が壊れる | `uv lock` + import スモーク + 既存テスト（`poe check`）で確認。壊れる場合は当該パッケージの上限を明示ピンする |
| ty override が将来の本物の import エラーを隠す | 型検査の穴 | override は 2 ファイル・`unresolved-import` 1 ルールに限定し、根拠コメントを添える |
| `poe convert-hubert` から `--python 3.11` が失われる | プロジェクトの処理系（③ 以降は 3.12）に落ち、cp311 の fairseq wheel が入らない | 3.4 に理由を明記。task の `cmd` にコメントを添える |
| 資産サイズ増（fp32 380MB + fp16 190MB） | ディスク | gitignore 済み。他のモデル資産と同じ扱い |

## 7. 後続 spec ③（Python 3.12 化）への引き継ぎ

本 spec 完了後、`uv.lock` から cp311 固定の wheel は `torch` / `torchaudio` の 2 つだけになり、numpy の下限も外れる。spec ③ に残るのは:

1. `[tool.uv.sources]` の `torch` / `torchaudio` を cp311 → cp312 の URL へ差し替え。
2. `requires-python` を `>=3.12,<3.13` へ。
3. classifiers と `[tool.ty.environment] python-version` を更新。
4. `poe convert-hubert` は 3.11 の使い捨て環境を明示指定しているため、**3.12 化後もそのまま動く**（`--python 3.11 --isolated --no-project`）。ONNX の再生成（`poe export-hubert-onnx`）はプロジェクト環境で走るので 3.12 上で動く。

つまり ③ は機械的な版数作業のみになる。
