# `change_voice` 純粋ヘルパ分解リファクタ 設計書

- 日付: 2026-06-25
- ステータス: 承認済み（実装計画へ）
- ブランチ: `refactor/change-voice-helpers`（vstreamer-tool）
- 関連: code-metrics スキルの計測で `change_voice` が **dep 129（リポジトリ最大の def-use 結合）** ・12引数と判明。cog 13 / ccn 11 は控えめで、cog/ccn 単独では見逃す「エンタングル状態」型の複雑性。本タスクはその解消。

## 1. 背景と目的

`vspeech/lib/rvc.py:207` の `change_voice` は bytes→音声変換の直線パイプライン（デコード→リサンプル→reflect パディング→HuBERT 特徴抽出→ピッチ調整→half 判定→推論→トリム）。dep 129 の正体は **1 スコープに共変ローカルが密集**していること:

- `feats` を 6 回、`pitch`/`pitchf` を各 3〜4 回、`audio` を複数回、`p_len_tensor` を 2 回再代入。
- DepDegree（def-use エッジ数）は関数単位の指標なので、各ステージを「入力→出力が明確な関数」に切り出せば、ローカルが各スコープに分散し `change_voice` 自身の dep は ~20–30 に低下する。本体は薄いオーケストレータ（~30 行）になる。

目的: **振る舞いを完全保存したまま** `change_voice` の def-use 結合を下げ、各ステージを単体検証可能にする。

## 2. スコープ / 非目標

- **スコープ**: `change_voice` 内部の純粋ヘルパ抽出。シグネチャ（12 引数）は不変。唯一の呼び出し元 [`vspeech/worker/vc.py` の `rvc_worker`](../../../vspeech/worker/vc.py)（warmup・本処理の 2 箇所）は不変。
- **非目標（YAGNI）**:
  - 12 引数を dataclass にバンドルする案は**不採用**（vc.py 2 箇所への波及を避け、blast radius を最小化）。
  - `infer` / `extract_features` / `pitch_extract` など既存関数のシグネチャ変更。
  - 数値挙動・dtype・演算順序・スライス意味の変更。

## 3. 分解設計

`change_voice` を以下のステージ関数に分割する（`pure` = GPU/モデル不要でユニットテスト可、`model` = ゴールデンで担保）:

| 新ヘルパ（`vspeech/lib/rvc.py` 内・module-private `_` 接頭辞） | 担当（現行行） | 種別 |
|---|---|---|
| `_pad_input_to_block(voice_frames) -> NDArray[float32]` | bytes→int16→128 境界へ切り上げ→/32768→左ゼロ詰め（L222–229） | **pure** |
| `_quality_padding(audio, rvc_config, voice_sample_rate, target_sample_rate) -> tuple[Tensor, int]` | repeat/quality_padding_sec/t_pad/t_pad_tgt 算出＋reflect pad → `(audio_pad, t_pad_tgt)`（L236–240） | **pure**（CPU torch） |
| `_extract_hubert_feats(hubert_model, audio_pad, device, half_available, emb_output_layer, use_final_proj) -> Tensor` | half/float→mean channels→view(1,-1)→extract_features→interpolate×2（L246–267） | model |
| `_select_pitch(audio_pad, rvc_config, f0_enabled, p_len, device, rmvpe_session) -> tuple[Tensor\|None, Tensor\|None]` | f0 分岐＋pitch_extract＋p_len スライス＋to-tensor（L273–288） | model（f0=False 枝は pure） |
| `_is_model_half(session) -> bool` | `session.get_inputs()[0].type == "tensor(float)"` 判定（L297–301） | **pure**（fake session） |
| `_align_pitch_to_feats(pitch, pitchf, feats_len) -> tuple[Tensor\|None, Tensor\|None]` | feats 長へ末尾スライス（L304–306） | **pure**（CPU torch） |
| `_postprocess(audio1, t_pad_tgt) -> NDArray[int16]` | パディングトリム→detach/cpu/numpy（L330–337） | **pure**（CPU torch） |

オーケストレータ `change_voice` は上記を順に呼ぶだけの薄い本体（device 転送・`sid` 構築・`p_len` 算出・`infer` 呼び・ログ・`del` のみを保持）。

### 3.1 ついでに直す latent（dep にも効く・ゴールデンで安全に担保）

- **L289 の `p_len_tensor` dead store**: `torch.tensor([p_len], ...)` は L307 で `torch.tensor([feats_len], ...)` に上書きされる前に一度も読まれない未使用代入。削除（def-use エッジ減＝dep に直接寄与、実質クルフト除去）。
- **L241–242 の `sid` 二重代入** (`sid = 0; sid = torch.tensor(...)`): 1 行に統合。

いずれも数値挙動不変。これ以外の挙動は厳密保存する。

## 4. 検証戦略（全工程この環境で実施）

実機確認済み: CUDA 有効・onnxruntime-gpu（CUDAExecutionProvider）・torch/torchaudio/fairseq import OK・モデル 3 点（RVC モデル onnx / HuBERT `.pt` / rmvpe onnx）存在。**rvc 依存は base env に無いため全実行は `uv run --extra rvc`**。

### 4.1 純粋ヘルパのユニットテスト（新規 `tests/test_rvc_helpers.py`）

実 torch（CPU）＋fake session で特性テスト。`test_pitch_extract.py` の fake-session 手法に倣う。CI でも回る（GPU 不要）。最低限:

- `_pad_input_to_block`: 128 非整数長→切り上げ・左ゼロ詰め・正規化値・dtype。
- `_quality_padding`: **quality>0**（reflect pad が効くケース）と **quality=0**（no-op）両方で `audio_pad` 形状と `t_pad_tgt`。
- `_is_model_half`: `"tensor(float)"`→False / それ以外→True（fake session）。
- `_align_pitch_to_feats`: 末尾 `feats_len` スライス、`None,None` パススルー。
- `_postprocess`: `t_pad_tgt=0`（トリム無し）と `>0`（前後トリム）、int16 化。
- `_select_pitch` の **f0_enabled=False 枝** → `(None, None)`。

> 補足: 本番 config は `quality=0` のためゴールデン単独ではパディング分岐が未到達。ここを quality>0 のユニットテストで網羅することが 7 分割＋ユニットテストの主目的。

### 4.2 数値ゴールデンテスト（全パイプライン担保）

> **重要な実測知見（2026-06-25）**: RVC シンセサイザ（VITS 系）は内部でランダムノイズを生成するため `change_voice` の最終出力は**設計上 run-to-run で非決定的**（実測: CUDA でも CPU EP でも自己差分 mean ≈ 3.6%／max ≈ 1.5万 LSB）。当初想定の「自己ノイズ床を許容差にした近似一致」では挙動保存と微小破壊を区別できない。一方、(a) この乱数は**シード可能** — 実行直前に `torch.manual_seed` + `torch.cuda.manual_seed_all` + `onnxruntime.set_seed` を設定すると**出力が bit-exact 再現**（実測 self-noise 0）。(b) リファクタが触る `infer` **前**の演算（feats/pitch/pitchf/p_len/sid）は無シードでも**完全決定的**（run 間 max=0、確率的なのは未改変の `infer` のみ）。よってゴールデンは **シード固定で厳密一致** に強化する（許容差ベースを廃し、`atol≈0`／微小マージンのみ）。

1. **リファクタ前**に `scripts/capture_change_voice_golden.py` を追加（本体は無改変）。`--config <RVC worker の TOML config>` を読み、`rvc_worker` と**同手順**で device(`get_device(gpu_id, gpu_name)`)／hubert／RVC session／rmvpe session を構築し modelmeta から `target_sample_rate`/`f0`/`embOutputLayer`/`useFinalProj` を取得。**固定の決定的入力**（シード固定のサイン波）に対し、**実行直前に全 RNG をシード**して `change_voice` を実行（保険として 2 回実行し bit-exact を確認）、入力・seed・seeded 出力 int16 を `tests/assets/rvc_golden/change_voice_golden.npz`（gitignore 下）に保存。
2. `change_voice` をリファクタ。
3. `tests/test_change_voice_golden.py` を追加。npz・CUDA・モデルが無ければ **skip**。保存 seed で**同一にシード**しリファクタ後 `change_voice` を実行、保存出力と **厳密一致**（`max|Δ| ≤ 1` の微小マージン）を確認。挙動破壊は infer 入力を変え RNG ストリームを逸らすため出力が数千 LSB ずれて確実に検知される。
4. `uv run --extra rvc poe check`（ruff/ty/pytest）を green に。

全工程をこの環境で私（Claude）が実行する。ユーザの実機操作は不要。

## 5. リスクと対処

- **GPU 非決定性**（torch.compile max-autotune＋onnxruntime CUDA で run-to-run 微差）: ゴールデンは「2 回実行の自己差分」を測りそれを上回る許容差を設定。int16 量子化境界での ±1 LSB 揺れも許容差に含める。
- **device 選択ずれ**（複数 GPU 環境で device0 が config 指定 GPU と異なる場合）: 捕捉スクリプトは worker と同じ `get_device(gpu_id, gpu_name)` を使い、production と同一デバイスを選ぶ。
- **warmup コスト**（初回推論が torch.compile/onnx グラフ構築で長い、最大 ~145s）: 捕捉実行のタイムアウトを十分長く取る。
- **dep が想定ほど下がらない**: メトリクスは `uv run poe metrics` で事後計測し、`change_voice` の dep 低下と各ヘルパが小さいことを確認する（受け入れ基準）。

## 6. 受け入れ基準

- `change_voice` の振る舞いがゴールデンテストで数値一致（許容差内）。
- 新規ユニットテスト＋既存テストが green、`uv run poe check` green。
- `uv run poe metrics` で `change_voice` の dep が大幅低下（目安 ~20–30）、新ヘルパは個別に低 dep。
- vc.py（呼び出し元）は無改変。
