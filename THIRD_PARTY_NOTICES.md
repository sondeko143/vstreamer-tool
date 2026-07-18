# Third-Party Notices / サードパーティ ライセンス表示

本リポジトリ（`voicerecog` / `vspeech` パッケージ + `gui`）のソースコードは
MIT License（同梱の [`LICENSE`](LICENSE)）で配布します。

## このファイルの対象

含むもの:

- 本リポジトリのコードが取り込んだ／基にした上流コード（第1節）
- 実行時に利用するモデル（第2節・第3節）

含まないもの:

- 宣言済みの pip 依存（torch・onnxruntime・faster-whisper など、各 wheel 同梱のライセンスに従う）

本ファイルは法的助言ではありません。再配布や同梱を行う場合は、各上流の LICENSE 原文
（必要に応じて専門家）を確認してください。記載内容は 2026-07-18 時点の各上流 LICENSE に
基づきます。

## モデルの同梱について

本リポジトリはモデルファイルを同梱しません。RMVPE / ContentVec (HuBERT) / Silero VAD /
RVC 声モデルは、利用者が各自取得して設定ファイルのパスで指定します（見つからなければ
起動を中止します）。`.gitignore` は `*.onnx` / `*.pt` / `*.pth` / `*.safetensors` /
`*.npz` を除外し、モデルファイルの誤コミットを防ぎます。本コードは MIT で配布し、各モデルの
ライセンスは第2節・第3節に記載します。

---

## 1. 上流コードの表示

本プロジェクトの一部コードは、以下のプロジェクトのコードまたは入出力規約を基にしています。
各上流の著作権表示を保持します。

### 1-a. MIT

#### RVC WebUI — Retrieval-based-Voice-Conversion-WebUI

<https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>

- Copyright (c) 2023 liujing04
- Copyright (c) 2023 源文雨
- Copyright (c) 2023 Ftps
- 該当箇所: `vspeech/lib/rvc.py`（fp16 判定 `half_precision_available` など）、
  `vspeech/lib/pitch_extract.py`（f0 の量子化）、`vspeech/worker/vc.py`
  （`apply_input_envelope`）。

#### VCClient — w-okada/voice-changer

<https://github.com/w-okada/voice-changer>

- Copyright (c) 2022 Wataru Okada
- 該当箇所: RVC 推論の ONNX 入出力名・fp16 判定・RMVPE の入出力契約。

#### Silero VAD — snakers4/silero-vad

<https://github.com/snakers4/silero-vad>

- Copyright (c) 2020-present Silero Team
- 該当箇所: `vspeech/lib/vad.py`（VAD 前処理の実装）。

#### MIT 許諾文（上記 3 プロジェクト共通）

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1-b. Apache-2.0

#### HuggingFace transformers

<https://github.com/huggingface/transformers>

- Copyright 2018- The Hugging Face team
- 該当箇所: `scripts/hubert_keymap.py` の fairseq→transformers キー対応。transformers の
  チェックポイント変換スクリプトと同じ対応関係を用います。Apache-2.0 の帰属要件に従い
  表示します。

---

## 2. 実行時に利用するモデル

以下のモデルは本リポジトリに含まれず、利用者が各自取得します。各モデルは独自のライセンスを
持ちます。

| モデル | 用途 | 入手元 | ライセンス |
| --- | --- | --- | --- |
| Silero VAD `silero_vad.onnx` (v6.2.1) | VAD ゲート | snakers4/silero-vad | MIT（※1） |
| ContentVec `hubert_base.pt` → 変換済み ONNX | RVC content encoder | auspicious3000/contentvec | MIT (Copyright (c) 2019) |
| RVC 声モデル (`.onnx`) | 声の変換 | 利用者が用意 | 自作または第三者モデル（第三者モデルは各配布元の規約に従う） |
| RMVPE `rmvpe_20231006.onnx` | f0 抽出（既定） | wok000/weights_gpl | GPL-3.0（※2） |

※1 Silero VAD の上流は repo 全体が単一の MIT LICENSE で、`.onnx` モデルを含みます。モデル
のみを対象とする別のライセンス条項はありません。
※2 詳細は第3節。

### ContentVec の変換について

RVC の content encoder は、利用者が ContentVec のチェックポイント（RVC では
`hubert_base.pt` として配布）を手元で ONNX へ変換して用意します。変換スクリプトは
`scripts/convert_hubert.py` と `scripts/export_hubert_onnx.py` で、これらが使う fairseq
（MIT, Copyright (c) Facebook, Inc. and its affiliates）と transformers（Apache-2.0）は
利用者の一時環境で動きます。本リポジトリはこれらのコードもモデルも配布しません。変換済み
モデルは ContentVec の MIT に従います。

---

## 3. 既定の RMVPE モデルのライセンス（GPL-3.0）

既定の f0 抽出器 `rmvpe` が使う `rmvpe_20231006.onnx` は、配布元
[`wok000/weights_gpl`](https://huggingface.co/wok000/weights_gpl) が GPL-3.0 として
配布しています（README: *"The license for the models included in this repository is
GPL-v3.0. A different license can be used by agreeing to a contract with the
developers."*）。

本リポジトリはこのモデルファイルを含みません。同モデルを再配布または配布物へ同梱する場合は
GPL-3.0 が適用されます。
