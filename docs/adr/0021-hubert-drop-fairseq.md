# 0021. RVC content encoder を fairseq から transformers.HubertModel へ外す

- Status: Accepted
- Date: 2026-07-09
- Related: spec [2026-07-09-rvc-hubert-fairseq-free-design.md](../superpowers/specs/2026-07-09-rvc-hubert-fairseq-free-design.md); [ADR-0023](0023-hubert-equivalence-gate.md)（等価ゲート）; runtime の transformers は [ADR-0022](0022-hubert-onnx-runtime.md) で撤去

## Context

`rvc` extra が依存する `fairseq` が、Python バージョン引き上げの唯一にして最大の障害になっていた。公式 `facebookresearch/fairseq` の最終リリースは 0.12.2（2022-06-27）で以降ゼロ、リポジトリは 2026-03-20 に archived（read-only）となり、3.11 / 3.12 対応の追跡 Issue は未対応のままアーカイブ入りした。公式修正は永久に来ない。3.11 以降では import 時点で dataclass の可変デフォルト規制により落ちる。後継の `fairseq2` は API 非互換の全面リライトで、本 repo が使う `checkpoint_utils` / `HubertModel` とはドロップイン互換にならない。このため自前ビルドの cp311 固定 wheel（`sondeko143/fairseq-311`）をピンしており、この 1 本が `torch` / `torchaudio` の cp311 URL ピンと `numpy<1.24`（1.23.5 は cp312 wheel が無い）を連鎖的に縛っていた。3 つのピンはすべて fairseq 由来であり、fairseq を外せば結び目ごと解ける。fairseq の実使用は `vspeech/lib/rvc.py` の 2 箇所（HuBERT ロードと特徴量抽出）に閉じており、使用層と final_proj 適用有無は RVC モデルの ONNX メタデータ（既定 `9` / `True`）でモデルごとに決まる。

## Decision

RVC の content encoder を fairseq から `transformers.HubertModel`（eager 実行）へ置換し、runtime を fairseq 非依存にする。fairseq → transformers への変換は 3.11 + fairseq の使い捨てオフライン環境で一度きり実行する工程とし、runtime パスからは完全に隔離する。共有 `final_proj`（768→256 の Linear）は全 RVC モデルで共有される手元 `hubert_base.pt` の同じ state_dict から抽出する（新規アセットの入手不要、fairseq が使っていた重みとの厳密一致が保証される）。fairseq は変換専用の `convert` extra へ隔離する。`output_layer` ↔ `hidden_states` の対応は変換時に実測で同定し `mapping.json` に固定して、runtime は推測しない。config フィールド `hubert_model_file` は名前を据え置き、意味だけ「`.pt` ファイル」→「変換済み資産ディレクトリ」へ変える。numpy 上限は fairseq 由来の `<1.24` を `<2` へ緩和するに留める。

## Alternatives rejected

- **fairseq を継続する** — archived で修正が来ず、cp311 固定 wheel が torch / numpy を連鎖拘束して Python 引き上げが不能。
- **fairseq2 へ移行する** — API 全面リライトで `checkpoint_utils` / `HubertModel` と非互換、ドロップイン代替にならない。
- **runtime で HF hub から重みをダウンロードする** — ネットワーク依存となり、既存の手元 fairseq 重みとの厳密一致が保証されない。
- **numpy を即 `<3` へ上げる** — `pyworld` / `faiss` / `onnxruntime-gpu` の numpy 2.x 対応が未確認で、1 手にリスクを集約する。

## Consequences

`torch` / `numpy` の pin が解け、Python 引き上げへの道が開く。代わりに `transformers` を runtime 依存として一時的に抱える（[ADR-0022](0022-hubert-onnx-runtime.md) で ONNX 化して撤去する）。golden 捕獲は fairseq が動く環境でしか行えないため、実装順序が「変換ツール + golden 捕獲 → runtime 置換 → 等価ゲートで検証」に固定される。逆順にすると正解データを失う。
