# 0022. HuBERT content encoder を ONNX 化し fairseq/transformers を lock から撤去する

- Status: Accepted (refines [ADR-0021](0021-hubert-drop-fairseq.md))
- Date: 2026-07-10
- Related: spec [2026-07-10-rvc-hubert-onnx-design.md](../superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md); [ADR-0021](0021-hubert-drop-fairseq.md); [ADR-0023](0023-hubert-equivalence-gate.md); [ADR-0024](0024-onnx-session-single-factory.md)

## Context

[ADR-0021](0021-hubert-drop-fairseq.md) で fairseq を `transformers` へ隔離し runtime を fairseq 非依存にしたが、「extra に隔離する」だけでは Python 3.12 化の最後の障害が解けなかった。`uv.lock` は universal lock であり、extra に宣言されているだけのパッケージもロックに残り、`uv audit` はロック全体を監査する。そのため `transformers` の CVE 3 件がロックに居座り続けた。さらに変換専用 `convert` extra が持つ `numpy>=1.18,<1.24` が universal 解決を numpy 1.23.5（cp312 wheel が存在しない 2022 年版）に固定していた。これらは spec ③（Python 3.12 化）の最後の障害そのものである。オフラインツールの依存を `pyproject.toml` から完全に外さない限り解けない。

## Decision

HuBERT content encoder を ONNX へ export し、runtime を `onnxruntime` の `InferenceSession` だけで動かす（`transformers` / `safetensors` を runtime から撤去する）。RVC 本体・RMVPE と同じセッション基盤に 3 本目として載る。`final_proj` をグラフに焼き込んだ 2 出力グラフ（`feats_l9_proj` / `feats_l12_raw`。実在する RVC モデルは v1 = `(9, True)` と v2 = `(12, False)` の 2 種のみ）を export し、層インデックスは export 時に解決してグラフへ固定する（runtime は `mapping.json` の対応表だけ読み、層を推測しない）。fairseq と transformers を `pyproject.toml` と `uv.lock` から完全に取り除き、変換（fairseq → transformers）と export（transformers → ONNX）の依存は extra ではなく `uv run --with` オーバーレイの poe task で供給する（既存の `poe security` / `poe deadcode` と同じ流儀）。export は dynamo exporter を第一候補とし、失敗時のみ legacy exporter へフォールバックするが、その際は必ず例外型と traceback を印字して大声で報告する。これにより `uv.lock` が解決する numpy が `>=1.24,<2` へ上がる（whisper/rvc extra の宣言下限は `>=1.23,<2` のまま。numpy 2 化そのものは [ADR-0026](0026-adopt-numpy-2.md) のフェーズ①で行う）。

## Alternatives rejected

- **transformers を convert extra に隔離継続 / runtime で eager 継続する** — universal lock に残るため CVE 3 件も numpy `<1.24` 固定も解けない。
- **final_proj を別 safetensors + Linear として runtime に残す** — runtime に `safetensors` / `torch.nn.Linear` が居残る。グラフに焼き込めば消える。
- **フォールバックを静かに行う** — 2026-07-10 に実際に事故。torch.onnx が進捗表示の絵文字を Windows cp1252 stdout へ書こうとして `UnicodeEncodeError` で落ち、広い `except` がそれを「dynamo 失敗」と誤認して劣った legacy 経路を静かに使っていた。

## Consequences

`fairseq` / `transformers` が lock から消え numpy 下限が外れ、Python 3.12 化の最後の障害が除去される。オフライン工程が既存の `poe security` / `poe deadcode` と同じ `--with` 流儀に揃う。runtime は `mapping.json` の対応表だけを信頼し、未登録の (layer, proj) 組合せには対応表を添えて明示例外を投げる。fp16 経路の妥当性判定は [ADR-0023](0023-hubert-equivalence-gate.md) が引き受ける。
