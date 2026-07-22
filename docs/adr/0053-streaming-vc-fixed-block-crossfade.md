# 0053. ストリーミング VC を固定ブロック+左文脈+クロスフェードのステートフル変換にする

- Status: Proposed
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0016](0016-change-voice-decompose-seeded-golden.md), [0017](0017-rvc-input-envelope-shape-transfer.md), [0019](0019-vc-silero-vad-gate.md), [0050](0050-streaming-vc-separate-subsystem.md)

## Context

既存の `change_voice` は発話全体を reflect-pad して全区間を一括推論するステートレス変換。これを短いブロックにそのまま適用すると、reflect-pad の文脈がゴミで境界が壊れ、独立推論した出力の単純連結でクリックが出る。ONNX デコーダはステートレスで隠れ状態を持ち越せない。f0(RMVPE/FCPE)や HuBERT は左右文脈がないと端が不安定になる。onnxruntime のグラフ構築は shape 依存で、可変長入力だと毎回オートチューンのコストがかかる(発話単位では 145s の初回 stall 既往)。

## Decision

ストリーミング VC を**固定長ブロック単位のステートフル変換**にする。

- rolling context buffer で直近の実音声を左文脈として保持し、毎 tick `[context | 新ブロック]` を推論して、出力は新ブロック相当だけ採用する(reflect-pad ではなく実音声の左文脈を与える)。
- 隣接出力の overlap 区間を等電力クロスフェードで overlap-add してクリックを消す(crossfade tail をブロック間状態として保持)。
- 入力 shape を固定し、warmup を 1 回で済ませる(以後 re-autotune なし)。
- 既存 `change_voice` の内部部品(HuBERT 特徴量抽出 / f0 抽出 / infer / int16 化)は再利用するが、発話系の `change_voice` 経路自体は無改変で温存する。
- envelope 整合は発話全体平均に依存するため、streaming では rolling(EMA)基準へ置換するか既定 off とする。VAD はブロック粒度のバタつきを避けるため hangover 付きステートフルゲートとする(既定 off、fail-open)。

## Alternatives rejected

- **発話全体 reflect-pad の `change_voice` をブロックにそのまま呼ぶ(ステートレス流用)** — 短ブロックの reflect-pad 文脈がゴミで境界破綻、クロスフェード無しでクリック、f0 端不安定、可変長で毎回グラフ再構築。streaming では成立しない。
- **発話単位の envelope/VAD をそのままブロックへ適用** — 発話全体平均 RMS 正規化やブロック粒度 VAD は、基準/ゲートがブロックごとに飛んで pumping/choppy になる。

## Consequences

クリック無し・ピッチ連続の連続変換が、固定 shape で安定して回る(warmup 1 回)。既存 `change_voice` を壊さず内部部品を再利用するので差分を局所化できる。反面、context 分の余剰推論で per-block RTF が増える(M1 実測の主対象)。クロスフェード/f0 連続性の正しさは seeded golden で担保が要る([0016](0016-change-voice-decompose-seeded-golden.md) の決定的シード基盤を流用)。
