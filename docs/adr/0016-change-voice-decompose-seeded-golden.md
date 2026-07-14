# 0016. change_voice を純粋ヘルパへ分解し seeded 厳密 golden で検証する

- Status: Accepted
- Date: 2026-06-25
- Related: [ADR-0008](0008-code-metrics-two-lens.md)（dep レンズが起点）/ spec [2026-06-25-change-voice-refactor-design](../superpowers/specs/2026-06-25-change-voice-refactor-design.md)

## Context

code-metrics の `dep` レンズ（[ADR-0008](0008-code-metrics-two-lens.md)、def-use 結合）が、RVC の `change_voice` を dep 129（リポジトリ最大の def-use 結合）・12引数と検出した。cog 13 / ccn 11 は控えめで、cog/ccn 単独では見逃す「エンタングル状態」型の複雑性である。正体は 1 スコープに共変ローカル（`feats`/`pitch`/`pitchf`/`audio`/`p_len_tensor` の多重再代入）が密集していること。各ステージを入出力の明確な関数へ切り出せばローカルが各スコープに分散し dep が下がる。ただし RVC シンセサイザ（VITS 系）は内部で乱数を生成するため `change_voice` の最終出力は設計上 run-to-run で非決定的（CUDA/CPU とも mean≈3.6%）であり、単純な許容差ゴールデンでは挙動保存と本物の微小ドリフトを分離できない。

## Decision

`change_voice`（dep 129・12引数）を挙動を完全保存したまま各ステージを純粋/モデルヘルパへ切り出し、本体を薄いオーケストレータにする。12引数シグネチャと唯一の呼び出し元（vc.py の 2 箇所）は不変とする。挙動保存の検証は、推論直前に `torch.manual_seed` + `torch.cuda.manual_seed_all` + `onnxruntime.set_seed` を設定すると出力が bit-exact 再現される性質を使い、シード固定の厳密ゴールデン（`max|Δ| ≤ 1` の微小マージン）で行う。許容差ベースの近似ゴールデンは採らない。

## Alternatives rejected

- **12引数を dataclass にバンドル** — 呼び出し元 2 箇所（vc.py）へ波及し blast radius が増える。引数の多さは別課題として残す。
- **許容差ベースの近似ゴールデン（`atol_max = max(8, 4×self_noise)`）** — 非決定性の床と本物の微小ドリフトを分離できず、挙動破壊を見逃す。挙動破壊は infer 入力を変えて RNG ストリームを逸らすため、seeded 厳密一致なら数千 LSB のずれで確実に検知できる。

## Consequences

dep が各スコープに分散して低下する（目安 ~20–30、リポジトリ最大の both-high から外れる）。純粋ヘルパは個別に単体検証可能になる。引数の多さ自体は別課題として残る。以後の HuBERT 置換検証（[ADR-0023](0023-hubert-equivalence-gate.md)）はこの seeded-golden の系譜を引き継ぐ。
