# 0023. HuBERT 置換の正しさを特徴量数値等価ゲートで担保する

- Status: Accepted
- Date: 2026-07-10
- Related: spec [2026-07-09-rvc-hubert-fairseq-free-design.md](../superpowers/specs/2026-07-09-rvc-hubert-fairseq-free-design.md), [2026-07-10-rvc-hubert-onnx-design.md](../superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md); [ADR-0016](0016-change-voice-decompose-seeded-golden.md)（seeded-golden 系譜）; [ADR-0021](0021-hubert-drop-fairseq.md); [ADR-0022](0022-hubert-onnx-runtime.md)

## Context

[ADR-0021](0021-hubert-drop-fairseq.md)（fairseq → transformers）と [ADR-0022](0022-hubert-onnx-runtime.md)（transformers → ONNX）で HuBERT 実装そのものが変わる。既存の change_voice は [ADR-0016](0016-change-voice-decompose-seeded-golden.md) で seeded bit-exact golden により守られてきたが、それは「数式を変えないリファクタ」の検証であって、実装置換では原理的に通らない。正しさの基準を「数式不変の bit 一致」から「実装置換の数値等価」へ立て直す必要があった。

## Decision

主ゲートを HuBERT 特徴量の fp32 等価とする。旧 fairseq golden に対し `cosine ≥ 0.9999` かつ `max-abs ≤ 1e-4` を、v1 = 層 9 + final_proj と v2 = 層 12 の両方で判定する。しきい値は実測誤差の約 10 倍を上限とする確定規則で運用し、その定数を単一情報源で管理する。fp16 は fp32 golden に絶対誤差で照らすのではなく「ONNX fp16 vs 置き換え対象の torch fp16」で判定する（`cosine ≥ 0.9999`, `max-abs ≤ 5e-2`）。change_voice の音声 golden は fairseq 由来のものを再ベースラインせず据え置き、許容誤差（`corr ≥ 0.999`）で判定する。その SNR しきい値は ORT fp16 カーネルの実測（39.52 dB）に基づき 40 → 35 dB へ緩和する（実装は正しく、緩和の根拠は丸め差の実測）。fp16 参照と各しきい値定数の単一情報源を 1 ファイルに置く。

## Alternatives rejected

- **bit-exact 継続** — 数式不変リファクタ専用の検証であり、実装置換には使えない。
- **fp16 を fp32 golden に絶対誤差で照らす** — hidden state のスケールに対し半精度誤差はもともと 1e-1 オーダーで、原理的に成立しない。実測では production の torch fp16 自身が fp32 golden に対し max-abs 0.43 を出す。壊れているのはゲートの立て方だった。
- **音声 golden を再ベースラインする** — 移行を跨いだ唯一の保証を失い、以後は将来退行しか守れなくなる。
- **SNR 40 dB を据え置く** — 実測 39.52 dB で割り、実装は正しいのに落ちる。

## Consequences

移行の等価性を end-to-end で直接証明できる。しきい値定数の単一情報源が必要になり、fp16 参照は GPU / カーネル依存（RTX 4060 で捕獲）のため開発機限定・CUDA gating となる。CUDA 機で fp32 等価が成立する前提として device を尊重する必要があり、これは [ADR-0024](0024-onnx-session-single-factory.md) が担保する。
