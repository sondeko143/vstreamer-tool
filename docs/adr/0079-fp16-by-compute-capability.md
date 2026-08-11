# 0079. fp16 の可否を GPU の製品名ではなく compute capability で判定する

- Status: Proposed
- Date: 2026-08-11
- 効力: 既定
- Related: [ADR-0078](0078-torch-free-device-resolution.md)（compute capability を取得する列挙経路）; spec [2026-08-11-torch-free-device-layer-design.md](../superpowers/specs/2026-08-11-torch-free-device-layer-design.md)

## Context

RVC の fp16 可否は現在、GPU の**製品名の文字列**で判定している。名前が `"16"` を含むか（ただし V100 は除く）、`P40` / `1070` / `1080` を含むかで fp16 を無効化する。上流 RVC から引き継いだ判定である。

判定したい性質はハードウェアの fp16 演算レートであって、製品名ではない。名前一致には 2 つの欠陥がある。

- **誤爆する。** `"16" in name` は製品名に含まれる 16 を無差別に拾う。`NVIDIA A16`（sm_86）のように 16 を含む無関係な製品名や、VRAM 容量を名前に含むドライバ表記に対して、理由なく fp16 を無効化する。
- **保守が要る。** 新しい GPU が出るたびに、除外すべきかをリストに反映し続けなければならない。

[ADR-0078](0078-torch-free-device-resolution.md) でデバイス列挙を CUDA Driver API に移すと、compute capability が同じ経路で取得できるようになる。

## Decision

fp16 可否を compute capability で決める。

| compute capability | fp16 のレート | 判定 |
|---|---|---|
| >= 7.0（Volta 以降） | tensor core があり fp32 以上 | 可 |
| == 6.0（GP100） | 2:1 | 可 |
| 6.1 / 6.2（GP10x = GTX 10 系, Tesla P40） | 1/64 | 不可 |
| < 6.0 | native fp16 なし | 不可 |

判定を `(major, minor) -> bool` の純関数として切り出し、GPU を持たない環境でも境界値を検証できるようにする。

判定関数はデバイス値型そのものを受け取り、CPU デバイスを明示的に不可とする。現行は CPU 時にデバイス番号が `None` のまま torch に渡って例外が飛び、`except` で False になる偶然頼りの経路になっている。

## Alternatives rejected

- **製品名のブラックリストを維持する** — 誤爆と保守コストの両方が残る。判定したい性質が名前から得られないという根本問題は解決しない。
- **ブラックリストと CC 判定を両方適用する** — GTX 16xx の除外を保てるが、その除外の根拠は torch の `half` 由来の NaN 問題であり、推論が ONNX Runtime に移った現在の経路には引き継がれていない。根拠のない除外を残すと、なぜ除外されているかを次の担当者が再調査することになる。
- **compute capability ではなく実測ベンチマークで判定する** — 最も正確だが、起動のたびにベンチマークを走らせることになり、削減しようとしている起動時間を自ら増やす。

## Consequences

現行と判定が変わるのは **GTX 16xx（sm_75）のみ**で、不可 → 可になる。TU116 / TU117 は tensor core を持たないが fp16 は 2:1 レートで回るため、CC 判定の方が物理的に正しい値を返す。

`Tesla P40`（sm_61）、`GTX 1070/1080`（sm_61）、`V100`（sm_70）、開発機の `RTX 4060`（sm_89）/ `RTX 5060 Ti`（sm_120）は現行と同じ結果になる。

新しい GPU に対してリスト保守が不要になる。

GTX 16xx の実機確認はできていない。この世代で fp16 に問題が出た場合、CC 判定に例外を足して名前判定に逆戻りさせるのではなく、fp16 を設定で切れるようにするのが筋である（今回のスコープ外）。
