# 0028. CUDA 13 へ移行する（torch cu130 / onnxruntime 1.27 / driver R580+）

- Status: Accepted
- Date: 2026-07-12
- Related: supersedes [ADR-0027](0027-cap-onnxruntime-cuda12.md); [ADR-0025](0025-target-python-314-phased.md); spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md)

## Context

CUDA 12 に留まると onnxruntime-gpu を [ADR-0027](0027-cap-onnxruntime-cuda12.md) の `<1.27` で恒久的に cap することになり、修正も新 opset も来ない版に取り残される。[ADR-0026](0026-adopt-numpy-2.md) のフェーズ①が green になり、CUDA を単独変数として動かせる状態になった。ここで cap を解除する。

## Decision

フェーズ②で torch / torchaudio を `+cu128` → `+cu130`、onnxruntime cap を `>=1.27` へ移し CUDA 13 化する。全 GPU ホストの NVIDIA ドライバ R580+ を前提とする（torch / onnxruntime が CUDA ランタイムを同梱するため、要件はドライバのみ）。

## Alternatives rejected

- **CUDA 12 に留まる** — onnxruntime-gpu が `<1.27` の恒久 cap で更新不能になる。

## Consequences

ctranslate2（CUDA 12 同梱）と torch / onnxruntime（CUDA 13）の混在 CUDA が 1 プロセスで同居する。runtime 利得はほぼ無く、価値は更新可能性の維持にある。既知の落とし穴: ctranslate2 は CUDA 12 専用で `cublas64_12.dll` を要するが、torch cu130 はそれを同梱しなくなったため、whisper GPU ホストはドライバに加えて CUDA 12 ツールキット / cuBLAS が別途必要になる（vc / RVC のみのホストはドライバだけで足りる）。
