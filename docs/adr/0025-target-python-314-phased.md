# 0025. 3.13 を経由せず段階移行で Python 3.14 を目標にする

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0026](0026-adopt-numpy-2.md), [ADR-0027](0027-cap-onnxruntime-cuda12.md), [ADR-0028](0028-migrate-to-cuda-13.md), [ADR-0029](0029-audioop-lts.md), [ADR-0030](0030-pyworld-lazy-default-rmvpe.md), [ADR-0031](0031-audio-pyaudio-to-sounddevice.md)

## Context

`requires-python` は `>=3.12,<3.13`。3.12 は bugfix フェーズ終了済みで security-only は 2028-10 まで（PEP 693）。サポート窓を延ばしたいが、numpy `<2` cap・CUDA 12.8・onnxruntime-gpu の cap・pyworld / audioop の cp314 ギャップが絡み合っている。これらはいずれも GPU / 音声の実機経路に触れ、CI では検証できない。

## Decision

3.13 を経由せず **Python 3.14（`>=3.14,<3.15`）を目標**とする。移行を「1 フェーズ = 1 変数」の 3 フェーズ（① numpy 2 化 → ② CUDA 13 化 → ③ Python 3.14 化）に分解し、各フェーズの実機検証が green になってから次へ進む。

## Alternatives rejected

- **3.13 を目標にする** — 有効化作業は 3.14 と同じで支援は 1 年短く、runtime 利得はゼロ（incremental GC は 3.13.0 final 直前に revert、free-threading は experimental）で 3.14 に劣る。
- **3.12 に留まる** — サポート窓が 2028-10 で切れる。
- **3 変数を同時に動かす単一移行** — GPU / 音声の退行の切り分けが不能になる。

## Consequences

得られるのはサポート窓の延長のみ（性能移行ではなく保守性移行であり緊急性は無い）。onnxruntime の cap がフェーズを跨いで進化する（① `<1.27` で凍結 → ② `>=1.27` で採用）。移行は遅いが、退行の原因を各フェーズ単位で追跡できる。
