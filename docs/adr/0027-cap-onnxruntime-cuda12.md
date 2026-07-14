# 0027. onnxruntime-gpu を一時 <1.27 cap で CUDA 12 を凍結する

- Status: Superseded by [ADR-0028](0028-migrate-to-cuda-13.md)
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

onnxruntime-gpu 1.27.0 は CUDA 13（`nvidia-*-cu13`）へ移行しており、torch `+cu128` と同居すると 1 プロセス内で CUDA 12 / 13 が混在し GPU 経路を静かに壊す。現状の制約 `>=1.24.4,<2` は 1.27.0 を許してしまい、次の `uv lock` が黙って 1.27 を引く。

## Decision

[ADR-0026](0026-adopt-numpy-2.md) のフェーズ①で numpy 移行を CUDA 変更から隔離するため、onnxruntime-gpu を `>=1.24.4,<1.27` にコメント付きで cap し、CUDA 12 を凍結する。CUDA 13 の採否はフェーズ②へ切り出す。

## Alternatives rejected

- **numpy 移行中に 1.27 を許す** — numpy と CUDA の 2 障害が重なり、退行の切り分けが不能になる。
- **cap にコメントを付けない** — 次の lock 再生成が黙って 1.27 を戻す。

## Consequences

numpy 移行を CUDA 変更から隔離できる。この cap は一時的なものであり、[ADR-0028](0028-migrate-to-cuda-13.md) のフェーズ②で解除される。
