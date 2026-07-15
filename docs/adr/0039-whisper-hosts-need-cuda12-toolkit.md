# 0039. whisper GPU ホストに CUDA 12 ツールキット（cuBLAS + cuDNN 9）を要求する（0028 を refine）

- Status: Accepted
- Date: 2026-07-16
- Related: refines [ADR-0028](0028-migrate-to-cuda-13.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

CUDA 13 化（[ADR-0028](0028-migrate-to-cuda-13.md)）後、whisper が使う `ctranslate2` 4.8.x（faster-whisper 経由）は **CUDA 12 専用**で、推論時に `cublas64_12.dll` を要求する。CUDA 13 ビルドは存在しない（faster-whisper [#1431](https://github.com/SYSTRAN/faster-whisper/issues/1431)）。cu128 時代は torch 同梱の `torch/lib/cublas64_12.dll` がこれを供給していたが、`+cu130` 化で torch が持つのは `cublas64_13.dll` になり供給元が消えた（`whisper warmup failed: Library cublas64_12.dll is not found`）。2026-07-12 に実機の whisper ホストへ CUDA Toolkit 12.8 を入れて cuBLAS を供給し、動作を確認した。

## Decision

whisper を GPU で回すホストには、R580+ ドライバに加えて **CUDA 12 ツールキット（cuBLAS + cuDNN 9）** を導入することをデプロイ要件とする。CUDA 13 化（0028）は維持する。CUDA 13.x ドライバは CUDA 12 アプリも走らせるので、torch / onnxruntime（CUDA 13）と ctranslate2（CUDA 12）が 1 プロセスで共存する。vc / RVC 専用ホスト（torch + onnxruntime のみ）はドライバだけでよい。

## Alternatives rejected

- **whisper extra に `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` を pip 依存で足し、whisper worker で `os.add_dll_directory` して bin を DLL 探索に載せる** — ツールキット導入は不要になるが、Windows の DLL 探索が壊れやすい（faster-whisper [#1276](https://github.com/SYSTRAN/faster-whisper/issues/1276) の沼）。実機 whisper 推論の検証コストを払う価値が出たら再検討する。
- **CUDA 13 化そのものを巻き戻す**（torch `cu130`→`cu128`・onnxruntime `1.27`→`<1.27`、cp314 は維持） — 全て CUDA 12 に揃い whisper は無改修で動くが、0028 の唯一の利点（onnxruntime を 1.27+ に保つ更新可能性）を失う。

## Consequences

whisper GPU ホストのプロビジョニングに、ドライバ以外の手順（CUDA 12 ツールキット）が 1 つ増える。混在 CUDA が 1 プロセスで同居する（runtime 利得は無く、価値は 0028 の更新可能性維持にある）。教訓: フェーズ②の混在 CUDA 検証を `ctranslate2.get_cuda_device_count()`（デバイス数照会だけで cuBLAS をロードしない）で済ませたため、cuBLAS 欠如をコミット前に検知できなかった。実際の whisper 推論を 1 回回していれば気づけた。
