# 0024. 全 onnxruntime セッションを単一 create_session で開き device を尊重する

- Status: Accepted
- Date: 2026-07-10
- Related: spec [2026-07-10-rvc-hubert-onnx-design.md](../superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md); [ADR-0019](0019-vc-silero-vad-gate.md)（VAD の CPU 固定は明示的例外）; [ADR-0022](0022-hubert-onnx-runtime.md)

## Context

[ADR-0022](0022-hubert-onnx-runtime.md) の HuBERT ONNX 化を実装する途中で既存バグを発見した。onnxruntime セッションの生成が呼び出し側の `device` を無視し、`torch.cuda.is_available()` だけで CUDA EP を選んでいた。config が CPU を指定しても GPU で走り、CUDA EP の TF32 で fp32 特徴量に `2.6e-3` の誤差が乗って、[ADR-0023](0023-hubert-equivalence-gate.md) の fp32 等価ゲートが CUDA 機で成立しない状態だった。等価ゲートを成立させる前提条件として、この device 無視を直す必要があった。

## Decision

GPU 対応の全 onnxruntime セッション（RVC デコーダ・HuBERT・RMVPE）を単一の `create_session` ファクトリから開き、呼び出し側が渡した `torch.device` を尊重する（provider を `cuda.is_available()` だけで決めない）。Silero VAD は小モデルで CUDA を取り合わないため、`CPUExecutionProvider` 固定の明示的な例外とする（[ADR-0019](0019-vc-silero-vad-gate.md)）。ファクトリの二重化を禁じ、provider ガードの修正が常に 1 箇所で済むようにする。

## Alternatives rejected

- なし（唯一の現実解。等価ゲート成立の前提条件）。ただし RMVPE 側の同型 device バグの同時修正は、スコープ管理のため本 ADR では見送る。

## Consequences

CPU 指定が CPU で走るようになり、fp32 等価ゲートが CUDA 機でも成立する。CLAUDE.md の onnx_session 不変条件および `tests/test_onnx_session.py`（`InferenceSession` の生成箇所を 2 ファイルに限定）と整合する。RMVPE の同型バグは未修正のまま残る。
