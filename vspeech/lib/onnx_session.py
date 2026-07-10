"""GPU を使いうる `InferenceSession` を組み立てる唯一の場所。

RVC decoder / HuBERT content encoder / RMVPE がここを通る。ここ以外で組み立てないこと。
複製すると execution provider の選択が片方でしか直らない。`tests/test_onnx_session.py`
が検査する。CPU 固定の Silero VAD (`vad.py`) だけが例外。
"""

from pathlib import Path
from typing import Any

import torch
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions


def create_session(model_file: Path, device: torch.device) -> InferenceSession:
    """`device` を尊重してセッションを開く。

    `torch.device("cuda")` は `index` が `None` になる。ORT には 0 を渡すこと。
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if device.type == "cuda" and torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(
            0,
            {
                "device_id": device.index if device.index is not None else 0,
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        )
    return InferenceSession(
        str(model_file.expanduser()),
        sess_options=sess_options,
        providers=providers,
        provider_options=providers_options,
    )
