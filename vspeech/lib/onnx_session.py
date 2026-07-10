"""onnxruntime の InferenceSession を開く唯一の入口。

RVC decoder / HuBERT content encoder / RMVPE の 3 セッションがここを通る。

かつては `rvc.create_session` と `pitch_extract.create_rmvpe_session` が同じ 20 行を
重複して持っており、どちらも `torch.cuda.is_available()` だけで CUDA EP を選んで呼び出し側の
`device` を無視していた。前者だけが修正されて後者が取り残された（config で CPU を指定しても
RMVPE が GPU で走る）。実装を 1 本に畳んで、その形のドリフトを構造的に不可能にする。
"""

from pathlib import Path
from typing import Any

import torch
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions


def create_session(model_file: Path, device: torch.device) -> InferenceSession:
    """`device` を尊重してセッションを開く。

    CUDA EP を積むのは「CUDA が使えて、かつ呼び出し側が cuda device を渡した」ときだけ。
    `torch.cuda.is_available()` だけで判断すると、config が CPU を指定していても GPU で
    走ってしまう。`torch.device("cuda")` は `index` が None なので、ORT へは 0 を渡す。
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
