"""The single place where a possibly-GPU `InferenceSession` is constructed.

The RVC decoder, the HuBERT content encoder and RMVPE all go through here. Never
construct one anywhere else: a duplicate means the execution-provider choice only ever
gets fixed in one of them. `tests/test_onnx_session.py` enforces this. The CPU-pinned
Silero VAD (`vad.py`) is the only exception.

`log_severity` is ORT's log threshold (0=VERBOSE / 1=INFO / 2=WARNING / 3=ERROR /
4=FATAL). The default of `SessionOptions().log_severity_level` is **-1 = inherit the
Env level** (usually WARNING); leaving this argument at its default None preserves that
inheritance. Passing an explicit value pins the level for that session alone, which
then stops responding to `onnxruntime.set_default_logger_severity`. Raise it from the
call site only when a particular model emits a benign warning on every inference (it
also silences that session's other warnings, so check any diagnostic you do not want to
lose programmatically instead).
"""

from pathlib import Path
from typing import Any

import torch
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions


def create_session(
    model_file: Path, device: torch.device, log_severity: int | None = None
) -> InferenceSession:
    """Open a session honouring `device`.

    `torch.device("cuda")` has an `index` of `None`. Pass 0 to ORT in that case.
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if log_severity is not None:
        sess_options.log_severity_level = log_severity
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
