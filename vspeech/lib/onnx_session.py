"""GPU を使いうる `InferenceSession` を組み立てる唯一の場所。

RVC decoder / HuBERT content encoder / RMVPE がここを通る。ここ以外で組み立てないこと。
複製すると execution provider の選択が片方でしか直らない。`tests/test_onnx_session.py`
が検査する。CPU 固定の Silero VAD (`vad.py`) だけが例外。

`log_severity` は ORT のログ閾値 (0=VERBOSE / 1=INFO / 2=WARNING / 3=ERROR / 4=FATAL)。
`SessionOptions().log_severity_level` の既定は **-1 = Env のレベルを継承** (通常は
WARNING)。この引数を既定 None のままにすればその継承が保たれる。明示値を渡すと
そのセッションだけレベルが固定され、`onnxruntime.set_default_logger_severity` の
影響を受けなくなる。特定モデルが良性の警告を毎推論吐く場合だけ呼び出し側で上げる
(そのセッションの他の警告も道連れに消えるので、消したくない診断は別途
プログラム的に検査すること)。
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
    """`device` を尊重してセッションを開く。

    `torch.device("cuda")` は `index` が `None` になる。ORT には 0 を渡すこと。
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
