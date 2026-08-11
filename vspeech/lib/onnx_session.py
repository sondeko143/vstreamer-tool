"""The single place where a possibly-GPU `InferenceSession` is constructed.

The RVC decoder, the HuBERT content encoder and RMVPE all go through here. Never
construct one anywhere else: a duplicate means the execution-provider choice only ever
gets fixed in one of them. `tests/test_onnx_session.py` enforces this. The CPU-pinned
Silero VAD (`vad.py`) is the only exception.

Whether the CUDA EP is requested comes from onnxruntime's own provider list, never from
another framework's view of the GPU (ADR-0078). A torch-shaped answer would be about a
different library's build, not about what this session can actually run on.

One inherited invariant is worth writing down, because ADR-0078 removed the `import
torch` that used to make it automatic: `onnxruntime-gpu` pulls in no `nvidia-*-cu13`
wheels, so the CUDA EP's cuBLAS/cuDNN come from torch's `torch/lib` (torch's import adds
that directory to the DLL search path). Every caller that opens a CUDA session today
reaches `vspeech.lib.rvc` first, which imports torch at module level, so the ordering
holds -- but it now holds by accident rather than by construction. A future CUDA session
opened from a path that never touches `rvc` (moving the transcription VAD off
`CPUExecutionProvider`, say) would silently fall back to CPU with no torch to supply the
libraries. Load torch first from that path, or add an explicit nvidia-cu13 dependency.

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

from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from onnxruntime import get_available_providers

from vspeech.lib.cuda_util import Device


def create_session(
    model_file: Path, device: Device, log_severity: int | None = None
) -> InferenceSession:
    """Open a session honouring `device`.

    `Device("cuda")` has an `index` of `None`. Pass 0 to ORT in that case.
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if log_severity is not None:
        sess_options.log_severity_level = log_severity
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if device.type == "cuda" and "CUDAExecutionProvider" in get_available_providers():
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
