"""The single place where a possibly-GPU `InferenceSession` is constructed.

The RVC decoder, the HuBERT content encoder and RMVPE all go through here. Never
construct one anywhere else: a duplicate means the execution-provider choice only ever
gets fixed in one of them. `tests/test_onnx_session.py` enforces this. The CPU-pinned
Silero VAD (`vad.py`) is the only exception.

Whether the CUDA EP is requested comes from onnxruntime's own provider list, never from
another framework's view of the GPU (ADR-0078). A torch-shaped answer would be about a
different library's build, not about what this session can actually run on.

This file is also where the CUDA libraries the CUDA EP links against are guaranteed to be
loaded (ADR-0083). `onnxruntime-gpu`'s wheel bundles no CUDA runtime, so something has to
supply `cublas64_13.dll` / `cublasLt64_13.dll` / `cufft64_12.dll` / `cudnn64_9.dll` --
`onnxruntime_providers_cuda.dll` imports all four, and without them it fails to load and
onnxruntime falls back to `CPUExecutionProvider` in silence. That supplier used to be
`torch/lib`, by way of the `import torch` that ADR-0078 removed; it is now the pinned
`nvidia-*` wheels in the `rvc` extra. The place that opens the session is the place that
guarantees the libraries, so the ADR-0024 single factory stays single.

`preload_dlls` returns without loading anything when torch is already imported (torch has
loaded the same libraries itself), so the two never both load a copy.

`log_severity` is ORT's log threshold (0=VERBOSE / 1=INFO / 2=WARNING / 3=ERROR /
4=FATAL). The default of `SessionOptions().log_severity_level` is **-1 = inherit the
Env level** (usually WARNING); leaving this argument at its default None preserves that
inheritance. Passing an explicit value pins the level for that session alone, which
then stops responding to `onnxruntime.set_default_logger_severity`. Raise it from the
call site only when a particular model emits a benign warning on every inference (it
also silences that session's other warnings, so check any diagnostic you do not want to
lose programmatically instead).
"""

import importlib.util
import threading
from pathlib import Path
from typing import Any

from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from onnxruntime import get_available_providers
from onnxruntime import preload_dlls

from vspeech.lib.cuda_util import Device

# A file each half of `preload_dlls` needs, used to locate the directory holding it.
# onnxruntime's own loader hard-codes the per-component layout NVIDIA used up to CUDA 12
# (`nvidia/cublas/bin`, `nvidia/cufft/bin`, ...), but the CUDA 13 wheels put every CUDA
# library in one shared directory (`nvidia/cu13/bin/x86_64`) and only cuDNN -- the one
# distribution that kept its `-cu13` name -- still uses the old shape. Finding the
# directory from the file means neither layout is baked in here.
_CUDA_PROBE = "cublasLt64_*.dll"
_CUDNN_PROBE = "cudnn64_*.dll"

_cuda_libraries_lock = threading.Lock()
_cuda_libraries_loaded = False


def _nvidia_wheel_dir(probe: str) -> str | None:
    """Directory inside the installed `nvidia` wheels that holds `probe`.

    `None` when the wheels are not installed, which leaves `preload_dlls` on its own
    default search (a CUDA-compatible torch's `lib`, then the legacy wheel layout, then
    the process DLL search path).
    """
    try:
        spec = importlib.util.find_spec("nvidia")
    except ImportError:
        return None
    if spec is None or spec.submodule_search_locations is None:
        return None
    for root in spec.submodule_search_locations:
        for hit in sorted(Path(root).rglob(probe)):
            return str(hit.parent)
    return None


def _preload_cuda_libraries() -> None:
    """Load the CUDA libraries the CUDA EP links against. Once per process (ADR-0083).

    Two calls because CUDA and cuDNN ship in different directories, and `preload_dlls`
    looks for every file it wants in the single directory it is given. Failures only
    print -- `check_cuda_provider` is what turns a CPU fallback into a startup error.
    """
    global _cuda_libraries_loaded
    with _cuda_libraries_lock:
        if _cuda_libraries_loaded:
            return
        preload_dlls(
            cuda=True, cudnn=False, msvc=True, directory=_nvidia_wheel_dir(_CUDA_PROBE)
        )
        preload_dlls(
            cuda=False,
            cudnn=True,
            msvc=False,
            directory=_nvidia_wheel_dir(_CUDNN_PROBE),
        )
        _cuda_libraries_loaded = True


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
        _preload_cuda_libraries()
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
