"""The single place where a possibly-GPU `InferenceSession` is constructed.

The RVC decoder, the HuBERT content encoder and RMVPE all go through here. Never
construct one anywhere else: a duplicate means the execution-provider choice only ever
gets fixed in one of them. `tests/lib/test_onnx_session.py` enforces this. The CPU-pinned
Silero VAD (`vad.py`) is the only exception.

Whether the CUDA EP is requested comes from onnxruntime's own provider list, never from
another framework's view of the GPU (ADR-0078). A torch-shaped answer would be about a
different library's build, not about what this session can actually run on.

This file is also where the CUDA libraries the CUDA EP links against are guaranteed to be
loaded (ADR-0083). `onnxruntime-gpu`'s wheel bundles no CUDA runtime, so something has to
supply `cublas64_13.dll` / `cublasLt64_13.dll` / `cufft64_12.dll` / `cudnn64_9.dll` --
`onnxruntime_providers_cuda.dll` imports all four, and without them it fails to load and
onnxruntime falls back to `CPUExecutionProvider` in silence. They come from the pinned
`nvidia-*` wheels in the `rvc` extra. The place that opens the session is the place that
guarantees the libraries, so the ADR-0024 single factory stays single.

`preload_dlls` returns without loading anything when torch is already imported *and* built
against the same CUDA major, because torch has then loaded the same libraries itself; the
two never both load a copy. A torch built against a different CUDA major does not stop it
-- onnxruntime warns about the mismatch and loads its own.

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
from onnxruntime import cuda_version
from onnxruntime import get_available_providers
from onnxruntime import preload_dlls

from vspeech.lib.cuda_util import Device

# A file each half of `preload_dlls` needs, used to locate the directory holding it.
# onnxruntime's own loader hard-codes the per-component layout NVIDIA used up to CUDA 12
# (`nvidia/cublas/bin`, `nvidia/cufft/bin`, ...), but the CUDA 13 wheels put every CUDA
# library in one shared directory (`nvidia/cu13/bin/x86_64`) and only cuDNN -- the one
# distribution that kept its `-cu13` name -- still uses the old shape. Finding the
# directory from the file means neither layout is baked in here.
#
# cuDNN gets a version-agnostic probe because every cuDNN wheel lands in that same one
# directory. The CUDA half must not: see `_cuda_probe`.
_CUDNN_PROBE = "cudnn64_*.dll"

_cuda_libraries_lock = threading.Lock()
_cuda_libraries_loaded = False


def _cuda_probe() -> str:
    """The cuBLASLt filename that onnxruntime's own build will ask for.

    The CUDA half has to name the generation. NVIDIA gives each CUDA generation its own
    directory (`nvidia/cu13/...`), so with two co-installed, a version-agnostic glob
    would return whichever sorted first -- `cu12` ahead of `cu13`, a future `cu14`
    behind it -- rather than the one this build wants. That directory holds none of the
    files `preload_dlls` then looks for, so nothing loads and the session falls back to
    CPU, with a fail-loud message pointing nowhere near the cause.
    """
    major = (cuda_version or "").split(".")[0]
    # No CUDA in this build; the caller cannot reach here, since the CUDA EP would not
    # be in `get_available_providers()`.
    return f"cublasLt64_{major}.dll" if major else "cublasLt64_*.dll"


def _nvidia_roots() -> list[str]:
    """The directories the installed `nvidia` namespace package spans."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except ImportError:
        return []
    if spec is None or spec.submodule_search_locations is None:
        return []
    return list(spec.submodule_search_locations)


def _nvidia_wheel_dir(probe: str) -> str | None:
    """Directory inside the installed `nvidia` wheels that holds `probe`.

    `None` when the wheels are not installed, which leaves `preload_dlls` on its own
    default search (a CUDA-compatible torch's `lib`, then the legacy wheel layout, then
    the process DLL search path).
    """
    for root in _nvidia_roots():
        for hit in sorted(Path(root).rglob(probe)):
            return str(hit.parent)
    return None


def _preload_cuda_libraries() -> None:
    """Load the CUDA libraries the CUDA EP links against. Once per process (ADR-0083).

    Two calls because CUDA and cuDNN ship in different directories, and `preload_dlls`
    looks for every file it wants in the single directory it is given.

    A library it cannot find is printed, not raised -- `check_cuda_provider` is what
    turns the resulting CPU fallback into a startup error. It does raise for a
    `directory` that does not exist, which cannot happen here because each path comes
    from having just found a file inside it.
    """
    global _cuda_libraries_loaded
    with _cuda_libraries_lock:
        if _cuda_libraries_loaded:
            return
        preload_dlls(
            cuda=True,
            cudnn=False,
            msvc=True,
            directory=_nvidia_wheel_dir(_cuda_probe()),
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
