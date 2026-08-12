"""Which device to run on, and whether fp16 is usable there (ADR-0078 / ADR-0079).

torch-free on purpose. `Device` keeps the `type` / `index` attributes and the `str()`
that `torch.device` presented, which is what let the RVC conversion path convert to a
real `torch.device` at its boundary while it still needed one (ADR-0081 removed that
last conversion) and what keeps the startup log line unchanged. `Device` is now what
`create_session` and the conversion path both speak end to end.
"""

from dataclasses import dataclass
from typing import Literal

from vspeech.exceptions import GpuNotFoundError
from vspeech.lib.cuda_driver import list_cuda_devices

# Compute capability at or above which fp16 runs at least as fast as fp32 (Volta
# onwards, where the tensor cores are). See ADR-0079 for why this replaced a product
# name blacklist.
_TENSOR_CORE_CC = (7, 0)
# GP100 is the one pre-Volta part with 2:1 fp16. Every other Pascal (6.1 / 6.2) runs it
# at 1/64 and is unusable.
_GP100_CC = (6, 0)


@dataclass(frozen=True)
class Device:
    """A compute device, in the shape `torch.device` presented."""

    type: Literal["cuda", "cpu"]
    index: int | None = None

    def __str__(self) -> str:
        if self.type == "cuda" and self.index is not None:
            return f"cuda:{self.index}"
        return self.type


def supports_fp16(cc_major: int, cc_minor: int) -> bool:
    """Whether a device of this compute capability runs fp16 at a usable rate."""
    return (cc_major, cc_minor) >= _TENSOR_CORE_CC or (cc_major, cc_minor) == _GP100_CC


def half_precision_available(device: Device) -> bool:
    """Whether fp16 should be used on `device`."""
    if device.type != "cuda":
        return False
    ordinal = device.index if device.index is not None else 0
    for cuda_device in list_cuda_devices():
        if cuda_device.ordinal == ordinal:
            return supports_fp16(cuda_device.cc_major, cuda_device.cc_minor)
    return False


def require_cuda_ordinal(device: Device, purpose: str) -> int:
    """The CUDA ordinal of `device`, for callers that only run on CUDA.

    Some backends (whisper via ctranslate2) hardcode a CUDA device and take the ordinal
    separately. Handing them 0 for a CPU-resolved device would run on GPU 0 while the
    startup log said `cpu`. Fail instead, naming the settings that decide this.
    """
    if device.type != "cuda":
        raise GpuNotFoundError(
            f"{purpose} は GPU が必要ですが、デバイスが {device} に解決されました。"
            " gpu_id か gpu_name を設定してください。"
        )
    return device.index if device.index is not None else 0


def get_device(gpu_id: int | None, gpu_name: str) -> tuple[Device, str]:
    """Resolve the configured GPU, or CPU.

    `gpu_id is not None`, not `gpu_id`. 0 is a valid device number; "unset" is None
    (`gpu_id: int | None = None` in config.py). When both are set, gpu_id wins over
    gpu_name (it is checked first). That is deliberate but untested -- in practice
    only one of the two is ever set, so it does no harm.

    No CUDA device visible means CPU, whatever the config asked for: a host without a
    driver has to start. A `gpu_name` that matches none of the devices that *are*
    visible is a different matter and raises (ADR-0078).
    """
    devices = list_cuda_devices()
    if not devices:
        return Device("cpu"), "cpu"
    if gpu_id is not None:
        for device in devices:
            if device.ordinal == gpu_id:
                return Device("cuda", gpu_id), device.name
        raise GpuNotFoundError(
            f"gpu_id {gpu_id} のデバイスがありません。"
            f"見えている GPU: {_describe(devices)}"
        )
    if gpu_name:
        for device in devices:
            if gpu_name in device.name:
                return Device("cuda", device.ordinal), device.name
        raise GpuNotFoundError(
            f"gpu_name '{gpu_name}' に一致する GPU がありません。"
            f"見えている GPU: {_describe(devices)}"
        )
    return Device("cpu"), "cpu"


def _describe(devices: list) -> str:
    return ", ".join(f"{d.ordinal}={d.name}" for d in devices)
