"""CUDA device enumeration through the driver API, without torch (ADR-0078).

`nvcuda.dll` is the driver, not the toolkit: it is present wherever the NVIDIA driver
is, which this project already requires (ADR-0028 / ADR-0039). So this adds no
dependency.

The driver API is used rather than NVML because the ordinals produced here are handed
to ctranslate2's `device_index` and to the CUDA execution provider's `device_id`, both
of which speak the CUDA ordinal space. The driver API and the runtime API honour the
same `CUDA_DEVICE_ORDER`; NVML has its own ordering and would silently select a
different GPU on a multi-GPU host.

Enumeration must never raise: `get_device` falls back to CPU when nothing is visible,
so "no driver" has to look the same as "no devices".
"""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from vspeech.logger import logger

# CUresult. Only the success value matters here: every failure is turned into an OSError
# carrying the numeric code, and the callers all fall back rather than branch on it.
_CUDA_SUCCESS = 0
# CUdevice_attribute
_CC_MAJOR = 75
_CC_MINOR = 76
# cuDeviceGetName truncates to the buffer it is given. The longest real product names
# are well under this.
_NAME_BUFFER_SIZE = 256


@dataclass(frozen=True)
class CudaDevice:
    """One CUDA device as the driver API reports it."""

    ordinal: int
    name: str
    cc_major: int
    cc_minor: int


class CudaDriver(Protocol):
    """The slice of the driver API this module needs.

    Narrow and Python-typed on purpose: the ctypes plumbing lives behind it, so the
    policy above can be exercised without an NVIDIA driver present.
    """

    def device_count(self) -> int: ...

    def device_name(self, ordinal: int) -> str: ...

    def compute_capability(self, ordinal: int) -> tuple[int, int]: ...


class _CtypesCudaDriver:
    """`CudaDriver` over `nvcuda.dll`.

    Every entry point returns a CUresult; a non-zero one becomes an OSError so that the
    policy above sees one failure type regardless of which call broke.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib

    @staticmethod
    def _check(result: int, call: str) -> None:
        if result != _CUDA_SUCCESS:
            raise OSError(f"{call} が CUresult {result} を返しました")

    def initialize(self) -> None:
        """`cuInit`. Does not create a context, so it allocates no VRAM."""
        self._check(self._lib.cuInit(0), "cuInit")

    def _handle(self, ordinal: int) -> ctypes.c_int:
        handle = ctypes.c_int()
        self._check(self._lib.cuDeviceGet(ctypes.byref(handle), ordinal), "cuDeviceGet")
        return handle

    def _attribute(self, attribute: int, handle: ctypes.c_int) -> int:
        value = ctypes.c_int()
        self._check(
            self._lib.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle),
            "cuDeviceGetAttribute",
        )
        return value.value

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self._lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return count.value

    def device_name(self, ordinal: int) -> str:
        buffer = ctypes.create_string_buffer(_NAME_BUFFER_SIZE)
        self._check(
            self._lib.cuDeviceGetName(buffer, _NAME_BUFFER_SIZE, self._handle(ordinal)),
            "cuDeviceGetName",
        )
        return buffer.value.decode(errors="replace")

    def compute_capability(self, ordinal: int) -> tuple[int, int]:
        handle = self._handle(ordinal)
        return self._attribute(_CC_MAJOR, handle), self._attribute(_CC_MINOR, handle)


def _open_nvcuda() -> ctypes.CDLL:
    """Load the driver library. Separate so tests can drive the failure path."""
    return ctypes.WinDLL("nvcuda.dll")


@lru_cache(maxsize=1)
def _load_cuda_driver() -> CudaDriver | None:
    """Open the CUDA driver, or None when it is unavailable.

    Cached: the answer cannot change while the process lives, and every worker that
    resolves a device calls in, so an uncached version would repeat the same failure
    warning at each startup path.
    """
    try:
        lib = _open_nvcuda()
    except OSError as e:
        logger.warning(
            "CUDA ドライバ (nvcuda.dll) を読み込めません。CPU で動作します: %s", e
        )
        return None
    driver = _CtypesCudaDriver(lib)
    try:
        driver.initialize()
    except OSError as e:
        logger.warning(
            "CUDA ドライバ (nvcuda.dll) を初期化できません。CPU で動作します: %s", e
        )
        return None
    return driver


def list_cuda_devices() -> list[CudaDevice]:
    """Every visible CUDA device, in ascending ordinal order.

    Returns an empty list when no driver is available.
    """
    driver = _load_cuda_driver()
    if driver is None:
        return []
    try:
        count = driver.device_count()
    except OSError as e:
        logger.warning("CUDA デバイス数を取得できません: %s", e)
        return []
    devices: list[CudaDevice] = []
    for ordinal in range(count):
        try:
            name = driver.device_name(ordinal)
            cc_major, cc_minor = driver.compute_capability(ordinal)
        except OSError as e:
            # Skip just this one. Dropping every GPU because the driver refused a single
            # query would turn a partial fault into a silent CPU run.
            logger.warning("CUDA デバイス %d の情報を取得できません: %s", ordinal, e)
            continue
        devices.append(
            CudaDevice(ordinal=ordinal, name=name, cc_major=cc_major, cc_minor=cc_minor)
        )
    return devices
