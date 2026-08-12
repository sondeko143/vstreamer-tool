"""Enumeration of CUDA devices through the driver API (ADR-0078).

These tests must pass on a machine with no NVIDIA driver at all, so every test
here drives the loader rather than the real DLL.
"""

import logging

import pytest

from vspeech.lib import cuda_driver


class FakeDriver:
    """A driver reporting a fixed device list, in the order the API would."""

    def __init__(self, devices: list[tuple[str, int, int]]) -> None:
        self._devices = devices

    def device_count(self) -> int:
        return len(self._devices)

    def device_name(self, ordinal: int) -> str:
        return self._devices[ordinal][0]

    def compute_capability(self, ordinal: int) -> tuple[int, int]:
        _, major, minor = self._devices[ordinal]
        return major, minor


def test_no_driver_yields_no_devices(monkeypatch):
    """A machine without an NVIDIA driver enumerates nothing, and does not raise.

    This is the CPU-only path: `get_device` has to be able to fall back to CPU, so
    enumeration failing must look like "no devices", never like an error.
    """
    monkeypatch.setattr(cuda_driver, "_load_cuda_driver", lambda: None)

    assert cuda_driver.list_cuda_devices() == []


def test_devices_are_reported_in_ordinal_order(monkeypatch):
    """Ordinal N of the result is ordinal N of the driver.

    The ordinal is what reaches ctranslate2's `device_index` and the CUDA execution
    provider's `device_id`, so re-ordering the list here would silently point them at a
    different GPU than the one that was named.
    """
    driver = FakeDriver(
        [
            ("NVIDIA GeForce RTX 4060 Laptop GPU", 8, 9),
            ("NVIDIA GeForce RTX 5060 Ti", 12, 0),
        ]
    )
    monkeypatch.setattr(cuda_driver, "_load_cuda_driver", lambda: driver)

    assert cuda_driver.list_cuda_devices() == [
        cuda_driver.CudaDevice(
            ordinal=0, name="NVIDIA GeForce RTX 4060 Laptop GPU", cc_major=8, cc_minor=9
        ),
        cuda_driver.CudaDevice(
            ordinal=1, name="NVIDIA GeForce RTX 5060 Ti", cc_major=12, cc_minor=0
        ),
    ]


def test_a_driver_reporting_zero_devices_yields_no_devices(monkeypatch):
    """A driver is loadable but no GPU is visible -- also a CPU fallback, not an error."""
    monkeypatch.setattr(cuda_driver, "_load_cuda_driver", lambda: FakeDriver([]))

    assert cuda_driver.list_cuda_devices() == []


def test_a_device_that_fails_to_report_is_skipped(monkeypatch):
    """One unreadable device does not lose the others.

    Enumeration is a startup fallback path: losing every GPU because the driver
    refused one query would turn a partial fault into a silent CPU run.
    """

    class PartlyBrokenDriver(FakeDriver):
        def device_name(self, ordinal: int) -> str:
            if ordinal == 0:
                raise OSError("cuDeviceGetName failed")
            return super().device_name(ordinal)

    monkeypatch.setattr(
        cuda_driver,
        "_load_cuda_driver",
        lambda: PartlyBrokenDriver(
            [("broken", 0, 0), ("NVIDIA GeForce RTX 5060 Ti", 12, 0)]
        ),
    )

    assert cuda_driver.list_cuda_devices() == [
        cuda_driver.CudaDevice(
            ordinal=1, name="NVIDIA GeForce RTX 5060 Ti", cc_major=12, cc_minor=0
        )
    ]


@pytest.fixture
def uncached_loader():
    """Drop the loader's cached result around a test that drives the load itself."""
    cuda_driver._load_cuda_driver.cache_clear()
    yield
    cuda_driver._load_cuda_driver.cache_clear()


def test_missing_driver_is_reported_once(monkeypatch, caplog, uncached_loader):
    """A host with no NVIDIA driver logs the reason once, not once per caller.

    Every worker that resolves a device calls in, so logging per call would repeat the
    same line at each startup path for a condition that cannot change while the process
    lives.
    """

    def no_driver():
        raise OSError("cannot find nvcuda.dll")

    monkeypatch.setattr(cuda_driver, "_open_nvcuda", no_driver)

    with caplog.at_level(logging.WARNING):
        assert cuda_driver._load_cuda_driver() is None
        assert cuda_driver._load_cuda_driver() is None

    records = [r for r in caplog.records if "nvcuda" in r.getMessage()]
    assert len(records) == 1
    assert "CUDA ドライバ" in records[0].getMessage()


def test_a_driver_missing_an_export_yields_no_devices(monkeypatch, uncached_loader):
    """A loadable but incomplete nvcuda.dll is still "no devices", not a crash.

    ctypes raises AttributeError -- not OSError -- when a symbol is absent, and
    `ctypes.WinDLL` itself is an AttributeError off win32. Either escaping would kill
    the CPU fallback the whole design rests on. Stub/shim drivers shipped by remote
    desktop and virtualisation layers are the realistic trigger.
    """

    class LibMissingCuInit:
        def __getattr__(self, name):
            raise AttributeError(f"function '{name}' not found")

    monkeypatch.setattr(cuda_driver, "_open_nvcuda", lambda: LibMissingCuInit())

    assert cuda_driver._load_cuda_driver() is None
    assert cuda_driver.list_cuda_devices() == []


def test_real_driver_reports_self_consistent_devices(uncached_loader):
    """On a host that has a driver, the real ctypes path agrees with itself.

    This is the only test that exercises the ctypes plumbing. It is skipped where there
    is no driver, so it cannot gate CI, but it does catch a signature or CUresult
    mistake on any developer machine with an NVIDIA GPU.
    """
    if cuda_driver._load_cuda_driver() is None:
        pytest.skip("no CUDA driver on this host")

    devices = cuda_driver.list_cuda_devices()

    assert [d.ordinal for d in devices] == list(range(len(devices)))
    assert all(d.name for d in devices)
    assert all(d.cc_major > 0 for d in devices)
