"""Device resolution policy, without torch (ADR-0078 / ADR-0079).

Every test drives the enumeration, so these run on a host with no NVIDIA driver.
"""

import pytest

from vspeech.exceptions import GpuNotFoundError
from vspeech.lib import cuda_util
from vspeech.lib.cuda_driver import CudaDevice

RTX_4060 = CudaDevice(
    ordinal=0, name="NVIDIA GeForce RTX 4060 Laptop GPU", cc_major=8, cc_minor=9
)
RTX_5060TI = CudaDevice(
    ordinal=1, name="NVIDIA GeForce RTX 5060 Ti", cc_major=12, cc_minor=0
)


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(cuda_util, "list_cuda_devices", lambda: [RTX_4060, RTX_5060TI])


@pytest.fixture
def no_gpus(monkeypatch):
    monkeypatch.setattr(cuda_util, "list_cuda_devices", lambda: [])


@pytest.mark.parametrize(
    ("cc_major", "cc_minor", "expected"),
    [
        (5, 2, False),  # Maxwell: no native fp16
        (6, 0, True),  # GP100: 2:1 fp16
        (6, 1, False),  # GTX 10-series: fp16 at 1/64 rate
        (6, 2, False),  # GP10x again
        (7, 0, True),  # V100
        (7, 5, True),  # Turing, incl. GTX 16xx (the one case that flips vs. the old
        # product-name blacklist -- TU11x has no tensor cores but does run fp16 at 2:1)
        (8, 9, True),  # RTX 4060 (Ada)
        (12, 0, True),  # RTX 5060 Ti (Blackwell)
    ],
)
def test_fp16_support_follows_compute_capability(cc_major, cc_minor, expected):
    """ADR-0079: the hardware property decides, not the product name."""
    assert cuda_util.supports_fp16(cc_major, cc_minor) is expected


def test_cpu_device_has_no_half_precision(no_gpus):
    """A CPU device is fp16-incapable by definition.

    The previous implementation reached this only by accident: it passed `index=None`
    into torch, caught the resulting exception, and returned False.
    """
    assert cuda_util.half_precision_available(cuda_util.Device("cpu")) is False


def test_half_precision_reads_the_capability_of_the_resolved_ordinal(two_gpus):
    """The answer is about the selected GPU, not about the first one."""
    assert cuda_util.half_precision_available(cuda_util.Device("cuda", 1)) is True


def test_half_precision_is_false_for_an_ordinal_that_is_not_visible(two_gpus):
    assert cuda_util.half_precision_available(cuda_util.Device("cuda", 7)) is False


def test_device_renders_like_the_torch_device_it_replaces():
    """The startup log prints the device with %s; the text must not change.

    `... worker device: cuda:0, <name>` is the line each host's migration is checked
    against, so its formatting is part of the contract.
    """
    assert str(cuda_util.Device("cuda", 0)) == "cuda:0"
    assert str(cuda_util.Device("cuda", 1)) == "cuda:1"
    assert str(cuda_util.Device("cpu")) == "cpu"


def test_gpu_id_zero_is_a_real_device(two_gpus):
    """`gpu_id = 0` means cuda:0, not "unset".

    "Unset" is `None` (`gpu_id: int | None = None`). Treating 0 as falsy would drop the
    `gpu_id = 0` configuration in config.toml.example down to CPU, which then makes
    `check_cuda_provider` fail at vc worker startup.
    """
    device, name = cuda_util.get_device(0, "")

    assert device == cuda_util.Device("cuda", 0)
    assert name == "NVIDIA GeForce RTX 4060 Laptop GPU"


def test_no_gpu_settings_resolve_to_cpu(two_gpus):
    device, name = cuda_util.get_device(None, "")

    assert device == cuda_util.Device("cpu")
    assert name == "cpu"


def test_gpu_id_wins_over_gpu_name(two_gpus):
    """Both set: `gpu_id` is checked first. Preserved from the previous implementation."""
    device, _ = cuda_util.get_device(1, "RTX 4060")

    assert device == cuda_util.Device("cuda", 1)


def test_gpu_name_selects_by_substring(two_gpus):
    device, name = cuda_util.get_device(None, "5060 Ti")

    assert device == cuda_util.Device("cuda", 1)
    assert name == "NVIDIA GeForce RTX 5060 Ti"


def test_a_gpu_name_matching_nothing_fails_loudly(two_gpus):
    """A `gpu_name` typo must say so, and name what was actually visible.

    The previous implementation returned `(0, None)` from its name lookup and then read
    `.name` off that None, so a typo surfaced as an AttributeError and the worker
    silently failed to start. Falling back to CPU instead was rejected: whisper and RVC
    on CPU are slow enough to look like a hang rather than a misconfiguration.
    """
    with pytest.raises(GpuNotFoundError) as excinfo:
        cuda_util.get_device(None, "RTX 9090")

    message = str(excinfo.value)
    assert "RTX 9090" in message
    assert "NVIDIA GeForce RTX 4060 Laptop GPU" in message
    assert "NVIDIA GeForce RTX 5060 Ti" in message


def test_a_gpu_id_outside_the_visible_range_fails_loudly(two_gpus):
    """Same reasoning as the name case: an out-of-range `gpu_id` is a typo, not CPU.

    The previous implementation let torch raise here; the outcome (the worker does not
    start) is kept, with a message that says which ordinals exist.
    """
    with pytest.raises(GpuNotFoundError) as excinfo:
        cuda_util.get_device(5, "")

    assert "5" in str(excinfo.value)
    assert "NVIDIA GeForce RTX 5060 Ti" in str(excinfo.value)


def test_settings_asking_for_a_gpu_fall_back_to_cpu_when_none_is_visible(no_gpus):
    """A CPU-only host must still start. ADR-0078: enumeration failure looks like CPU."""
    by_id, _ = cuda_util.get_device(0, "")
    by_name, _ = cuda_util.get_device(None, "RTX 4060")

    assert by_id == cuda_util.Device("cpu")
    assert by_name == cuda_util.Device("cpu")
