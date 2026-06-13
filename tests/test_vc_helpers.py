import pytest

from vspeech.worker.vc import check_cuda_provider


def test_check_cuda_provider_passes_when_cuda_present():
    # Should not raise.
    check_cuda_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_check_cuda_provider_raises_when_only_cpu():
    with pytest.raises(RuntimeError) as excinfo:
        check_cuda_provider(["CPUExecutionProvider"])
    # Message should be actionable about the missing GPU runtime.
    assert "CUDAExecutionProvider" in str(excinfo.value)
