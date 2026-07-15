import pytest

from vspeech.exceptions import WorkerStartupError
from vspeech.exceptions import worker_startup


def test_translates_arbitrary_exception():
    with pytest.raises(WorkerStartupError) as ei:
        with worker_startup("vc"):
            raise RuntimeError("no CUDA provider")
    assert ei.value.worker == "vc"
    assert "no CUDA provider" in ei.value.detail


def test_passes_through_worker_startup_error():
    with pytest.raises(WorkerStartupError) as ei:
        with worker_startup("tts"):
            raise WorkerStartupError("voicevox", "boom")
    assert ei.value.worker == "voicevox"  # 内側を保持


def test_no_error_is_transparent():
    with worker_startup("vc"):
        x = 1 + 1
    assert x == 2
