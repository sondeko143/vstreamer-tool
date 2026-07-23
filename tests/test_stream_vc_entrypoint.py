import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_create_stream_vc_task_imports():
    # CPU import smoke: subsystem module must import without torch/rvc/audio extras.
    from vspeech.stream_vc.subsystem import create_stream_vc_task

    assert callable(create_stream_vc_task)


_STREAM_ENV = "VSPEECH_STREAM_VC_CONFIG"
_stream_config = os.environ.get(_STREAM_ENV)


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _cuda_available()
    or _stream_config is None
    or not Path(_stream_config).exists(),
    reason=f"CUDA / ${_STREAM_ENV} (real mic+speaker+model config) not available",
)
def test_entrypoint_boots_stream_vc():
    # エントリポイントを実際に起動する(「テストだけでなくエントリポイントを走らせる」)。
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "vspeech", "--config", str(_stream_config)],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    booted = False
    import threading

    # Enforce the 120s budget even if the child emits no stdout at all: a bare
    # readline()/iterator has no timeout, so without this a silent startup hang
    # would block indefinitely. The timer kills the child, which closes its
    # stdout pipe -> the `for line in proc.stdout` loop ends at EOF and we
    # assert cleanly (booted stays False) instead of hanging.
    killer = threading.Timer(120.0, proc.kill)
    killer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if "stream vc worker started" in line:
                booted = True
                break
    finally:
        killer.cancel()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert booted, "entrypoint did not reach 'stream vc worker started'"
