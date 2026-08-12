"""Fixtures and asset gates shared by the whole suite.

Two things live here, both because they were being copied instead:

- **Global-singleton hygiene.** `vspeech.lib.telemetry.telemetry` is process-wide, and 52
  hand-written `reset()` / `configure()` pairs across 11 files used to stand in for a
  fixture. `_reset_telemetry` below is autouse, so a test that forgets is no longer a
  test-order bug waiting for the next file (ADR-0088).
- **Asset and hardware gates.** GPU/model/config requirements are declared as
  `requires_*` markers registered in `pyproject.toml`, and evaluated here in one place
  rather than as an ad-hoc `skipif` per file (ADR-0089). `uv run pytest --markers` lists
  them; `uv run pytest -m requires_cuda` selects them.

Pure builders that are called at module level (PCM, packets) are not fixtures and live in
`tests/pcm.py` instead.
"""

import importlib.util
import os
from collections.abc import Callable
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from vspeech.lib.telemetry import telemetry

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# asset / hardware gates (ADR-0089)
# --------------------------------------------------------------------------------------


def _env_path(name: str) -> Path | None:
    """The path `$name` points at, or None if unset or missing on disk."""
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


@cache
def cuda_available() -> bool:
    """Whether the CUDA driver enumerates at least one device.

    Cached: the gate is asked once per gated test, and the answer cannot change inside a
    run. Imported lazily so a CPU-only run never touches nvcuda.dll.
    """
    from vspeech.lib.cuda_driver import list_cuda_devices

    return bool(list_cuda_devices())


def _needs_cuda() -> str | None:
    if cuda_available():
        return None
    return "no CUDA device (the CUDA driver enumerates none)"


def _needs_env(name: str, what: str, how: str) -> str | None:
    if _env_path(name) is not None:
        return None
    return f"${name} is unset or does not exist -- point it at {what} ({how})"


def _needs_hubert_golden(filename: str) -> str | None:
    root = _env_path("VSPEECH_HUBERT_GOLDEN_DIR")
    if root is not None and (root / filename).exists():
        return None
    return (
        f"$VSPEECH_HUBERT_GOLDEN_DIR/{filename} not available "
        "(uv run poe export-hubert-onnx)"
    )


def _needs_rvc_golden_npz() -> str | None:
    npz = REPO_ROOT / "tests" / "assets" / "rvc_golden" / "change_voice_golden.npz"
    if npz.exists():
        return None
    return (
        f"{npz.relative_to(REPO_ROOT).as_posix()} not captured "
        "(scripts/capture_change_voice_golden.py)"
    )


def _needs_torch() -> str | None:
    if importlib.util.find_spec("torch") is not None:
        return None
    return (
        "torch is an offline-tool-only dependency (ADR-0080); it comes from the export "
        "task's `uv run --with` overlay, not from pyproject"
    )


#: marker name -> a check returning None when satisfied, or the skip reason.
#: Every key must also be registered in pyproject.toml's `[tool.pytest.ini_options]
#: markers`, or it will not show up in `pytest --markers`.
REQUIREMENTS: dict[str, Callable[[], str | None]] = {
    "requires_cuda": _needs_cuda,
    "requires_rvc_config": lambda: _needs_env(
        "VSPEECH_RVC_GOLDEN_CONFIG", "an RVC worker config TOML", "machine-specific"
    ),
    "requires_rvc_golden": _needs_rvc_golden_npz,
    "requires_hubert_assets": lambda: (
        _needs_env(
            "VSPEECH_HUBERT_ASSET_DIR",
            "the converted HuBERT asset directory",
            "uv run poe convert-hubert",
        )
        or _needs_hubert_golden("hubert_golden.npz")
    ),
    "requires_hubert_fp16_golden": lambda: _needs_hubert_golden(
        "hubert_golden_fp16.npz"
    ),
    "requires_fcpe_onnx": lambda: _needs_env(
        "VSPEECH_FCPE_ONNX", "fcpe.onnx", "uv run poe export-fcpe-onnx"
    ),
    "requires_vad_model": lambda: _needs_env(
        "VSPEECH_VAD_MODEL", "silero_vad.onnx", "machine-specific"
    ),
    "requires_stream_vc_config": lambda: _needs_env(
        "VSPEECH_STREAM_VC_CONFIG",
        "a real mic+speaker+model stream_vc config",
        "machine-specific",
    ),
    "requires_torch": _needs_torch,
}


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test whose declared `requires_*` markers are not all satisfied.

    The first unmet requirement wins, so a test marked `requires_cuda` *and*
    `requires_rvc_config` says which of the two is missing rather than reporting the pair.
    """
    for marker in item.iter_markers():
        check = REQUIREMENTS.get(marker.name)
        if check is None:
            continue
        reason = check()
        if reason:
            pytest.skip(reason)


# --------------------------------------------------------------------------------------
# global singletons
# --------------------------------------------------------------------------------------


def _pristine_telemetry() -> None:
    """The state `Telemetry.__init__` leaves behind: off, 5000 samples, nothing recorded."""
    telemetry.reset()
    telemetry.configure(enabled=False, max_samples=5000)


@pytest.fixture(autouse=True)
def _reset_telemetry() -> Iterator[None]:
    _pristine_telemetry()
    yield
    _pristine_telemetry()


@pytest.fixture
def enabled_telemetry(_reset_telemetry: None):
    """The process-wide telemetry, recording, with the surrounding reset already done."""
    telemetry.configure(enabled=True, max_samples=1000)
    return telemetry


# --------------------------------------------------------------------------------------
# sounddevice stubs
# --------------------------------------------------------------------------------------


@pytest.fixture
def stub_device_table(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]:
    """Install a fake `sd.query_devices` / `sd.query_hostapis` pair.

    The tables stay with the caller: an input-side file and an output-side file describe
    different endpoints, and the shared part was only ever the lookup.
    """

    def install(devices: list[dict[str, Any]], hostapis: list[dict[str, Any]]) -> None:
        import vspeech.lib.audio as audio

        def _query_devices(index: int | None = None):
            if index is None:
                return devices
            return next(d for d in devices if d["index"] == index)

        monkeypatch.setattr(audio.sd, "query_devices", _query_devices)
        monkeypatch.setattr(audio.sd, "query_hostapis", lambda: hostapis)

    return install


@pytest.fixture
def record_opened_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any, str, Any], list[Any]]:
    """Replace `<module>.sd.<attr>` with a constructor spy; return the list it fills.

    Each caller passes its own stand-in stream class, because what a test asserts about an
    opened stream (frames written, close order, start failures) differs per boundary.
    """

    def install(module: Any, attr: str, stream_cls: Any) -> list[Any]:
        opened: list[Any] = []

        def _open(**kwargs: Any) -> Any:
            stream = stream_cls(**kwargs)
            opened.append(stream)
            return stream

        monkeypatch.setattr(module.sd, attr, _open)
        return opened

    return install
