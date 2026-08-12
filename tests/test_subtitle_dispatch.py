import io
import sys

from pydantic import SecretStr

from vspeech.config import Config
from vspeech.config import SubtitleWorkerType


def test_subtitle_worker_type_defaults_to_tk():
    # Does not change the behaviour of an existing config (worker_type unspecified).
    assert Config().subtitle.worker_type == SubtitleWorkerType.TK


def test_subtitle_worker_type_round_trips_through_toml():
    config = Config()
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    toml_bytes = config.export_to_toml().encode("utf-8")
    toml_file = io.BytesIO(toml_bytes)
    toml_file.name = "config.toml"
    reloaded = Config.read_config_from_file(toml_file)
    assert reloaded.subtitle.worker_type == SubtitleWorkerType.OBS


def test_importing_the_subtitle_dispatcher_does_not_import_tkinter():
    """The crux of the headless goal: tkinter must not be pulled in through the dispatcher
    (ADR-0040).

    tkinter is stdlib, so it may already be loaded via another path. What we want to check
    is not "subtitle does not depend on tkinter" but "importing subtitle does not newly
    load tkinter", so it is dropped first and then verified.

    [Open, deferred 2026-08-13 -- outside ADR-0087's scope] This does name a module in
    order to assert its absence, which ADR-0087 removed from every *weight* claim in this
    repo. It was left because it is not a weight claim: ADR-0040 split the back ends so an
    OBS pipeline runs on a host with no GUI toolkit available at all, which is a
    module-boundary constraint about the dispatcher, of the same kind as
    tests/test_onnx_session.py's single-session-factory rule -- and ADR-0087's decision
    covers the paths that were asserted framework-free for weight, not this. The weight
    half is covered independently now: tests/test_runtime_footprint.py measures a
    `subtitle_obs` path (dispatcher + OBS back end) whose recorded module set contains no
    GUI toolkit, so a toolkit arriving there fires on cost without being named. Whether
    this assertion should follow is ADR-0040's call to make, not this one's.
    """
    for name in list(sys.modules):
        if name == "tkinter" or name.startswith("tkinter."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("vspeech.worker.subtitle"):
            del sys.modules[name]

    import vspeech.worker.subtitle  # noqa: F401

    assert "tkinter" not in sys.modules


def test_obs_password_survives_a_toml_round_trip():
    """export_to_toml expands SecretStr from a hard-coded list. Add a new secret without
    adding it here and the saving path corrupts the config."""
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.password = SecretStr("hunter2")
    dumped = config.export_to_toml()
    assert "hunter2" in dumped
    assert "**" not in dumped
    reloaded = Config.read_config_from_file(
        _named_bytes_io(dumped.encode("utf-8"), "config.toml")
    )
    assert reloaded.subtitle.obs.password.get_secret_value() == "hunter2"


def _named_bytes_io(data: bytes, name: str):
    buf = io.BytesIO(data)
    buf.name = name
    return buf
