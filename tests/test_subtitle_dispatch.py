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
