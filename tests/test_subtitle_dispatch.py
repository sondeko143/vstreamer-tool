import io

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


# ADR-0040's headless invariant -- that importing the subtitle dispatcher pulls in no GUI
# toolkit -- used to be asserted here by naming tkinter and checking it was absent from
# `sys.modules`. ADR-0087 took every assertion of that shape out of this repo, and this one
# was redundant besides: `tests/test_runtime_footprint.py` measures a `subtitle_obs` path
# whose imports go *through* the dispatcher, and its recorded module set contains no GUI
# toolkit. A toolkit arriving through the dispatcher therefore fails that gate, which names
# it as an unlisted newcomer without anything here having to forbid it in advance.


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
