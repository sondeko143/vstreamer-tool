"""The subtitle worker_type dispatcher (ADR-0040).

Only the config contract is left here: which backend a config selects, and that the
selection survives a save/load. The backends themselves are in `test_subtitle_obs.py` and
`test_subtitle_tk_*.py`.
"""

import io

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


# `test_obs_password_survives_a_toml_round_trip` used to be here: it set
# subtitle.obs.password and checked it came back out of export_to_toml() as plaintext.
# It was subsumed 17 minutes after it was written, by
# tests/config/test_secret.py::test_every_secret_str_field_survives_export_to_toml, which
# walks Config's schema for SecretStr fields and covers subtitle.obs.password among them
# -- a structural gate that also catches the *next* secret nobody remembers to add here.
# ADR-0088 removed the narrow copy; it survived this long only because it sat under a file
# name nobody searching for "secret" would open.
