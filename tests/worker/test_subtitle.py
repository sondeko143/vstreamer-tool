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
