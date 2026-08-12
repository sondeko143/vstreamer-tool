"""Regression tests for how configuration gets into the pipeline (ADR-0066).

The pipeline reads its configuration from a `--config` file and from nothing
else. These pin the parts of that contract that are cheap to regress by
accident: the environment must not leak in, an unknown top-level key must still
be rejected, and a file must resolve to the values it names.
"""

import pytest
from pydantic import ValidationError

from vspeech.config import Config


def test_environment_variables_do_not_reach_the_config(monkeypatch):
    monkeypatch.setenv("vspeech_listen_port", "9999")
    monkeypatch.setenv("vspeech_transcription__enable", "true")
    monkeypatch.setenv("PORT", "9999")  # the alias that actually worked pre-ADR-0066

    config = Config()

    assert config.listen_port == 8080
    assert config.transcription.enable is False


def test_a_top_level_unknown_key_is_rejected():
    # Before ADR-0066 this came for free from SettingsConfigDict; it is now
    # stated outright on the model. A typo in config.toml must not be swallowed.
    with pytest.raises(ValidationError):
        Config.model_validate({"listen_prot": 9999})


def test_the_port_key_no_longer_sets_the_listen_port():
    # `PORT` was an alias serving the Cloud Run contract only (ADR-0067).
    with pytest.raises(ValidationError):
        Config.model_validate({"PORT": 9999})


def test_a_config_file_resolves_to_the_values_it_names(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "listen_port = 9100\n\n[recording]\nenable = true\n", encoding="utf-8"
    )

    with config_file.open("rb") as opened:
        config = Config.read_config_from_file(opened)

    assert config.listen_port == 9100
    assert config.recording.enable is True
