from pathlib import Path

from gui.paths import resolve_paths
from gui.profile import PipelineEntry
from gui.profile import Profile
from gui.profile import load_default_config
from gui.profile import load_pipeline_config
from gui.profile import load_profile
from gui.profile import save_pipeline_config
from gui.profile import save_profile
from vspeech.config import Config


def test_default_config_created_when_missing(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    config = load_default_config(paths)
    assert isinstance(config, Config)
    assert paths.default_config.exists()


def test_default_config_corrupt_falls_back(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.default_config.write_text("this = = not valid toml", encoding="utf-8")
    config = load_default_config(paths)
    assert isinstance(config, Config)
    assert (tmp_path / "default.toml.bak-1").exists()


def test_profile_roundtrip(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    profile = Profile(
        pipelines=[PipelineEntry(id="ab12", name="p1", port=8080, recipe="mic_vc")]
    )
    save_profile(paths, profile)
    loaded = load_profile(paths)
    assert loaded.pipelines[0].id == "ab12"
    assert loaded.pipelines[0].port == 8080
    assert loaded.pipelines[0].recipe == "mic_vc"


def test_profile_missing_is_empty(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    assert load_profile(paths).pipelines == []


def test_profile_corrupt_falls_back_to_empty(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text("= broken", encoding="utf-8")
    assert load_profile(paths).pipelines == []
    assert (tmp_path / "pipelines.toml.bak-1").exists()


def test_pipeline_config_roundtrip_injects_port(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    entry = PipelineEntry(id="ab12", name="p1", port=8085, recipe="blank")
    save_pipeline_config(paths, entry, Config())
    result = load_pipeline_config(paths, entry)
    assert result.ok is True
    assert result.value is not None
    assert result.value.listen_port == 8085


def test_pipeline_config_corrupt_surfaces_raw(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.pipelines_dir.mkdir(parents=True, exist_ok=True)
    entry = PipelineEntry(id="ab12", name="p1", port=8085, recipe="blank")
    paths.pipeline_config("ab12").write_text("listen_port = = broken", encoding="utf-8")
    result = load_pipeline_config(paths, entry)
    assert result.ok is False
    assert result.raw_text == "listen_port = = broken"
    assert result.quarantined_path is not None
    assert result.quarantined_path.exists()
