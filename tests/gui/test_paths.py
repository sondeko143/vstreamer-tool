from pathlib import Path

from gui.paths import resolve_paths


def test_resolve_paths_layout(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    assert paths.root == tmp_path
    assert paths.default_config == tmp_path / "default.toml"
    assert paths.manifest == tmp_path / "pipelines.toml"
    assert paths.pipelines_dir == tmp_path / "pipelines"
    assert paths.pipeline_config("ab12") == tmp_path / "pipelines" / "ab12.toml"


def test_default_root_used_when_none(monkeypatch):
    import gui.paths as paths_mod

    monkeypatch.setattr(paths_mod, "default_root", lambda: Path("/fake/root"))
    paths = resolve_paths(None)
    assert paths.root == Path("/fake/root")
