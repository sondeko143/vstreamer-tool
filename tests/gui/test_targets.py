from gui.paths import resolve_paths
from gui.targets import CURRENT_VERSION
from gui.targets import Target
from gui.targets import TargetList
from gui.targets import load_targets
from gui.targets import save_targets


def test_missing_file_loads_an_empty_list(tmp_path):
    targets = load_targets(resolve_paths(tmp_path))
    assert targets.targets == []
    assert targets.version == CURRENT_VERSION


def test_round_trips_through_the_config_dir(tmp_path):
    paths = resolve_paths(tmp_path / "vstreamer")
    saved = TargetList(
        targets=[
            Target(
                name="win02", host="host.example", port=8081, config_path="D:/c.toml"
            )
        ]
    )
    save_targets(paths, saved)
    assert paths.targets.exists()
    assert load_targets(paths) == saved


def test_corrupt_file_is_quarantined_and_load_still_succeeds(tmp_path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.targets.write_text("this is not = = toml", encoding="utf-8")
    targets = load_targets(paths)
    assert targets.targets == []
    backups = list(tmp_path.glob("targets.toml.*.bak"))
    assert len(backups) == 1
    # 退避は非破壊 — 元の中身が読み出せること。
    assert backups[0].read_text(encoding="utf-8") == "this is not = = toml"


def test_invalid_port_is_quarantined_too(tmp_path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.targets.write_text(
        '[[targets]]\nname = "x"\nhost = "h"\nport = 70000\n', encoding="utf-8"
    )
    assert load_targets(paths).targets == []
    assert len(list(tmp_path.glob("targets.toml.*.bak"))) == 1


def test_address_joins_host_and_port():
    assert Target(name="x", host="203.0.113.5", port=8080).address == "203.0.113.5:8080"


def test_address_brackets_an_ipv6_host():
    # 括弧が無いと最初の ":" が port 区切りとして読まれる。
    assert Target(name="x", host="::1", port=8080).address == "[::1]:8080"
    assert Target(name="x", host="[::1]", port=8080).address == "[::1]:8080"
