from pathlib import Path

from gui.migration import Migration
from gui.migration import migrate_dict
from gui.migration import quarantine


def test_migrate_dict_applies_only_newer_steps():
    m2 = Migration(2, "add b", lambda d: {**d, "b": 2})
    m3 = Migration(3, "add c", lambda d: {**d, "c": 3})
    out, ver = migrate_dict({"a": 1}, from_version=1, migrations=[m3, m2], current=3)
    assert out == {"a": 1, "b": 2, "c": 3}
    assert ver == 3


def test_migrate_dict_skips_already_applied():
    m2 = Migration(2, "add b", lambda d: {**d, "b": 2})
    m3 = Migration(3, "add c", lambda d: {**d, "c": 3})
    out, _ = migrate_dict({"a": 1}, from_version=2, migrations=[m2, m3], current=3)
    assert out == {"a": 1, "c": 3}


def test_migrate_dict_empty_chain_is_identity():
    out, ver = migrate_dict({"a": 1}, from_version=0, migrations=[], current=1)
    assert out == {"a": 1}
    assert ver == 1


def test_quarantine_is_non_destructive_and_increments(tmp_path: Path):
    p = tmp_path / "x.toml"
    p.write_text("first", encoding="utf-8")
    b1 = quarantine(p)
    assert b1.name == "x.toml.bak-1"
    assert b1.read_text(encoding="utf-8") == "first"
    assert p.read_text(encoding="utf-8") == "first"
    p.write_text("second", encoding="utf-8")
    b2 = quarantine(p)
    assert b2.name == "x.toml.bak-2"
    assert b2.read_text(encoding="utf-8") == "second"
