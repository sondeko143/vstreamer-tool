from pathlib import Path

from gui.shared_paths import SHARED_ASSET_FIELDS
from gui.shared_paths import apply_shared
from gui.shared_paths import shared_values
from vspeech.config import Config
from vspeech.config import RvcConfig


def test_shared_values_reads_every_shared_field():
    config = Config(rvc=RvcConfig(model_file=Path("m.pth")))
    values = shared_values(config)
    assert set(values) == set(SHARED_ASSET_FIELDS)
    assert values["rvc.model_file"] == Path("m.pth")


def test_apply_shared_copies_and_reports_changed_fields():
    source = Config(rvc=RvcConfig(model_file=Path("new.pth")))
    target = Config(rvc=RvcConfig(model_file=Path("old.pth")))
    changed = apply_shared(source, target)
    assert changed == ["rvc.model_file"]
    assert target.rvc.model_file == Path("new.pth")


def test_apply_shared_is_noop_when_already_equal():
    source = Config(rvc=RvcConfig(model_file=Path("same.pth")))
    target = Config(rvc=RvcConfig(model_file=Path("same.pth")))
    assert apply_shared(source, target) == []


def test_apply_shared_does_not_touch_non_shared_fields():
    source = Config(rvc=RvcConfig(model_file=Path("m.pth"), f0_up_key=12))
    target = Config(rvc=RvcConfig(model_file=Path("old.pth"), f0_up_key=-3))
    apply_shared(source, target)
    assert target.rvc.f0_up_key == -3  # 調整つまみは pipeline 固有なので保つ
