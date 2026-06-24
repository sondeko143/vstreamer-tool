from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402


def test_normalize_path_converts_backslashes():
    assert metrics.normalize_path("vspeech\\lib\\command.py") == "vspeech/lib/command.py"


def test_simple_name_strips_class_prefix():
    assert metrics.simple_name("RVCModel::infer") == "infer"
    assert metrics.simple_name("process_command") == "process_command"


def test_derive_targets_prefers_build_backend_module_name():
    pyproject = {
        "project": {"name": "voicerecog"},
        "tool": {"uv": {"build-backend": {"module-name": ["vspeech"]}}},
    }
    targets = metrics.derive_targets(pyproject)
    assert targets.packages == ["vspeech"]


def test_derive_targets_falls_back_to_normalized_name():
    pyproject = {"project": {"name": "my-app"}}
    targets = metrics.derive_targets(pyproject)
    assert targets.packages == ["my_app"]
