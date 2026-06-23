from tools.python_health.scripts import health


def test_derive_targets_from_module_name():
    pyproject = {
        "project": {"name": "voicerecog"},
        "tool": {"uv": {"build-backend": {"module-name": ["vspeech"]}}},
    }
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["vspeech"]
    assert targets.project_name == "voicerecog"


def test_derive_targets_falls_back_to_script_entrypoint():
    pyproject = {"project": {"name": "foo", "scripts": {"foo": "foo_pkg.main:cmd"}}}
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["foo_pkg"]


def test_derive_targets_falls_back_to_normalized_project_name():
    pyproject = {"project": {"name": "my-tool"}}
    targets = health.derive_targets(pyproject)
    assert targets.packages == ["my_tool"]
