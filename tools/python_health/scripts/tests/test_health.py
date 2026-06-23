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


def test_classify_pass_fail_skip():
    assert health.classify(0, "", "") == "pass"
    assert health.classify(1, "errors", "") == "fail"
    assert health.classify(127, "", "command not found: ruff") == "skipped"


def test_build_gates_uses_targets_for_cov_and_security():
    targets = health.Targets(packages=["vspeech"], project_name="voicerecog")
    gates = health.build_gates(targets)
    names = [g.name for g in gates]
    assert names == [
        "ruff-format",
        "ruff-lint",
        "ty",
        "pytest-cov",
        "uv-lock-check",
        "pip-audit",
        "outdated",
        "bandit",
        "vulture",
    ]
    pytest_gate = next(g for g in gates if g.name == "pytest-cov")
    assert "--cov=vspeech" in pytest_gate.check
    fmt = next(g for g in gates if g.name == "ruff-format")
    assert fmt.kind == "fixable" and fmt.fix is not None
    outdated = next(g for g in gates if g.name == "outdated")
    assert outdated.advisory is True
