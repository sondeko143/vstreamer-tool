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


class _ScriptedRunner:
    # callable class (not a function with an attached attribute) so `ty`
    # resolves `.calls` — a function-attribute assignment fails `ty check`.
    def __init__(self, script):
        self._script = list(script)
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self._script.pop(0)


def _scripted_runner(script):
    return _ScriptedRunner(script)


def test_run_gate_pass():
    gate = health.Gate("ty", "static", ["ty", "check"], "report")
    run = _scripted_runner([(0, "ok", "")])
    result = health.run_gate(gate, run)
    assert result.status == "pass"


def test_run_gate_report_fail():
    gate = health.Gate("ty", "static", ["ty", "check"], "report")
    run = _scripted_runner([(1, "type error", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fail"
    assert "type error" in result.detail


def test_run_gate_fixable_autofixes():
    gate = health.Gate(
        "ruff-format", "static", ["fmt", "--check"], "fixable", fix=["fmt"]
    )
    # check fails, then fix runs, then re-check passes
    run = _scripted_runner([(1, "would reformat", ""), (0, "", ""), (0, "", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fixed"
    assert run.calls == [["fmt", "--check"], ["fmt"], ["fmt", "--check"]]


def test_run_gate_fixable_incomplete_stays_fail():
    gate = health.Gate(
        "ruff-lint", "static", ["lint"], "fixable", fix=["lint", "--fix"]
    )
    run = _scripted_runner([(1, "E501", ""), (0, "", ""), (1, "E501 remains", "")])
    result = health.run_gate(gate, run)
    assert result.status == "fail"
    assert "remains" in result.detail


def test_run_gate_skipped_when_tool_missing():
    gate = health.Gate("bandit", "extra", ["bandit"], "report")
    run = _scripted_runner([(127, "", "command not found: bandit")])
    result = health.run_gate(gate, run)
    assert result.status == "skipped"


def test_run_gate_prepare_failure_skips():
    gate = health.Gate(
        "pip-audit",
        "deps",
        ["pip-audit", "-r", "x"],
        "report",
        prepare=[["uv", "export"]],
    )
    run = _scripted_runner([(127, "", "command not found: uv")])
    result = health.run_gate(gate, run)
    assert result.status == "skipped"


def test_overall_exit_advisory_does_not_fail():
    results = [
        health.GateResult("ruff-format", "fixed", "auto-fixed"),
        health.GateResult("outdated", "fail", "stale", advisory=True),
    ]
    assert health.overall_exit(results) == 0


def test_overall_exit_hard_fail():
    results = [health.GateResult("ty", "fail", "needs attention")]
    assert health.overall_exit(results) == 1


def test_render_summary_lists_each_gate():
    results = [
        health.GateResult("ruff-format", "fixed", "auto-fixed"),
        health.GateResult("ty", "fail", "needs attention"),
    ]
    text = health.render_summary(results)
    assert "ruff-format" in text
    assert "FIXED" in text
    assert "FAIL" in text


def test_parse_coverage_reads_total():
    out = "TOTAL                      1234    200    84%\n"
    assert health.parse_coverage(out) == 84.0


def test_parse_coverage_none_when_absent():
    assert health.parse_coverage("no coverage here") is None


def test_run_all_collects_all_by_default():
    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    run = _scripted_runner([(1, "boom", ""), (0, "", "")])
    results = health.run_all(gates, run, fail_fast=False)
    assert [r.status for r in results] == ["fail", "pass"]


def test_run_all_fail_fast_stops_after_first_hard_fail():
    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    run = _scripted_runner([(1, "boom", "")])
    results = health.run_all(gates, run, fail_fast=True)
    assert [r.status for r in results] == ["fail"]


def test_run_all_isolates_gate_exception_as_error():
    def boom(cmd):
        raise OSError("kaboom")

    gates = [
        health.Gate("a", "static", ["a"], "report"),
        health.Gate("b", "static", ["b"], "report"),
    ]
    results = health.run_all(gates, boom, fail_fast=False)
    assert [r.status for r in results] == ["error", "error"]
    assert "kaboom" in results[0].detail


def test_overall_exit_error_status_hard_fails():
    results = [health.GateResult("x", "error", "gate raised")]
    assert health.overall_exit(results) == 1


def test_apply_no_fix_strips_only_fixable():
    fixable = health.Gate("f", "static", ["c"], "fixable", fix=["c", "--fix"])
    report = health.Gate("r", "deps", ["d"], "report")
    out = health.apply_no_fix([fixable, report])
    assert out[0].kind == "report"
    assert out[0].fix is None
    assert out[1].kind == "report"
    assert out[1].check == ["d"]
    assert out[1].fix is None
