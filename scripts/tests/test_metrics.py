import json
from pathlib import Path

from scripts import metrics

LIZARD_CSV = (
    '66,21,474,2,66,"process_command@30-95@vspeech\\lib\\command.py",'
    '"vspeech\\lib\\command.py","process_command","process_command( c )",30,95\n'
    '13,13,120,1,13,"operation_to_event@97-130@vspeech\\shared_context.py",'
    '"vspeech\\shared_context.py","operation_to_event","operation_to_event( op )",97,130\n'
    '8,3,60,1,8,"draw_text@41-70@vspeech\\worker\\subtitle.py",'
    '"vspeech\\worker\\subtitle.py","draw_text","draw_text( s )",41,70\n'
    '5,2,30,1,5,"helper@10-15@vspeech\\lib\\ami.py",'
    '"vspeech\\lib\\ami.py","helper","helper( x )",10,15\n'
)

COMPLEXIPY_JSON = json.dumps(
    [
        {
            "complexity": 28,
            "file_name": "command.py",
            "function_name": "process_command",
            "path": "vspeech/lib/command.py",
            "refactor_plans": [],
        },
        {
            "complexity": 3,
            "file_name": "shared_context.py",
            "function_name": "operation_to_event",
            "path": "vspeech/shared_context.py",
            "refactor_plans": [],
        },
        {
            "complexity": 18,
            "file_name": "subtitle.py",
            "function_name": "draw_text",
            "path": "vspeech/worker/subtitle.py",
            "refactor_plans": [],
        },
        {
            "complexity": 1,
            "file_name": "ami.py",
            "function_name": "helper",
            "path": "vspeech/lib/ami.py",
            "refactor_plans": [],
        },
        {
            "complexity": 12,
            "file_name": "vc.py",
            "function_name": "RVCModel::orphan",
            "path": "vspeech/worker/vc.py",
            "refactor_plans": [],
        },
    ]
)


def test_parse_lizard_csv_extracts_fields():
    rows = metrics.parse_lizard_csv(LIZARD_CSV)
    assert len(rows) == 4
    pc = next(r for r in rows if r.function == "process_command")
    assert pc.file == "vspeech/lib/command.py"
    assert pc.ccn == 21
    assert pc.nloc == 66
    assert pc.params == 2
    assert pc.line == 30
    assert pc.cognitive is None


def test_parse_lizard_csv_skips_non_numeric_rows():
    rows = metrics.parse_lizard_csv("not,a,real,row\n" + LIZARD_CSV)
    assert len(rows) == 4


def test_normalize_path_converts_backslashes():
    assert (
        metrics.normalize_path("vspeech\\lib\\command.py") == "vspeech/lib/command.py"
    )


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


def test_parse_complexipy_json_strips_class_and_normalizes():
    rows = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    assert ("vspeech/worker/vc.py", "orphan", 12) in rows
    assert ("vspeech/lib/command.py", "process_command", 28) in rows


def test_build_cognitive_index_marks_conflicts_ambiguous():
    rows = [
        ("a.py", "run", 5),
        ("a.py", "run", 9),
        ("b.py", "go", 7),
    ]
    index = metrics.build_cognitive_index(rows)
    assert index[("a.py", "run")] is None  # conflicting -> ambiguous
    assert index[("b.py", "go")] == 7


def test_join_fills_cognitive_and_appends_orphans():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)

    pc = next(m for m in joined if m.function == "process_command")
    assert pc.ccn == 21 and pc.cognitive == 28

    orphan = next(m for m in joined if m.function == "orphan")
    assert orphan.ccn is None
    assert orphan.line is None
    assert orphan.cognitive == 12
    assert orphan.file == "vspeech/worker/vc.py"

    # exactly one orphan appended (the 4 lizard rows + 1 complexipy-only)
    assert len(joined) == 5


def test_bucket_classifies_each_case():
    t = metrics.Thresholds()
    both = metrics.FunctionMetric("f.py", "a", 1, 21, 60, 2, 28)
    ccn_only = metrics.FunctionMetric("f.py", "b", 1, 13, 13, 1, 3)
    cog_only = metrics.FunctionMetric("f.py", "c", 1, 3, 8, 1, 18)
    ok = metrics.FunctionMetric("f.py", "d", 1, 2, 5, 1, 1)
    assert metrics.bucket(both, t) == "both-high"
    assert metrics.bucket(ccn_only, t) == "high-ccn"
    assert metrics.bucket(cog_only, t) == "high-cognitive"
    assert metrics.bucket(ok, t) == "ok"


def test_rank_orders_by_cognitive_then_ccn_none_last():
    a = metrics.FunctionMetric("f.py", "a", 1, 5, 5, 1, 28)
    b = metrics.FunctionMetric("f.py", "b", 1, 9, 9, 1, 18)
    c = metrics.FunctionMetric("f.py", "c", 1, 30, 9, 1, None)
    ranked = metrics.rank_metrics([c, b, a])
    assert [m.function for m in ranked] == ["a", "b", "c"]


def test_render_summary_flags_and_explains():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    out = metrics.render_summary(joined, metrics.Thresholds(), top=15)
    assert "process_command" in out
    assert "both-high" in out
    assert "draw_text" in out  # high-cognitive
    # de-prioritized flat dispatcher is named in the "likely fine" prose
    assert "operation_to_event" in out


def test_metrics_to_json_is_valid_and_has_bucket():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    payload = json.loads(metrics.metrics_to_json(joined, metrics.Thresholds()))
    assert payload[0]["function"] == "process_command"
    assert payload[0]["bucket"] == "both-high"


def test_collect_complexipy_reads_file_despite_nonzero_exit(tmp_path):
    out_file = tmp_path / "cx.json"

    def fake_run(cmd, env_extra=None):
        # complexipy exits 1 when it finds high-complexity functions
        out_file.write_text(COMPLEXIPY_JSON, encoding="utf-8")
        assert env_extra == {"PYTHONIOENCODING": "utf-8"}
        return 1, "", ""

    text = metrics.collect_complexipy(fake_run, ["vspeech"], str(out_file))
    assert text is not None
    assert "process_command" in text


def test_collect_lizard_returns_none_when_missing():
    def fake_run(cmd, env_extra=None):
        return 127, "", "command not found: uvx"

    assert metrics.collect_lizard(fake_run, ["vspeech"]) is None


def test_main_runs_advisory_and_returns_zero(tmp_path, capsys, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n', encoding="utf-8"
    )

    def fake_runner(cmd, env_extra=None):
        if "lizard" in cmd:
            return 0, LIZARD_CSV, ""
        if "complexipy" in cmd:
            idx = cmd.index("--output") + 1
            Path(cmd[idx]).write_text(COMPLEXIPY_JSON, encoding="utf-8")
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(metrics, "subprocess_runner", fake_runner)
    rc = metrics.main(["--root", str(tmp_path)])
    assert rc == 0
    assert "process_command" in capsys.readouterr().out


def test_ccn_band_classifies():
    t = metrics.Thresholds()
    high = metrics.FunctionMetric("f.py", "a", 1, 21, 1, 1, 5)
    watch = metrics.FunctionMetric("f.py", "b", 1, 13, 1, 1, 5)
    ok = metrics.FunctionMetric("f.py", "c", 1, 3, 1, 1, 5)
    na = metrics.FunctionMetric("f.py", "d", 1, None, None, None, 5)
    assert metrics.ccn_band(high, t) == "high"
    assert metrics.ccn_band(watch, t) == "watch"
    assert metrics.ccn_band(ok, t) == "ok"
    assert metrics.ccn_band(na, t) == "n/a"


def test_render_summary_surfaces_high_ccn_band():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    out = metrics.render_summary(joined, metrics.Thresholds(), top=15)
    assert "Highest cyclomatic (ccn > 20):" in out
    assert "process_command" in out


# --- DepDegree (def-use coupling) lens ---------------------------------------

DEP_SRC = """
def simple(a):
    return a


def entangled(x):
    y = x
    z = y
    y = z + x
    return y
"""


def test_compute_depdegree_counts_def_use_edges():
    rows = {name: dep for name, _line, dep in metrics.compute_depdegree(DEP_SRC)}
    # simple: param `a` has a single use -> 1 def-use edge
    assert rows["simple"] == 1
    # entangled by hand: x->{y=x, y=z+x}=2, y(=x)->{z=y}=1, z->{y=z+x}=1,
    #                    y(=z+x)->{return y}=1  => 5 edges
    assert rows["entangled"] == 5
    assert rows["entangled"] > rows["simple"]


def test_compute_depdegree_reports_lineno():
    rows = metrics.compute_depdegree(DEP_SRC)
    by_name = {name: line for name, line, _dep in rows}
    assert by_name["simple"] == 2
    assert by_name["entangled"] == 6


ASYNC_EXCEPT_SRC = """
async def worker(ctx, q):
    while True:
        item = ctx.read()
        try:
            out = transform(item)
            q.put(out)
        except* ValueError:
            q.put(item)
"""


def test_compute_depdegree_handles_async_and_except_star():
    # 3.11 `except*` must parse and the async function must be measured
    rows = {
        name: dep for name, _line, dep in metrics.compute_depdegree(ASYNC_EXCEPT_SRC)
    }
    assert "worker" in rows
    assert rows["worker"] > 0


def test_collect_depdegree_walks_package_files(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__pycache__").mkdir()
    (pkg / "mod.py").write_text(DEP_SRC, encoding="utf-8")
    (pkg / "__pycache__" / "junk.py").write_text("def x(): pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rows = metrics.collect_depdegree(["pkg"])
    assert rows is not None
    by_key = {(f, fn): dep for f, fn, dep in rows}
    assert by_key[("pkg/mod.py", "entangled")] == 5
    # __pycache__ is skipped
    assert not any(fn == "x" for _f, fn, _dep in rows)


def test_join_fills_dep_from_dep_rows():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    dep_rows = [("vspeech/lib/command.py", "process_command", 70)]
    joined = metrics.join_metrics(lizard, cog, dep_rows)
    pc = next(m for m in joined if m.function == "process_command")
    assert pc.dep == 70
    # functions with no dep row keep dep=None
    helper = next(m for m in joined if m.function == "helper")
    assert helper.dep is None


def test_join_without_dep_rows_keeps_dep_none():
    lizard = metrics.parse_lizard_csv(LIZARD_CSV)
    cog = metrics.parse_complexipy_json(COMPLEXIPY_JSON)
    joined = metrics.join_metrics(lizard, cog)
    assert all(m.dep is None for m in joined)


def test_entangled_predicate_uses_dep_threshold():
    t = metrics.Thresholds(dep_warn=40)
    hot = metrics.FunctionMetric("f.py", "a", 1, 3, 80, 1, 2, dep=60)
    cold = metrics.FunctionMetric("f.py", "b", 1, 3, 80, 1, 2, dep=10)
    none = metrics.FunctionMetric("f.py", "c", 1, 3, 80, 1, 2, dep=None)
    assert metrics.entangled(hot, t) is True
    assert metrics.entangled(cold, t) is False
    assert metrics.entangled(none, t) is False


def test_bucket_labels_pure_entangled():
    t = metrics.Thresholds(dep_warn=40)
    # low ccn + low cognitive but high def-use coupling -> "entangled"
    m = metrics.FunctionMetric("f.py", "e", 1, 3, 80, 1, 2, dep=60)
    assert metrics.bucket(m, t) == "entangled"
    # ccn/cog flags still win over entangled
    both = metrics.FunctionMetric("f.py", "f", 1, 21, 80, 1, 28, dep=60)
    assert metrics.bucket(both, t) == "both-high"


def test_render_summary_surfaces_entangled_prose():
    t = metrics.Thresholds(dep_warn=40)
    metrics_list = [
        metrics.FunctionMetric("a.py", "tangled", 10, 5, 80, 1, 6, dep=90),
        metrics.FunctionMetric("a.py", "calm", 20, 3, 10, 1, 2, dep=4),
    ]
    out = metrics.render_summary(metrics_list, t, top=15)
    assert "Entangled state" in out
    assert "tangled" in out
    # the entangled function is shown even though ccn/cognitive are within bands
    assert "a.py:10" in out


def test_metrics_to_json_includes_dep():
    m = [metrics.FunctionMetric("a.py", "f", 1, 5, 10, 1, 6, dep=42)]
    payload = json.loads(metrics.metrics_to_json(m, metrics.Thresholds()))
    assert payload[0]["dep"] == 42
