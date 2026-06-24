import json

from tools.code_metrics.scripts import metrics

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
