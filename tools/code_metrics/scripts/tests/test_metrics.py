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
