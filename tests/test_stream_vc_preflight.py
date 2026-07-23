from pathlib import Path

from vspeech.config import Config
from vspeech.preflight import collect_problems


def _fields(problems) -> set[str]:
    return {p.field for p in problems}


def test_stream_vc_disabled_no_problems():
    c = Config()  # stream_vc disabled by default
    fields = _fields(collect_problems(c))
    assert not any(f and f.startswith("stream_vc") for f in fields)


def test_stream_vc_enabled_missing_model_reported():
    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.rvc.model_file = Path("/nonexistent/voice.onnx")
    c.stream_vc.rvc.hubert_model_file = Path("/nonexistent/hubert")
    fields = _fields(collect_problems(c))
    assert "stream_vc.rvc.model_file" in fields
    assert "stream_vc.rvc.hubert_model_file" in fields


def test_stream_vc_enabled_crossfade_gt_block_reported():
    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.block_ms = 10.0
    c.stream_vc.crossfade_ms = 20.0  # crossfade must be < block
    problems = collect_problems(c)
    assert any(p.field == "stream_vc.crossfade_ms" for p in problems)
