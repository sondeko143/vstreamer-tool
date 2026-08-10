from vspeech.config import StreamVcConfig
from vspeech.stream_vc.runner import make_stream_envelope
from vspeech.stream_vc.runner import make_stream_packet


def test_make_stream_envelope_gated_by_flag():
    assert make_stream_envelope(StreamVcConfig()) is None  # default off
    env = make_stream_envelope(StreamVcConfig(envelope_follow=True))
    assert env is not None
    assert env.strength == 1.0


def test_make_stream_packet_pts_is_seq_times_hop():
    p = make_stream_packet("sess", 5, 0.08, b"\x01\x02", 40000)
    assert p.session_id == "sess"
    assert p.seq == 5
    assert abs(p.pts - 0.4) < 1e-9  # 5 * 0.08
    assert p.pcm == b"\x01\x02"
    assert p.sample_rate == 40000


def test_apply_input_boost_scales_and_clips():
    import numpy as np

    from vspeech.stream_vc.runner import apply_input_boost

    block = np.array([0.1, 0.5, -0.5, 0.9], dtype=np.float32)
    out = apply_input_boost(block, 2.0)
    assert out.dtype == np.float32
    # 0.1*2=0.2, 0.5*2=1.0, -0.5*2=-1.0, 0.9*2=1.8 -> clip to 1.0
    assert np.allclose(out, np.array([0.2, 1.0, -1.0, 1.0], dtype=np.float32))


def test_apply_input_boost_identity_at_one():
    import numpy as np

    from vspeech.stream_vc.runner import apply_input_boost

    block = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert apply_input_boost(block, 1.0) is block  # identity fast-path


def test_make_streaming_vc_extends_the_context_by_the_lookahead(monkeypatch):
    """The analysis window is passed extended by the lookahead (so the left context
    does not shrink)."""
    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc import runner as runner_mod

    captured: dict[str, object] = {}

    class _Spy:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("vspeech.lib.stream_vc.StreamingVc", _Spy)
    sv = StreamVcConfig(context_ms=500.0, lookahead_ms=160.0, block_ms=160.0)
    rt = {
        "rvc_config": sv.rvc,
        "device": None,
        "hubert_model": None,
        "session": None,
        "f0_session": None,
        "target_sample_rate": 40000,
        "f0_enabled": True,
        "emb_output_layer": 9,
        "use_final_proj": True,
    }
    runner_mod.make_streaming_vc(rt, sv)
    assert captured["context_len"] == round((500.0 + 160.0) * 16)
    assert captured["lookahead_len"] == round(160.0 * 16)
    # at the default (0) the window length is unchanged
    captured.clear()
    runner_mod.make_streaming_vc(rt, StreamVcConfig(context_ms=500.0))
    assert captured["context_len"] == round(500.0 * 16)
    assert captured["lookahead_len"] == 0


def test_geometry_summary_reports_the_window_and_both_delays():
    """The startup log carries the window, the emit delay, and the added latency."""
    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc.runner import geometry_summary

    sv = StreamVcConfig(context_ms=500.0, block_ms=160.0, lookahead_ms=160.0)
    line = geometry_summary(sv, emit_delay_samples=8400, target_sample_rate=40000)
    assert "解析窓 820ms" in line  # 500 + 160 + 160
    assert "emit 遅延 210.0ms" in line  # 8400 / 40000
    assert "付加遅延 160ms" in line
