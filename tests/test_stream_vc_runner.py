from vspeech.stream_vc.runner import make_stream_packet


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
