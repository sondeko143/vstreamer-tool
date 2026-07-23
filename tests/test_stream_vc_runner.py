from vspeech.stream_vc.runner import make_stream_packet


def test_make_stream_packet_pts_is_seq_times_hop():
    p = make_stream_packet("sess", 5, 0.08, b"\x01\x02", 40000)
    assert p.session_id == "sess"
    assert p.seq == 5
    assert abs(p.pts - 0.4) < 1e-9  # 5 * 0.08
    assert p.pcm == b"\x01\x02"
    assert p.sample_rate == 40000
