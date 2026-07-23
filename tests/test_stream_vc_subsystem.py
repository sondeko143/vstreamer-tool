from vspeech.config import StreamVcRole
from vspeech.stream_vc.subsystem import loops_for_role


def test_loops_for_role():
    assert loops_for_role(StreamVcRole.local) == frozenset(
        {"capture", "vc", "playback"}
    )
    assert loops_for_role(StreamVcRole.producer) == frozenset({"capture", "vc"})
    assert loops_for_role(StreamVcRole.consumer) == frozenset({"playback"})
