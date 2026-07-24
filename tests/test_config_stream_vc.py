import pytest
from pydantic import ValidationError

from vspeech.config import Config
from vspeech.config import StreamVcRole
from vspeech.config import TransportType


def test_stream_vc_defaults_are_local_in_process():
    sv = Config().stream_vc
    assert sv.role is StreamVcRole.local
    assert sv.transport_type is TransportType.in_process
    assert sv.jitter_buffer_ms == 0.0


def test_stream_vc_producer_parses_udp_and_peer():
    sv = Config.model_validate(
        {
            "stream_vc": {
                "role": "producer",
                "transport_type": "udp",
                "peer_host": "playback-host",
                "peer_port": 9999,
            }
        }
    ).stream_vc
    assert sv.role is StreamVcRole.producer
    assert sv.transport_type is TransportType.udp
    assert (sv.peer_host, sv.peer_port) == ("playback-host", 9999)


def test_stream_vc_consumer_parses_bind_and_jitter():
    sv = Config.model_validate(
        {
            "stream_vc": {
                "role": "consumer",
                "transport_type": "udp",
                "bind_host": "0.0.0.0",
                "bind_port": 9999,
                "jitter_buffer_ms": 60.0,
            }
        }
    ).stream_vc
    assert sv.role is StreamVcRole.consumer
    assert (sv.bind_host, sv.bind_port) == ("0.0.0.0", 9999)
    assert sv.jitter_buffer_ms == 60.0


def test_stream_vc_envelope_defaults_off():
    sv = Config().stream_vc
    assert sv.envelope_follow is False
    assert sv.envelope_strength == 1.0
    assert sv.envelope_min_gain == 0.1
    assert sv.envelope_max_gain == 1.0
    assert sv.envelope_window_ms == 25.0
    assert sv.envelope_ema_ms == 2000.0


def test_stream_vc_envelope_parses():
    sv = Config.model_validate(
        {
            "stream_vc": {
                "envelope_follow": True,
                "envelope_strength": 1.5,
                "envelope_min_gain": 0.2,
                "envelope_ema_ms": 1500.0,
            }
        }
    ).stream_vc
    assert sv.envelope_follow is True
    assert (sv.envelope_strength, sv.envelope_min_gain, sv.envelope_ema_ms) == (
        1.5,
        0.2,
        1500.0,
    )


def test_stream_vc_envelope_min_gt_max_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"stream_vc": {"envelope_min_gain": 0.5, "envelope_max_gain": 0.2}}
        )
