import pytest

from vspeech.config import EventType
from vspeech.shared_context import EventAddress
from vspeech.shared_context import FollowingEvents
from vspeech.shared_context import WorkerOutput


def test_event_address_from_string():
    for v in ["transcription", "/transcription"]:
        ea = EventAddress.from_string(v)
        assert ea.event == EventType.transcription
        assert ea.remote == ""
    for v in ["//localhost/transcription", "//localhost/transcription/"]:
        ea = EventAddress.from_string(v)
        assert ea.event == EventType.transcription
        assert ea.remote == "//localhost"
    for v in ["//localhost:443/transcription", "//localhost:443/transcription/"]:
        ea = EventAddress.from_string(v)
        assert ea.event == EventType.transcription
        assert ea.remote == "//localhost:443"


@pytest.fixture(scope="module")
def followings() -> FollowingEvents:
    return [
        [
            EventAddress(event=EventType.transcription),
            EventAddress(event=EventType.tts),
        ],
        [
            EventAddress(event=EventType.transcription),
            EventAddress(event=EventType.subtitle),
        ],
        [
            EventAddress(event=EventType.transcription),
            EventAddress(event=EventType.translation),
            EventAddress(event=EventType.subtitle_translated),
        ],
        [
            EventAddress(event=EventType.transcription, remote="remote"),
            EventAddress(event=EventType.translation),
            EventAddress(event=EventType.tts),
        ],
        [
            EventAddress(event=EventType.transcription),
            EventAddress(event=EventType.translation),
            EventAddress(event=EventType.subtitle_translated),
        ],
        [
            EventAddress(event=EventType.playback),
        ],
        [
            EventAddress(event=EventType.playback, remote="remote"),
        ],
        [],
    ]


def test_worker_output_remotes(followings: FollowingEvents):
    output = WorkerOutput(followings=followings)
    remotes = output.remotes
    assert remotes == set(["", "remote"])
    remotes = output.events("")
    assert remotes == [
        [EventType.transcription, EventType.tts],
        [EventType.transcription, EventType.subtitle],
        [
            EventType.transcription,
            EventType.translation,
            EventType.subtitle_translated,
        ],
        [
            EventType.transcription,
            EventType.translation,
            EventType.subtitle_translated,
        ],
        [EventType.playback],
    ]
    remotes = output.events("remote")
    assert remotes == [
        [
            EventAddress(event=EventType.transcription, remote="remote"),
            EventType.translation,
            EventType.tts,
        ],
        [EventAddress(event=EventType.playback, remote="remote")],
    ]
