import pytest
from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import TTS
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2 import OperationRoute

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.shared_context import EventAddress
from vspeech.shared_context import FollowingEvents
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
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
    output = WorkerOutput(
        followings=followings,
        sound=SoundOutput(
            data=b"aaa", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        text="aaa",
    )
    remotes = output.remotes
    assert remotes == set(["", "remote"])


def test_worker_output_to_inputs(followings: FollowingEvents):
    output = WorkerOutput(
        followings=followings,
        sound=SoundOutput(
            data=b"aaa", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        text="aaa",
    )
    inputs = WorkerInput.from_output(output, remote="")
    assert {i.current_event: i.following_events for i in inputs} == {
        EventType.transcription: [
            [EventType.tts],
            [EventType.subtitle],
            [
                EventType.translation,
                EventType.subtitle_translated,
            ],
            [
                EventType.translation,
                EventType.subtitle_translated,
            ],
        ],
        EventType.playback: [[]],
    }


def test_worker_output_to_command(followings: FollowingEvents):
    output = WorkerOutput(
        followings=followings,
        sound=SoundOutput(
            data=b"aaa", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        text="aaa",
    )
    command = output.to_pb("remote")
    assert command.chains == [
        OperationChain(
            operations=[
                OperationRoute(operation=TRANSCRIBE, remote="remote"),
                OperationRoute(operation=TRANSLATE, remote=""),
                OperationRoute(operation=TTS, remote=""),
            ],
        ),
        OperationChain(
            operations=[OperationRoute(operation=PLAYBACK, remote="remote")]
        ),
    ]
