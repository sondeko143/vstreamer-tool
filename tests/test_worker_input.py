import pytest
from pydantic import ValidationError
from vstreamer_protos.commander.commander_pb2 import PAUSE
from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import RELOAD
from vstreamer_protos.commander.commander_pb2 import RESUME
from vstreamer_protos.commander.commander_pb2 import SET_FILTERS
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TTS
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operand
from vstreamer_protos.commander.commander_pb2 import Operation
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2 import OperationRoute
from vstreamer_protos.commander.commander_pb2 import Sound

from vspeech.config import Config
from vspeech.config import EventType
from vspeech.shared_context import Params
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


@pytest.fixture(scope="module")
def app_context() -> SharedContext:
    return SharedContext(config=Config())


def create_command(
    operations: list[Operation],
    sound: Sound | None = None,
    text: str | None = None,
    file_path: str | None = None,
    filters: list[str] | None = None,
) -> Command:
    return Command(
        chains=[
            OperationChain(operations=[OperationRoute(operation=op)])
            for op in operations
        ],
        operand=Operand(
            sound=sound,
            text=text,
            file_path=file_path,
            filters=filters,
        ),
    )


def test_from_command_sound_data_invalid():
    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(create_command(operations=[TRANSCRIBE]))
    assert "sound input is invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(create_command(operations=[PLAYBACK], sound=Sound()))
    assert "sound input is invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            create_command(operations=[TRANSCRIBE], sound=Sound(data=b"dummy"))
        )
    assert "sound input is invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            create_command(operations=[PLAYBACK], sound=Sound(rate=44110))
        )
    assert "sound input is invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            create_command(operations=[TRANSCRIBE], sound=Sound(data=b"", rate=0))
        )
    assert "sound input is invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(create_command(operations=[PLAYBACK, TRANSCRIBE]))
    assert "sound input is invalid" in str(excinfo.value)


def test_from_command_sound_data_valid():
    worker_input = WorkerInput.from_command(
        create_command(
            operations=[TRANSCRIBE],
            sound=Sound(data=b"dummy", rate=44110, channels=1, format=8),
        )
    )
    assert (
        worker_input[0].current_event == EventType.transcription
        and worker_input[0].sound.data == b"dummy"
        and worker_input[0].sound.rate == 44110
        and worker_input[0].sound.channels == 1
        and worker_input[0].sound.format == 8
    )

    worker_input = WorkerInput.from_command(
        create_command(
            operations=[PLAYBACK, TRANSCRIBE],
            sound=Sound(data=b"dummy", rate=44110, channels=1, format=8),
        )
    )
    assert (
        worker_input[0].current_event == EventType.playback
        and worker_input[0].sound.data == b"dummy"
        and worker_input[0].sound.rate == 44110
        and worker_input[0].sound.channels == 1
        and worker_input[0].sound.format == 8
    )
    assert (
        worker_input[1].current_event == EventType.transcription
        and worker_input[1].sound.data == b"dummy"
        and worker_input[1].sound.rate == 44110
        and worker_input[1].sound.channels == 1
        and worker_input[1].sound.format == 8
    )


def test_from_command_text_data_valid():
    worker_input = WorkerInput.from_command(
        create_command(operations=[TTS], text="dummy")
    )
    assert (
        worker_input[0].current_event == EventType.tts
        and worker_input[0].text == "dummy"
    )

    worker_input = WorkerInput.from_command(
        create_command(operations=[TTS, SUBTITLE], text="dummy")
    )
    assert (
        worker_input[0].current_event == EventType.tts
        and worker_input[0].text == "dummy"
    )
    assert (
        worker_input[1].current_event == EventType.subtitle
        and worker_input[1].text == "dummy"
    )


def test_from_command_reload_invalid():
    with pytest.raises(ValidationError) as excinfo:
        WorkerInput.from_command(create_command(operations=[RELOAD]))
    assert "file_path is required" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        WorkerInput.from_command(create_command(operations=[RELOAD], file_path=None))
    assert "file_path is required" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        WorkerInput.from_command(create_command(operations=[RELOAD], file_path=""))
    assert "file_path is required" in str(excinfo.value)


def test_from_command_reload_valid():
    worker_input = WorkerInput.from_command(
        create_command(operations=[RELOAD], file_path="dummy")
    )
    assert (
        worker_input[0].current_event == EventType.reload
        and worker_input[0].file_path == "dummy"
    )


def test_from_command_set_filters_valid():
    worker_input = WorkerInput.from_command(create_command(operations=[SET_FILTERS]))
    assert (
        worker_input[0].current_event == EventType.set_filters
        and [f for f in worker_input[0].filters] == []
    )

    worker_input = WorkerInput.from_command(
        create_command(operations=[SET_FILTERS], filters=None)
    )
    assert (
        worker_input[0].current_event == EventType.set_filters
        and [f for f in worker_input[0].filters] == []
    )

    worker_input = WorkerInput.from_command(
        create_command(operations=[SET_FILTERS], filters=["f1", "f2"])
    )

    assert worker_input[0].current_event == EventType.set_filters and [
        f for f in worker_input[0].filters
    ] == [
        "f1",
        "f2",
    ]


def test_from_command_set_pause_resume_valid():
    worker_input = WorkerInput.from_command(create_command(operations=[PAUSE]))
    assert worker_input[0].current_event == EventType.pause

    worker_input = WorkerInput.from_command(
        create_command(operations=[RESUME], filters=None)
    )
    assert worker_input[0].current_event == EventType.resume


def test_parse_queries_from_pb():
    p = Params.from_pb(
        {
            "t": "ja",
            "s": "en",
            "p": "n",
            "i": "1",
            "v": "100",
            "spd": "1.1",
            "pit": "-0.05",
        }
    )
    assert p.position == "n"
    assert p.target_language_code == "ja"
    assert p.source_language_code == "en"
    assert p.speaker_id == 1
    assert p.volume == 100
    assert p.speed == 1.1
    assert p.pitch == -0.05


def test_parse_queries_to_pb():
    p = Params(
        t="ja",
        s="en",
        p="n",
        i=1,
        v=100,
        spd=1.1,
        pit=-0.05,
    )
    q = p.to_pb()
    o = OperationRoute(queries=q)
    assert o.queries["position"] == "n"
    assert o.queries["target_language_code"] == "ja"
    assert o.queries["source_language_code"] == "en"
    assert o.queries["speaker_id"] == "1"
    assert o.queries["volume"] == "100"
    assert o.queries["speed"] == "1.1"
    assert o.queries["speed"] == "1.1"
