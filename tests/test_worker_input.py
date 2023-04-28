import pytest
from vstreamer_protos.commander.commander_pb2 import PAUSE
from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import RELOAD
from vstreamer_protos.commander.commander_pb2 import RESUME
from vstreamer_protos.commander.commander_pb2 import SET_FILTERS
from vstreamer_protos.commander.commander_pb2 import SPEECH
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import SUBTITLE_TRANSLATED
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Sound

from vspeech.config import Config
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


@pytest.fixture(scope="module")
def app_context() -> SharedContext:
    return SharedContext(config=Config())


def test_from_command_sound_data_invalid():
    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[TRANSCRIBE]))
    assert "sound data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[PLAYBACK], sound=Sound()))
    assert "sound data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            Command(operations=[TRANSCRIBE], sound=Sound(data=b"dummy"))
        )
    assert "sound data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            Command(operations=[PLAYBACK], sound=Sound(rate=44110))
        )
    assert "sound data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            Command(operations=[TRANSCRIBE], sound=Sound(data=b"", rate=0))
        )
    assert "sound data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[PLAYBACK, TRANSCRIBE]))
    assert "sound data is required" == str(excinfo.value)


def test_from_command_sound_data_valid():
    worker_input = WorkerInput.from_command(
        Command(
            operations=[TRANSCRIBE],
            sound=Sound(data=b"dummy", rate=44110, channels=1, format=8),
        )
    )
    assert (
        worker_input.operations == [TRANSCRIBE]
        and worker_input.sound.data == b"dummy"
        and worker_input.sound.rate == 44110
        and worker_input.sound.channels == 1
        and worker_input.sound.format == 8
    )

    worker_input = WorkerInput.from_command(
        Command(
            operations=[PLAYBACK],
            sound=Sound(data=b"dummy", rate=44110, channels=1, format=8),
        )
    )
    assert (
        worker_input.operations == [PLAYBACK]
        and worker_input.sound.data == b"dummy"
        and worker_input.sound.rate == 44110
        and worker_input.sound.channels == 1
        and worker_input.sound.format == 8
    )

    worker_input = WorkerInput.from_command(
        Command(
            operations=[PLAYBACK, TRANSCRIBE],
            sound=Sound(data=b"dummy", rate=44110, channels=1, format=8),
        )
    )
    assert (
        worker_input.operations == [PLAYBACK, TRANSCRIBE]
        and worker_input.sound.data == b"dummy"
        and worker_input.sound.rate == 44110
        and worker_input.sound.channels == 1
        and worker_input.sound.format == 8
    )


def test_from_command_text_data_invalid():
    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[SPEECH]))
    assert "text data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[SUBTITLE], text=None))
    assert "text data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[SUBTITLE_TRANSLATED], text=""))
    assert "text data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[TRANSLATE]))
    assert "text data is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(
            Command(operations=[SPEECH, SUBTITLE, SUBTITLE_TRANSLATED, TRANSLATE])
        )
    assert "text data is required" == str(excinfo.value)


def test_from_command_text_data_valid():
    worker_input = WorkerInput.from_command(Command(operations=[SPEECH], text="dummy"))
    assert worker_input.operations == [SPEECH] and worker_input.text == "dummy"

    worker_input = WorkerInput.from_command(
        Command(
            operations=[SPEECH, SUBTITLE, SUBTITLE_TRANSLATED, TRANSLATE], text="dummy"
        )
    )
    assert (
        worker_input.operations == [SPEECH, SUBTITLE, SUBTITLE_TRANSLATED, TRANSLATE]
        and worker_input.text == "dummy"
    )


def test_from_command_reload_invalid():
    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[RELOAD]))
    assert "file path is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[RELOAD], file_path=None))
    assert "file path is required" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        WorkerInput.from_command(Command(operations=[RELOAD], file_path=""))
    assert "file path is required" == str(excinfo.value)


def test_from_command_reload_valid():
    worker_input = WorkerInput.from_command(
        Command(operations=[RELOAD], file_path="dummy")
    )
    assert worker_input.operations == [RELOAD] and worker_input.file_path == "dummy"


def test_from_command_set_filters_valid():
    worker_input = WorkerInput.from_command(Command(operations=[SET_FILTERS]))
    assert worker_input.operations == [SET_FILTERS] and worker_input.filters == []

    worker_input = WorkerInput.from_command(
        Command(operations=[SET_FILTERS], filters=None)
    )
    assert worker_input.operations == [SET_FILTERS] and worker_input.filters == []

    worker_input = WorkerInput.from_command(
        Command(operations=[SET_FILTERS], filters=["f1", "f2"])
    )
    assert worker_input.operations == [SET_FILTERS] and worker_input.filters == [
        "f1",
        "f2",
    ]


def test_from_command_set_pause_resume_valid():
    worker_input = WorkerInput.from_command(Command(operations=[PAUSE]))
    assert worker_input.operations == [PAUSE]

    worker_input = WorkerInput.from_command(Command(operations=[RESUME], filters=None))
    assert worker_input.operations == [RESUME]
