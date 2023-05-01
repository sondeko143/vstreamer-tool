from asyncio import Event
from asyncio import Queue
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Iterable
from typing import TypeVar
from typing import Union

from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import RELOAD
from vstreamer_protos.commander.commander_pb2 import SPEECH
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import SUBTITLE_TRANSLATED
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operation

from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import SampleFormat

T = TypeVar("T")


@dataclass
class SoundOutput:
    data: bytes
    rate: int
    format: SampleFormat
    channels: int


@dataclass
class WorkerOutput:
    source: EventType
    sound: SoundOutput | None
    text: str | None


@dataclass
class SoundInput:
    data: bytes
    rate: int
    format: SampleFormat
    channels: int


@dataclass
class WorkerInput:
    operations: Iterable["Operation"]
    text: str
    sound: SoundInput
    file_path: str
    filters: Iterable[str]

    @classmethod
    def validate(
        cls, operand: Union[WorkerOutput, Command], operations: Iterable["Operation"]
    ):
        sound = operand.sound or SoundInput(
            data=b"", rate=0, channels=0, format=SampleFormat.INVALID
        )
        if any(o in [TRANSCRIBE, PLAYBACK] for o in operations):
            if (
                not operand.sound
                or not operand.sound.rate
                or not operand.sound.data
                or not operand.sound.format
                or not operand.sound.channels
            ):
                raise ValueError("sound data is required")
        text = operand.text or ""
        if any(
            o in [SPEECH, SUBTITLE, SUBTITLE_TRANSLATED, TRANSLATE] for o in operations
        ):
            if not operand.text:
                raise ValueError("text data is required")
        if not isinstance(operand, Command):
            return cls(
                operations=operations,
                sound=SoundInput(
                    data=sound.data,
                    rate=sound.rate,
                    channels=sound.channels,
                    format=SampleFormat(sound.format),
                ),
                text=text,
                file_path="",
                filters=[],
            )
        if any(o in [RELOAD] for o in operations):
            if not operand.file_path:
                raise ValueError("file path is required")
        return cls(
            operations=operations,
            sound=SoundInput(
                data=sound.data,
                rate=sound.rate,
                channels=sound.channels,
                format=SampleFormat(sound.format),
            ),
            text=text,
            file_path=operand.file_path,
            filters=operand.filters,
        )

    @classmethod
    def from_output(cls, output: WorkerOutput, operations: Iterable["Operation"]):
        return WorkerInput.validate(operand=output, operations=operations)

    @classmethod
    def from_command(cls, command: Command):
        return WorkerInput.validate(operand=command, operations=command.operations)


InputQueues = Dict[EventType, Queue[WorkerInput]]


@dataclass
class SharedContext:
    config: Config
    input_queues: InputQueues = field(default_factory=dict)
    resume: Event = field(default_factory=Event)
    sender_queue: Queue[WorkerOutput] = field(default_factory=Queue)
    reload: Dict[str, bool] = field(
        default_factory=lambda: {worker_name.name: False for worker_name in EventType}
    )

    def __post_init__(self):
        self.resume.set()
