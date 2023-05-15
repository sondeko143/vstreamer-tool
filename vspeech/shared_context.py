from asyncio import Event
from asyncio import Queue
from asyncio import current_task
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Iterable
from typing import TypeAlias
from typing import cast
from urllib.parse import urlparse
from urllib.parse import urlunparse
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import root_validator
from vstreamer_protos.commander.commander_pb2 import PAUSE
from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import RELOAD
from vstreamer_protos.commander.commander_pb2 import RESUME
from vstreamer_protos.commander.commander_pb2 import SET_FILTERS
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import SUBTITLE_TRANSLATED
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import TTS
from vstreamer_protos.commander.commander_pb2 import VC
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operation
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2 import OperationRoute
from vstreamer_protos.commander.commander_pb2 import Sound

from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import RoutesList
from vspeech.config import SampleFormat
from vspeech.exceptions import EventToOperationConvertError


def event_to_operation(event: EventType) -> Operation:
    if event == EventType.transcription:
        return TRANSCRIBE
    if event == EventType.translation:
        return TRANSLATE
    if event == EventType.subtitle:
        return SUBTITLE
    if event == EventType.subtitle_translated:
        return SUBTITLE_TRANSLATED
    if event == EventType.tts:
        return TTS
    if event == EventType.vc:
        return VC
    if event == EventType.playback:
        return PLAYBACK
    if event == EventType.pause:
        return PAUSE
    if event == EventType.resume:
        return RESUME
    if event == EventType.reload:
        return RELOAD
    if event == EventType.set_filters:
        return SET_FILTERS
    raise EventToOperationConvertError(f"Unsupported event type {event}")


def is_sound_event(event: EventType) -> bool:
    if event in (
        EventType.transcription,
        EventType.vc,
        EventType.playback,
    ):
        return True
    return False


def is_text_event(event: EventType) -> bool:
    if event in (
        EventType.translation,
        EventType.subtitle,
        EventType.subtitle_translated,
        EventType.tts,
    ):
        return True
    return False


def operation_to_event(operation: Operation) -> EventType:
    if operation == TRANSCRIBE:
        return EventType.transcription
    if operation == TRANSLATE:
        return EventType.translation
    if operation == SUBTITLE:
        return EventType.subtitle
    if operation == SUBTITLE_TRANSLATED:
        return EventType.subtitle_translated
    if operation == TTS:
        return EventType.tts
    if operation == VC:
        return EventType.vc
    if operation == PLAYBACK:
        return EventType.playback
    if operation == PAUSE:
        return EventType.pause
    if operation == RESUME:
        return EventType.resume
    if operation == RELOAD:
        return EventType.reload
    if operation == SET_FILTERS:
        return EventType.set_filters
    raise EventToOperationConvertError(f"Unsupported operation {operation}")


@dataclass
class EventAddress:
    event: EventType
    remote: str = ""

    @classmethod
    def from_string(cls, destination: str):
        """<endpoint_uri>/<event_name>"""
        url = urlparse(destination)
        target_name = url.path.strip("/")
        address = (
            urlunparse((url.scheme, url.netloc, "", "", "", "")) if url.netloc else ""
        )
        return cls(event=EventType.from_string(target_name), remote=address)

    def to_pb(self) -> OperationRoute:
        return OperationRoute(
            operation=event_to_operation(self.event), remote=self.remote
        )

    def __hash__(self) -> int:
        return hash(self.event) + hash(self.remote)

    def __eq__(self, __value: object) -> bool:
        if not self.remote and isinstance(__value, EventType):
            return self.event == __value
        if not isinstance(__value, EventAddress):
            return False
        return self.event == __value.event and self.remote == __value.remote

    @classmethod
    def from_pb(cls, op_route: OperationRoute) -> "EventAddress":
        return cls(event=operation_to_event(op_route.operation), remote=op_route.remote)


FollowingEvents: TypeAlias = list[list[EventAddress]]


def get_remotes_of_events(followings: FollowingEvents):
    next_events = [events[0] for events in followings if events]
    return set(n.remote for n in next_events)


def get_events_of_remote(followings: FollowingEvents, remote: str) -> FollowingEvents:
    return [events for events in followings if events and events[0].remote == remote]


def get_first_event_map(
    followings: FollowingEvents,
) -> dict[EventAddress, FollowingEvents]:
    maps: dict[EventAddress, FollowingEvents] = defaultdict(list)
    [maps[events[0]].append(events[1:]) for events in followings if events]
    return maps


def command_to_events(command: Command) -> FollowingEvents:
    return [[EventAddress.from_pb(o) for o in cs.operations] for cs in command.chains]


@dataclass
class SoundOutput:
    data: bytes
    rate: int
    format: SampleFormat
    channels: int

    def to_pb(self) -> Sound:
        return Sound(
            data=self.data,
            rate=self.rate,
            format=self.format,
            channels=self.channels,
        )


@dataclass
class WorkerOutput:
    input_id: UUID
    followings: FollowingEvents
    sound: SoundOutput | None = None
    text: str | None = None

    @property
    def remotes(self):
        return get_remotes_of_events(self.followings)

    def events(self, remote: str):
        return get_events_of_remote(self.followings, remote=remote)

    def to_pb(self, remote: str) -> Command:
        events = self.events(remote)
        return Command(
            chains=[
                OperationChain(operations=[f.to_pb() for f in fs]) for fs in events
            ],
            text=self.text,
            sound=self.sound.to_pb() if self.sound else None,
        )

    @classmethod
    def from_input(cls, worker_input: "WorkerInput"):
        return cls(
            input_id=worker_input.input_id,
            followings=worker_input.following_events,
        )

    @classmethod
    def from_routes_list(cls, routes_list: RoutesList):
        return cls(
            input_id=uuid4(),
            followings=[
                [EventAddress.from_string(d) for d in ds] for ds in routes_list
            ],
        )


class SoundInput(BaseModel):
    data: bytes
    rate: int
    format: SampleFormat
    channels: int

    class Config:
        orm_mode = True

    def is_invalid(self):
        return any(
            (
                self.rate <= 0,
                self.channels == 0,
                not self.data,
                self.format == SampleFormat.INVALID,
            )
        )

    @classmethod
    def invalid(cls) -> "SoundInput":
        return cls(data=b"", rate=0, format=SampleFormat.INVALID, channels=0)


class WorkerInput(BaseModel):
    input_id: UUID
    current_event: EventType
    following_events: FollowingEvents
    text: str
    sound: SoundInput
    file_path: str
    filters: Iterable[str]

    @root_validator(pre=False, skip_on_failure=True)
    @classmethod
    def root_validator(cls, values: dict[str, Any]):
        sound = cast(SoundInput, values.get("sound"))
        event = cast(EventType, values.get("current_event"))
        if is_sound_event(event) and sound.is_invalid():
            raise ValueError("sound input is invalid")
        if event == EventType.reload and not values.get("file_path"):
            raise ValueError("file_path is required")
        return values

    @classmethod
    def from_output(cls, output: WorkerOutput, remote: str):
        first_event_maps = get_first_event_map(output.events(remote))
        return [
            WorkerInput(
                input_id=output.input_id,
                current_event=first_event.event,
                following_events=following_events,
                text=output.text or "",
                sound=SoundInput.from_orm(output.sound)
                if output.sound
                else SoundInput.invalid(),
                file_path="",
                filters=[],
            )
            for first_event, following_events in first_event_maps.items()
        ]

    @classmethod
    def from_command(cls, command: Command) -> list["WorkerInput"]:
        input_id = uuid4()
        events = command_to_events(command)
        first_event_maps = get_first_event_map(events)
        return [
            WorkerInput(
                input_id=input_id,
                current_event=first_event.event,
                following_events=following_events,
                text=command.text,
                sound=SoundInput.from_orm(command.sound),
                file_path=command.file_path,
                filters=command.filters,
            )
            for first_event, following_events in first_event_maps.items()
        ]


InputQueues = Dict[EventType, Queue[WorkerInput]]


@dataclass
class SharedContext:
    config: Config
    input_queues: InputQueues = field(default_factory=dict)
    running: Event = field(default_factory=Event)
    sender_queue: Queue[WorkerOutput] = field(default_factory=Queue)
    worker_need_reload: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        self.running.set()

    @property
    def need_reload(self) -> bool:
        task = current_task()
        if not task:
            return False
        return self.worker_need_reload.get(task.get_name(), False)

    def reset_need_reload(self):
        task = current_task()
        if not task:
            return
        self.worker_need_reload[task.get_name()] = False
