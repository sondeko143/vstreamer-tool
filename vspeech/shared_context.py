from asyncio import Event
from asyncio import Queue
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import TypeVar

from pyaudio import PyAudio

from vspeech.config import Config


class EventType(Enum):
    speech = "speech"
    subtitle = "subtitle"
    subtitle_translated = "subtitle_translated"
    transcription = "transcription"
    translation = "translation"
    recording = "recording"


EventRouting = Dict[EventType, List[EventType]]

T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    source: EventType
    content: T


@dataclass
class SharedContext:
    config: Config
    event_routing: EventRouting
    input_queues: Dict[EventType, Queue[Any]] = field(default_factory=dict)
    resume: Event = field(default_factory=Event)
    audio: PyAudio = field(default_factory=PyAudio)
    broker_queue: Queue[Message[Any]] = field(default_factory=Queue)
    reload: Dict[str, bool] = field(
        default_factory=lambda: {worker_name.name: False for worker_name in EventType}
    )

    def __post_init__(self):
        self.resume.set()
