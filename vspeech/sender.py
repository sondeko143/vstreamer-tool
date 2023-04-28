from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from collections import defaultdict
from typing import DefaultDict
from typing import Iterable
from typing import List

import grpc
from vstreamer_protos.commander.commander_pb2 import PLAYBACK
from vstreamer_protos.commander.commander_pb2 import SPEECH
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import SUBTITLE_TRANSLATED
from vstreamer_protos.commander.commander_pb2 import TRANSCRIBE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operation
from vstreamer_protos.commander.commander_pb2 import Sound
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.config import Config
from vspeech.exceptions import EventDestinationNotFoundError
from vspeech.exceptions import EventToOperationConvertError
from vspeech.logger import logger
from vspeech.receiver import process_command
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def convert_event_to_operation(event: EventType):
    if event == EventType.transcription:
        return TRANSCRIBE
    if event == EventType.translation:
        return TRANSLATE
    if event == EventType.subtitle:
        return SUBTITLE
    if event == EventType.subtitle_translated:
        return SUBTITLE_TRANSLATED
    if event == EventType.speech:
        return SPEECH
    if event == EventType.playback:
        return PLAYBACK
    raise EventToOperationConvertError(f"Unsupported event type {event}")


def get_event_destination(config: Config, source: EventType) -> List[str]:
    try:
        worker_config = getattr(config, source.value)
        destinations = worker_config.destinations
        logger.info("source: %s. destinations: %s", source, destinations)
        return destinations
    except AttributeError:
        raise EventDestinationNotFoundError(f"Unknown event name {source}")


def get_operations_per_address(config: Config, source: EventType):
    targets = get_event_destination(config=config, source=source)
    operations_per_address: DefaultDict[str, List[Operation]] = defaultdict(list)
    for target in targets:
        if "/" in target:
            address, target_name = target.split("/")
        else:
            address = ""
            target_name = target
        try:
            target_event = EventType(target_name)
            operations_per_address[address].append(
                convert_event_to_operation(target_event)
            )
        except (ValueError, EventToOperationConvertError):
            logger.warning("unsupported event %s", target_name)
            continue
    return operations_per_address


async def send_command(
    address: str,
    operations: Iterable["Operation"],
    worker_output: WorkerOutput,
):
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            stub = CommanderStub(channel)
            logger.info("send: %s to %s", operations, address)
            await stub.process_command(
                Command(
                    operations=operations,
                    text=worker_output.text,
                    sound=Sound(
                        data=worker_output.sound.data,
                        rate=worker_output.sound.rate,
                        format=worker_output.sound.format,
                        channels=worker_output.sound.channels,
                    )
                    if worker_output.sound
                    else None,
                )
            )
    except grpc.aio.AioRpcError as e:
        logger.warning(e)


async def sender(
    context: SharedContext,
    in_queue: Queue[WorkerOutput],
):
    logger.info("sender worker started")
    try:
        while True:
            try:
                worker_output = await in_queue.get()
                operations_per_address = get_operations_per_address(
                    context.config, source=worker_output.source
                )
                for address, operations in operations_per_address.items():
                    if address:
                        await send_command(
                            address=address,
                            operations=operations,
                            worker_output=worker_output,
                        )
                    else:
                        process_command(
                            context=context,
                            request=WorkerInput.from_output(
                                output=worker_output, operations=operations
                            ),
                        )
            except EventDestinationNotFoundError as e:
                logger.warning("unsupported event: %s", e)
    except CancelledError:
        logger.info("sender worker cancelled")
        raise


def create_sender_task(loop: AbstractEventLoop, context: SharedContext):
    return loop.create_task(
        sender(context=context, in_queue=context.sender_queue), name="sender"
    )
