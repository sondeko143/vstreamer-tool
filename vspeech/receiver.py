import re
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import cast

import grpc
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
from vstreamer_protos.commander.commander_pb2 import Response
from vstreamer_protos.commander.commander_pb2_grpc import CommanderServicer
from vstreamer_protos.commander.commander_pb2_grpc import (
    add_CommanderServicer_to_server,
)

from vspeech.config import Config
from vspeech.config import ReplaceFilter
from vspeech.exceptions import ReplaceFilterParseError
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import InputQueues
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


def text_filter(filters: List[ReplaceFilter], text: Optional[str]):
    if not text:
        return ""
    replaced_text = f"{text}"
    for replace_filter in filters:
        replaced_text = re.sub(
            replace_filter.pattern, replace_filter.replaced, replaced_text
        )
    return replaced_text


def transform_content(command: WorkerInput, filters: List[ReplaceFilter]):
    if any(
        o in command.operations
        for o in [
            TRANSLATE,
            SUBTITLE,
            SUBTITLE_TRANSLATED,
            SPEECH,
        ]
    ):
        command.text = text_filter(filters=filters, text=command.text)


def put_queue(input_queues: InputQueues, dest_event: EventType, request: WorkerInput):
    try:
        queue = input_queues[dest_event]
        queue.put_nowait(request)
    except KeyError:
        logger.info("worker %s not activated", dest_event)


def process_command(context: SharedContext, request: WorkerInput):
    transform_content(filters=context.config.filters, command=request)
    if TRANSCRIBE in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.transcription,
            request=request,
        )
    if TRANSLATE in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.translation,
            request=request,
        )
    if SUBTITLE in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.subtitle,
            request=request,
        )
    if SUBTITLE_TRANSLATED in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.subtitle_translated,
            request=request,
        )
    if SPEECH in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.speech,
            request=request,
        )
    if PLAYBACK in request.operations:
        put_queue(
            input_queues=context.input_queues,
            dest_event=EventType.playback,
            request=request,
        )
    if PAUSE in request.operations:
        logger.info("pause")
        context.resume.clear()
    if RESUME in request.operations:
        logger.info("resume")
        context.resume.set()
    if RELOAD in request.operations:
        logger.info("reload")
        file_path = request.file_path
        with open(file_path, "rb") as f:
            context.config = Config.read_config_from_file(f)
        for worker_name in context.reload.keys():
            context.reload[worker_name] = True
    if SET_FILTERS in request.operations:
        context.config.filters.clear()
        filters = request.filters
        for filter in filters:
            if not filter:
                continue
            try:
                context.config.filters.append(ReplaceFilter.from_str(filter))
            except ReplaceFilterParseError:
                logger.warning("ignore invalid filter string %s", filter)


@dataclass
class Commander(CommanderServicer):
    context: SharedContext

    def process_command(self, request: Command, context: grpc.ServicerContext):
        worker_input = WorkerInput.from_command(request)
        logger.info(
            "receive: o(%s), t(%s), sound(%s, %s, %s), %s, %s from %s",
            request.operations,
            request.text,
            request.sound.rate,
            request.sound.format,
            request.sound.channels,
            request.file_path,
            request.filters,
            cast(str, context.peer()),
        )
        process_command(self.context, worker_input)
        return Response(result=True)


async def receiver_worker(context: SharedContext):
    server = grpc.aio.server()
    add_CommanderServicer_to_server(servicer=Commander(context=context), server=server)
    address = f"{context.config.listen_address}:{context.config.listen_port}"
    server.add_insecure_port(address)
    logger.info("Starting server on %s", address)
    try:
        await server.start()
        await server.wait_for_termination()
    except CancelledError:
        logger.info("receiver worker cancelled")
        raise


def create_receiver_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    return loop.create_task(
        receiver_worker(context=context),
        name="receiver",
    )
