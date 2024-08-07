from asyncio import CancelledError
from asyncio import TaskGroup
from dataclasses import dataclass
from typing import cast

from grpc import ServicerContext
from grpc.aio import server as grpc_aio_server
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Response
from vstreamer_protos.commander.commander_pb2_grpc import CommanderServicer
from vstreamer_protos.commander.commander_pb2_grpc import (
    add_CommanderServicer_to_server,
)

from vspeech.exceptions import shutdown_worker
from vspeech.lib.command import process_command
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


@dataclass
class Commander(CommanderServicer):
    context: SharedContext

    def process_command(self, request: Command, context: ServicerContext):
        logger.info(
            "receive: t(%s), sound(%s), %s, %s from %s",
            request.operand.text,
            len(request.operand.sound.data),
            request.operand.file_path,
            request.operand.filters,
            cast(str, context.peer()),
        )
        logger.debug("receive: chain(%s)", request.chains)
        for worker_input in WorkerInput.from_command(request):
            process_command(self.context, worker_input)
        return Response(result=True)

    async def sync_process_command(self, request: Command, context: ServicerContext):
        logger.info(
            "receive: t(%s), sound(%s), %s, %s from %s",
            request.operand.text,
            len(request.operand.sound.data),
            request.operand.file_path,
            request.operand.filters,
            cast(str, context.peer()),
        )
        logger.debug("receive: chain(%s)", request.chains)
        for worker_input in WorkerInput.from_command(request):
            process_command(self.context, worker_input)
        return Response(result=True)


async def receiver_worker(context: SharedContext):
    server = grpc_aio_server()
    add_CommanderServicer_to_server(servicer=Commander(context=context), server=server)
    address = f"{context.config.listen_address}:{context.config.listen_port}"
    server.add_insecure_port(address)
    logger.info("Starting server on %s", address)
    try:
        await server.start()
        await server.wait_for_termination()
    except CancelledError as e:
        raise shutdown_worker(e)


def create_receiver_task(
    tg: TaskGroup,
    context: SharedContext,
):
    return tg.create_task(
        receiver_worker(context=context),
        name="receiver",
    )
