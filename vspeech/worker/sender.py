from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from typing import List
from typing import cast
from urllib.parse import urlparse

from google.auth.exceptions import MutualTLSChannelError
from google.auth.exceptions import RefreshError
from google.auth.transport.grpc import AuthMetadataPlugin
from google.auth.transport.requests import Request
from google.oauth2.service_account import IDTokenCredentials
from grpc import composite_channel_credentials
from grpc import metadata_call_credentials
from grpc import ssl_channel_credentials
from grpc.aio import AioRpcError
from grpc.aio import insecure_channel
from grpc.aio import secure_channel
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Response
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.config import Config
from vspeech.exceptions import EventDestinationNotFoundError
from vspeech.lib.command import process_command
from vspeech.lib.gcp import get_id_token_credentials
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def get_event_destination(config: Config, source: EventType) -> List[str]:
    try:
        worker_config = getattr(config, source.value)
        destinations = worker_config.destinations
        logger.info("source: %s. destinations: %s", source, destinations)
        return destinations
    except AttributeError:
        raise EventDestinationNotFoundError(f"Unknown event name {source}")


def async_secure_authorized_channel(
    credentials: IDTokenCredentials, request: Request, target: str
):
    # async of google.auth.transport.grpc.secure_authorized_channel
    metadata_plugin = AuthMetadataPlugin(credentials, request)
    google_auth_credentials = metadata_call_credentials(metadata_plugin)
    ssl_credentials = ssl_channel_credentials()
    composite_credentials = composite_channel_credentials(
        ssl_credentials, google_auth_credentials
    )
    return secure_channel(target, composite_credentials)


def get_channel(address: str, credentials: IDTokenCredentials):
    url = urlparse(address)
    if url.scheme == "https" or url.port == 443:
        request = Request()
        id_token_cred: IDTokenCredentials = credentials.with_target_audience(
            f"https://{url.hostname}/"
        )
        id_token_cred.refresh(request)
        return async_secure_authorized_channel(
            credentials=id_token_cred, request=request, target=address.strip("/")
        )
    else:
        return insecure_channel(address.strip("/"))


async def send_command(
    credentials: IDTokenCredentials,
    address: str,
    command: Command,
):
    try:
        async with get_channel(address=address, credentials=credentials) as channel:
            stub = CommanderStub(channel)
            logger.info(
                "send: o(%s), s(%s), t(%s), to %s",
                command.chains,
                len(command.sound.data),
                command.text,
                address,
            )
            res = cast(Response, await stub.process_command(command))
            logger.info("success response: %s", res)
    except (RefreshError, MutualTLSChannelError, AioRpcError) as e:
        logger.warning(e)


async def sender(
    context: SharedContext,
    in_queue: Queue[WorkerOutput],
):
    credentials = get_id_token_credentials(context.config.gcp)
    logger.info("sender worker started")
    try:
        while True:
            try:
                worker_output = await in_queue.get()
                for remote in worker_output.remotes:
                    if remote:
                        await send_command(
                            credentials=credentials,
                            address=remote,
                            command=worker_output.to_pb(remote=remote),
                        )
                    else:
                        for worker_input in WorkerInput.from_output(
                            output=worker_output, remote=remote
                        ):
                            process_command(
                                context=context,
                                request=worker_input,
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
