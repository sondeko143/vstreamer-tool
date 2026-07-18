from asyncio import CancelledError
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull
from asyncio import TaskGroup
from collections.abc import Callable
from typing import cast
from urllib.parse import urlparse

from google.auth.exceptions import MutualTLSChannelError
from google.auth.exceptions import RefreshError
from google.auth.transport.grpc import AuthMetadataPlugin
from google.auth.transport.requests import Request
from grpc import composite_channel_credentials
from grpc import metadata_call_credentials
from grpc import ssl_channel_credentials
from grpc.aio import AioRpcError
from grpc.aio import Channel
from grpc.aio import insecure_channel
from grpc.aio import secure_channel
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Response
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.exceptions import EventDestinationNotFoundError
from vspeech.exceptions import shutdown_worker
from vspeech.lib.command import process_command
from vspeech.lib.gcp import GcpIDTokenCredentials
from vspeech.lib.gcp import build_auth_session
from vspeech.lib.gcp import get_id_token_credentials
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput

# gRPC の既定再接続バックオフ（min 20s / max 120s）は、receiver より先に sender を
# 起動して初回接続に失敗すると致命的に遅い: バックオフ待ちの間に来た RPC は
# wait_for_ready=False のため即失敗し、キャッシュ済みの前回接続エラー（例: WSA 10060
# Connection timed out）をそのまま返し続ける。チャネルは永続再利用（ADR-0004）なので
# 新規チャネルでバックオフがリセットされることもない。よって receiver 起動後も同じ
# エラーが数十秒〜最大2分出続ける。これを有界化し、冷起動からの復帰を高速化する。
#  - initial: 初回リトライまでの待ち
#  - min:     1回の接続試行のデッドライン下限（SYN 黙殺=10060 時に各試行が張り付く上限）
#  - max:     試行間バックオフの上限（既定 120s → 数秒へ）
RECONNECT_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.initial_reconnect_backoff_ms", 500),
    ("grpc.min_reconnect_backoff_ms", 1000),
    ("grpc.max_reconnect_backoff_ms", 5000),
]


def async_secure_authorized_channel(
    credentials: GcpIDTokenCredentials, request: Request, target: str
):
    # async of google.auth.transport.grpc.secure_authorized_channel
    metadata_plugin = AuthMetadataPlugin(credentials, request)
    google_auth_credentials = metadata_call_credentials(metadata_plugin)
    ssl_credentials = ssl_channel_credentials()
    composite_credentials = composite_channel_credentials(
        ssl_credentials, google_auth_credentials
    )
    return secure_channel(
        target, composite_credentials, options=RECONNECT_CHANNEL_OPTIONS
    )


def get_channel(address: str, credentials: GcpIDTokenCredentials | None):
    url = urlparse(address)
    secure_port = url.scheme == "https" or url.port == 443
    if secure_port and credentials:
        # retry 付き session を積んだ Request を使う (ADR-0048)。この 1 個の
        # Request が 2 箇所で効く: 直下の初回 refresh と、
        # async_secure_authorized_channel が組む AuthMetadataPlugin が以後
        # 行う更新 (同じ request を持ち回る)。素の Request() だと、どちらも
        # 約 1 時間 idle した死んだプール接続を掴んで落ちうる。
        request = Request(session=build_auth_session())
        id_token_cred: GcpIDTokenCredentials = credentials.with_target_audience(
            f"https://{url.hostname}/"
        )
        id_token_cred.refresh(request)
        return async_secure_authorized_channel(
            credentials=id_token_cred, request=request, target=address.strip("/")
        )
    else:
        if secure_port:
            logger.warning(
                "Could not obtain credentials, so transport with insecure channel."
            )
        return insecure_channel(address.strip("/"), options=RECONNECT_CHANNEL_OPTIONS)


REMOTE_QUEUE_MAXSIZE = 16


class RemoteSender:
    def __init__(
        self,
        remote: str,
        credentials: GcpIDTokenCredentials | None,
        maxsize: int = REMOTE_QUEUE_MAXSIZE,
    ):
        self.remote = remote
        self.credentials = credentials
        self.queue: Queue[Command] = Queue(maxsize=maxsize)
        self.channel: Channel | None = None

    def enqueue(self, command: Command):
        try:
            self.queue.put_nowait(command)
        except QueueFull:
            try:
                self.queue.get_nowait()
                logger.warning("drop oldest command for %s (queue full)", self.remote)
            except QueueEmpty:
                pass
            self.queue.put_nowait(command)

    async def _send(self, command: Command):
        try:
            if self.channel is None:
                self.channel = get_channel(self.remote, self.credentials)
            stub = CommanderStub(self.channel)
            logger.info(
                "send: s(%s), t(%s), to %s",
                len(command.operand.sound.data),
                command.operand.text,
                self.remote,
            )
            if command.chains[0].operations[0].operation == SUBTITLE:
                command.operand.sound.data = b""
            logger.debug("send: chains(%s)", command.chains)
            res = cast(Response, await stub.process_command(command))
            logger.info("success response: %s", str(res))
        except (RefreshError, MutualTLSChannelError, AioRpcError) as e:
            logger.warning("%s", e)
        except Exception as e:  # noqa: BLE001 - 宛先タスクを死なせない
            logger.warning("send error to %s: %s", self.remote, e)

    async def run(self):
        # Must only ever terminate via CancelledError. This task runs under the
        # nested TaskGroup in `sender`; any other exception escaping here would
        # cancel the dispatcher and surface as an ExceptionGroup that bypasses
        # `sender`'s `except CancelledError`, crashing the process instead of a
        # graceful WorkerShutdown. `_send` swallows all non-cancellation
        # Exceptions to uphold this invariant.
        try:
            while True:
                command = await self.queue.get()
                await self._send(command)
        finally:
            if self.channel is not None:
                try:
                    await self.channel.close()
                except Exception as e:  # noqa: BLE001 - クローズ失敗は無視
                    logger.debug("channel close error for %s: %s", self.remote, e)


def _dispatch_output(
    context: SharedContext,
    senders: dict[str, RemoteSender],
    credentials: GcpIDTokenCredentials | None,
    spawn: Callable[[RemoteSender], None],
    worker_output: WorkerOutput,
):
    for remote in worker_output.remotes:
        if remote:
            rs = senders.get(remote)
            if rs is None:
                rs = RemoteSender(remote=remote, credentials=credentials)
                spawn(rs)
                senders[remote] = rs
            rs.enqueue(worker_output.to_pb(remote=remote))
        else:
            for worker_input in WorkerInput.from_output(
                output=worker_output, remote=remote
            ):
                process_command(context=context, request=worker_input)


async def sender(
    context: SharedContext,
    in_queue: Queue[WorkerOutput],
):
    credentials = get_id_token_credentials(context.config.gcp)
    logger.info("sender worker started")
    try:
        async with TaskGroup() as send_tg:
            senders: dict[str, RemoteSender] = {}

            def spawn(rs: RemoteSender):
                send_tg.create_task(rs.run(), name=f"sender:{rs.remote}")

            while True:
                try:
                    worker_output = await in_queue.get()
                    _dispatch_output(
                        context=context,
                        senders=senders,
                        credentials=credentials,
                        spawn=spawn,
                        worker_output=worker_output,
                    )
                except EventDestinationNotFoundError as e:
                    logger.warning("unsupported event: %s", e)
    except CancelledError as e:
        raise shutdown_worker(e)


def create_sender_task(tg: TaskGroup, context: SharedContext):
    return tg.create_task(
        sender(context=context, in_queue=context.sender_queue), name="sender"
    )
