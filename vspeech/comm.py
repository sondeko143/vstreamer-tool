import json
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import StreamReader
from asyncio import StreamWriter
from asyncio import start_server
from enum import Enum
from functools import partial
from locale import getpreferredencoding
from typing import Any
from uuid import uuid4

from vspeech.config import Config
from vspeech.logger import logger
from vspeech.shared_context import EventType
from vspeech.shared_context import Message
from vspeech.shared_context import SharedContext
from vspeech.transcription import Transcription


class Method(Enum):
    t = "t"
    p = "p"
    r = "r"
    l = "l"


async def handle_message(
    context: SharedContext,
    out_queue: Queue[Message[Any]],
    reader: StreamReader,
    writer: StreamWriter,
):
    data = await reader.readline()
    try:
        message = data.decode(getpreferredencoding(), errors="replace")
        method = Method(message[0])
        if method == Method.t:
            text = message[1:].rstrip()
            text = text.replace("\n", "")
            if not text:
                return
            transcription_id = str(uuid4())
            transcription = Transcription(id=transcription_id, text=text, spoken=text)
            out_queue.put_nowait(
                Message(source=EventType.transcription, content=transcription)
            )
        elif method == Method.p:
            logger.info("pause")
            context.resume.clear()
        elif method == Method.r:
            logger.info("resume")
            context.resume.set()
        elif method == Method.l:
            logger.info("reload")
            file_path = message[1:].rstrip()
            with open(file_path, "r") as f:
                config_obj = json.loads(f.read())
                context.config = Config.parse_obj(config_obj)
            for worker_name in context.reload.keys():
                context.reload[worker_name] = True
        else:
            pass
    except UnicodeDecodeError as e:
        logger.exception(e)
    writer.close()


async def sock_reader_worker(context: SharedContext, out_queue: Queue[Message[Any]]):
    server = await start_server(
        partial(handle_message, context, out_queue), "localhost", context.config.port
    )
    addrs = ",".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info(f"serving on {addrs}")

    try:
        async with server:
            await server.serve_forever()
    except CancelledError:
        logger.info("sock reader worker cancelled")
        raise


def create_comm_worker(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    return loop.create_task(
        sock_reader_worker(context=context, out_queue=context.broker_queue),
        name="comm_task",
    )
