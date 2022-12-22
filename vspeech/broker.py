from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from typing import Any

from vspeech.logger import logger
from vspeech.shared_context import Message
from vspeech.shared_context import SharedContext


async def broker(
    context: SharedContext,
    in_queue: Queue[Message[Any]],
):
    routing = context.event_routing
    logger.info("broker worker started")
    try:
        while True:
            event = await in_queue.get()
            try:
                targets = routing[event.source]
                for target in targets:
                    queue = context.input_queues[target]
                    queue.put_nowait(event.content)
            except (KeyError, AttributeError):
                logger.warning("Unknown event name %s", event.source)
    except CancelledError:
        logger.info("broker worker cancelled")
        raise


def create_broker_worker(loop: AbstractEventLoop, context: SharedContext):
    return loop.create_task(
        broker(context=context, in_queue=context.broker_queue), name="broker"
    )
