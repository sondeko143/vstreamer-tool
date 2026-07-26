"""Wiring of the streaming VC subsystem (ADR-0050).

A self-contained subsystem outside Command/routing. capture (its own mic) -> conversion
-> transport -> continuous playback are bundled in an inner TaskGroup and started as a
single task. It uses neither `context.add_worker` nor `sender_queue` (it never rides on
the utterance-path routing). The heavy imports (capture/runner/playback, which pull in
sounddevice/torch) are deferred to startup so this module itself can be imported on CPU.
"""

from asyncio import CancelledError
from asyncio import Event
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from typing import Any
from uuid import uuid4

from vspeech.config import StreamVcConfig
from vspeech.config import StreamVcRole
from vspeech.exceptions import WorkerShutdown
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.stream_vc.transport import Transport


def _iter_leaves(exc: BaseException):
    """Flatten a (possibly nested) exception group into its leaf exceptions."""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _iter_leaves(sub)
    else:
        yield exc


def loops_for_role(role: StreamVcRole) -> frozenset[str]:
    """The set of loop names a role starts (a pure function = the single authority on
    this branch, ADR-0055)."""
    if role is StreamVcRole.producer:
        return frozenset({"capture", "vc"})
    if role is StreamVcRole.consumer:
        return frozenset({"playback"})
    return frozenset({"capture", "vc", "playback"})  # local


async def _build_transport(sv_config: StreamVcConfig) -> Transport:
    """Build the transport from the role. Creating a UDP endpoint is async.

    A bind/connect failure is fail-loud through worker_startup (never hide a config
    problem, ADR-0038). A config with role=producer/consumer whose transport_type is not
    udp is rejected by preflight (role != local implies udp). When a second network
    transport (TCP/bidi) arrives, branch on transport_type inside the producer/consumer
    arms below.
    """
    role = sv_config.role
    if role is StreamVcRole.local:
        from vspeech.stream_vc.transport import InProcessTransport

        return InProcessTransport(max_queued=sv_config.max_queued_blocks)
    with worker_startup("stream_vc"):
        if role is StreamVcRole.producer:
            from vspeech.stream_vc.udp import create_udp_producer_transport

            peer_host = sv_config.peer_host
            peer_port = sv_config.peer_port
            if peer_host is None or peer_port is None:
                raise ValueError(
                    "stream_vc.role=producer requires peer_host and peer_port"
                )
            return await create_udp_producer_transport(peer_host, peer_port)
        from vspeech.stream_vc.udp import create_udp_consumer_transport

        bind_port = sv_config.bind_port
        if bind_port is None:
            raise ValueError("stream_vc.role=consumer requires bind_port")
        return await create_udp_consumer_transport(
            sv_config.bind_host, bind_port, sv_config.max_queued_blocks
        )


async def _stream_vc_subsystem(context: SharedContext) -> None:
    sv_config = context.config.stream_vc
    role = sv_config.role
    runs = loops_for_role(role)
    session_id = uuid4().hex
    transport = await _build_transport(sv_config)
    try:
        async with TaskGroup() as tg:
            if "capture" in runs or "vc" in runs:
                from vspeech.stream_vc.capture import capture_loop
                from vspeech.stream_vc.capture import ms_to_samples
                from vspeech.stream_vc.runner import vc_loop

                hop = ms_to_samples(sv_config.block_ms)
                capture_queue: Queue[Any] = Queue(maxsize=sv_config.max_queued_blocks)
                vc_ready = Event()
                tg.create_task(
                    # context.running is passed not to stop capture but to keep the
                    # deliberate drops during a pause from being misreported as a
                    # backpressure anomaly (capture keeps running while paused =
                    # ADR-0050).
                    capture_loop(
                        sv_config, capture_queue, hop, vc_ready, context.running
                    ),
                    name="stream_vc_capture",
                )
                tg.create_task(
                    vc_loop(
                        context,
                        sv_config,
                        capture_queue,
                        transport,
                        session_id,
                        vc_ready,
                    ),
                    name="stream_vc_runner",
                )
            if role is StreamVcRole.local:
                from vspeech.stream_vc.playback import playback_loop

                tg.create_task(
                    playback_loop(sv_config, transport), name="stream_vc_playback"
                )
            elif role is StreamVcRole.consumer:
                from vspeech.stream_vc.consumer import network_playback_loop

                tg.create_task(
                    network_playback_loop(sv_config, transport),
                    name="stream_vc_playback",
                )
    except CancelledError as e:
        raise shutdown_worker(e)
    except BaseExceptionGroup as eg:
        # A child task (capture/vc/playback) failed and aborted the inner TaskGroup.
        # Streaming is an opt-in feature, so an unrecoverable fault in it is not
        # swallowed: take the whole process down (fail-loud -> a daemon restarts it;
        # ADR-0050). TaskGroup, however, folds the cause into a BaseExceptionGroup that
        # also contains the WorkerShutdown of cancelled siblings. To keep the real cause
        # (RuntimeError / an ORT Fail / WorkerStartupError, ...) from drowning in that
        # aggregation noise, log it explicitly once here and then **re-raise as-is**
        # (neither swallow nor restart). Sibling WorkerShutdowns are excluded: they are a
        # product of the cancel, not the cause. On a pure shutdown (all WorkerShutdown)
        # causes is empty = no extra logging, and main's except* WorkerShutdown handles
        # it as usual.
        causes = [e for e in _iter_leaves(eg) if not isinstance(e, WorkerShutdown)]
        for cause in causes:
            logger.error(
                "stream_vc subsystem: unrecoverable fault in an explicitly-enabled "
                "feature — failing the whole process (fail-loud, a daemon restarts "
                "it): %r",
                cause,
            )
        raise
    finally:
        transport.close()


def create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task[None]:
    return tg.create_task(_stream_vc_subsystem(context), name="stream_vc")
