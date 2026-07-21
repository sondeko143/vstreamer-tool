"""Regression tests for the vspeech entry point (`vspeech.main`)."""

import asyncio
from unittest.mock import patch

import pytest

from vspeech.main import cmd


def test_cmd_creates_its_own_event_loop_without_a_current_one():
    """Python 3.14 startup regression.

    On Python 3.14 `asyncio.get_event_loop()` raises `RuntimeError` when the
    calling thread has no *current* event loop (3.14 removed the implicit loop
    creation). `cmd()` must therefore create and set its own loop so that
    `python -m vspeech` starts — a path the unit tests, the RVC golden, and the
    audio smoke all skip because none of them run the entry point.

    Clear the current loop, then assert `cmd()` gets PAST loop acquisition:
    with the server coroutine mocked to return immediately, `cmd()` terminates
    via `SystemExit` (its `exit(1)` after the coro returns) rather than raising
    `RuntimeError` at loop acquisition.
    """

    async def _noop_coro(config):
        return

    # No current event loop in this thread -> the removed get_event_loop()
    # behaviour would raise RuntimeError here on 3.14.
    asyncio.set_event_loop(None)
    try:
        with (
            patch("vspeech.main.vspeech_coro", _noop_coro),
            patch("vspeech.main.configure_logger"),
            patch("vspeech.main.telemetry.configure"),
        ):
            # click types Command.callback as Optional; it is set here.
            assert cmd.callback is not None
            with pytest.raises(SystemExit):
                cmd.callback(config_file=None)
    finally:
        # cmd() leaves its (now closed) loop as current; reset so we don't
        # hand a closed loop to the next test.
        asyncio.set_event_loop(None)
