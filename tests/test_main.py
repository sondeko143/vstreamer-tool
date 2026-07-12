"""Regression tests for the vspeech entry point (`vspeech.main`)."""

import asyncio
from unittest.mock import patch

import pytest

from vspeech.main import cmd


def test_cmd_creates_its_own_event_loop_without_a_current_one():
    """Python 3.14 startup regression.

    `cmd()` used to call `asyncio.get_event_loop()`, which on Python 3.14
    raises `RuntimeError` when the calling thread has no *current* event loop
    (3.14 removed the implicit loop creation). That crashed
    `python -m vspeech` before any worker started — a path the unit tests,
    the RVC golden, and the audio smoke all skipped because none of them run
    the entry point.

    Reproduce the failing state by clearing the current loop, then assert
    `cmd()` gets PAST loop acquisition: with the server coroutine mocked to
    return immediately, `cmd()` terminates via `SystemExit` (its `exit(1)`
    after the coro returns) rather than raising `RuntimeError` at loop
    acquisition. Reverting to `get_event_loop()` makes this test fail on 3.14.
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
