import logging

import pytest

# translation.py imports google-cloud-translate at import time; skip cleanly
# where it isn't installed.
pytest.importorskip("google.cloud.translate_v3")

from google.api_core.exceptions import BadRequest  # noqa: E402
from google.api_core.exceptions import ServiceUnavailable  # noqa: E402
from google.cloud.translate_v3 import TranslateTextRequest  # noqa: E402

from vspeech.worker.translation import translate_request  # noqa: E402


class FakeClient:
    """Async translate client whose behaviors are consumed one per call."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    async def translate_text(self, request, timeout):
        self.calls += 1
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


async def _call(client, **kw):
    kw.setdefault("timeout", 1.0)
    kw.setdefault("max_retry_count", 5)
    kw.setdefault("retry_delay_sec", 0)
    return await translate_request(client=client, request=TranslateTextRequest(), **kw)


async def test_retries_transient_then_succeeds():
    client = FakeClient([ServiceUnavailable("boom"), ServiceUnavailable("boom"), "OK"])
    result = await _call(client)
    assert result == "OK"
    assert client.calls == 3


async def test_transient_recovery_logs_warning_without_traceback(caplog):
    client = FakeClient([ServiceUnavailable("boom"), "OK"])
    with caplog.at_level(logging.DEBUG):
        result = await _call(client)
    assert result == "OK"
    # A transient that recovers must NOT dump a full ERROR traceback.
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a concise retry warning"
    assert all(r.exc_info is None for r in warnings), "warning must carry no traceback"


async def test_raises_when_retries_exhausted():
    client = FakeClient([ServiceUnavailable("boom")] * 10)
    with pytest.raises(ServiceUnavailable):
        await _call(client, max_retry_count=2)
    assert client.calls == 3  # initial attempt + 2 retries


async def test_bad_request_not_retried():
    client = FakeClient([BadRequest("nope"), "OK"])
    with pytest.raises(BadRequest):
        await _call(client)
    assert client.calls == 1
