"""The CLI's surface -- argument interpretation, output, and exit codes.

The sending itself is substituted, so this only observes "what the CLI tried to send" and
"how it reports the result". Whether it actually arrives is checked by test_client.py,
which pushes through to the receiving side.
"""

import pytest
from click.testing import CliRunner

from cli import main
from cli.client import SendResult
from cli.main import cmd
from cli.main import normalize_address
from vspeech.config import EventType


@pytest.fixture
def sent(monkeypatch):
    """Capture send() and record the call arguments. Returns success by default."""
    calls: list[dict] = []

    def fake_send(address, event, config_path="", timeout=None):
        calls.append(
            {
                "address": address,
                "event": event,
                "config_path": config_path,
                "timeout": timeout,
            }
        )
        return SendResult(ok=True, elapsed_ms=12.3)

    monkeypatch.setattr(main, "send", fake_send)
    return calls


def invoke(args, **kwargs):
    return CliRunner().invoke(cmd, args, **kwargs)


@pytest.mark.parametrize(
    ("command", "event"),
    [
        ("ping", EventType.ping),
        ("pause", EventType.pause),
        ("resume", EventType.resume),
    ],
)
def test_each_operation_sends_its_own_event(sent, command, event):
    result = invoke([command, "--to", "host.example:8080"])
    assert result.exit_code == 0
    assert sent[0]["event"] == event
    assert sent[0]["address"] == "host.example:8080"
    assert sent[0]["config_path"] == ""
    assert "OK" in result.output
    assert command in result.output


def test_success_prints_address_and_round_trip(sent):
    result = invoke(["ping", "--to", "host.example:8080"])
    assert result.output.strip() == "OK  host.example:8080  ping  12ms"


def test_failure_exits_nonzero_with_the_reason(monkeypatch):
    monkeypatch.setattr(
        main,
        "send",
        lambda *a, **k: SendResult(
            ok=False, elapsed_ms=3001.0, detail="DEADLINE_EXCEEDED: Deadline Exceeded"
        ),
    )
    result = invoke(["pause", "--to", "host.example:8080"])
    # The exit code is the outcome of the operation itself -- so it chains from a script.
    assert result.exit_code == 1
    assert "NG" in result.output
    assert "DEADLINE_EXCEEDED" in result.output


def test_target_falls_back_to_the_environment_variable(sent):
    result = invoke(["ping"], env={"VSPEECH_TARGET": "host.example:9000"})
    assert result.exit_code == 0
    assert sent[0]["address"] == "host.example:9000"


def test_missing_target_is_a_usage_error(sent):
    result = invoke(["ping"], env={"VSPEECH_TARGET": None})
    assert result.exit_code == 2
    assert not sent


def test_reload_requires_a_config_path(sent):
    result = invoke(["reload", "--to", "host.example:8080"])
    assert result.exit_code == 2
    assert not sent


def test_reload_rejects_a_blank_config_path(sent):
    result = invoke(["reload", "--to", "host.example:8080", "--config-path", "  "])
    assert result.exit_code == 2
    assert not sent


def test_reload_passes_the_remote_config_path_through(sent):
    result = invoke(
        ["reload", "--to", "host.example:8080", "--config-path", "D:/vs/config.toml"]
    )
    assert result.exit_code == 0
    assert sent[0]["event"] == EventType.reload
    assert sent[0]["config_path"] == "D:/vs/config.toml"


def test_timeout_is_forwarded(sent):
    invoke(["ping", "--to", "host.example:8080", "--timeout", "0.5"])
    assert sent[0]["timeout"] == 0.5


def test_a_target_without_a_port_is_rejected_before_sending(sent):
    # Letting a portless target through means no response from gRPC until the deadline.
    result = invoke(["ping", "--to", "host.example"])
    assert result.exit_code == 2
    assert not sent


@pytest.mark.parametrize(
    "address", ["host.example:0", "host.example:70000", "host.example:http", "::1:8080"]
)
def test_bad_addresses_are_rejected(sent, address):
    assert invoke(["ping", "--to", address]).exit_code == 2
    assert not sent


def test_normalize_address_keeps_bracketed_ipv6():
    assert normalize_address("[::1]:8080") == "[::1]:8080"
    assert normalize_address("  host.example:8080  ") == "host.example:8080"
