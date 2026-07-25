"""CLI が送る Command が、受け側で意図した制御イベントになることの検証。

ここが本当の契約 — 送信側で Command を組むだけのテストは、受け側の
WorkerInput 変換や validation とずれても気付けない。build_command の出力を
実際に vspeech.lib.command.process_command まで通して効果を見る。
"""

import grpc
import pytest

from cli import client
from cli.client import SendResult
from cli.client import build_command
from cli.client import send
from vspeech.config import Config
from vspeech.config import EventType
from vspeech.config import RecordingConfig
from vspeech.lib.command import process_command
from vspeech.shared_context import SharedContext
from vspeech.shared_context import WorkerInput


def apply(context: SharedContext, event: EventType, config_path: str = "") -> None:
    """GUI の Command を受け側と同じ経路で解釈して適用する。"""
    for worker_input in WorkerInput.from_command(
        build_command(event, config_path=config_path)
    ):
        process_command(context, worker_input)


def test_ping_command_carries_a_single_ping_operation():
    inputs = WorkerInput.from_command(build_command(EventType.ping))
    assert len(inputs) == 1
    assert inputs[0].current_event.event == EventType.ping
    # 制御操作は後続を持たない。持つと受け側が別の worker へ流してしまう。
    assert inputs[0].following_events == [[]]


def test_pause_then_resume_toggles_the_running_gate():
    context = SharedContext(config=Config())
    assert context.running.is_set()
    apply(context, EventType.pause)
    assert not context.running.is_set()
    apply(context, EventType.resume)
    assert context.running.is_set()


def test_reload_reads_the_config_path_from_the_command(tmp_path):
    context = SharedContext(config=Config())
    assert context.config.recording.enable is False
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        Config(recording=RecordingConfig(enable=True)).export_to_toml(),
        encoding="utf-8",
    )
    apply(context, EventType.reload, config_path=str(config_file))
    assert context.config.recording.enable is True
    # reload は自分でゲートを閉じるが、閉じっぱなしにはしない。
    assert context.running.is_set()


def test_reload_without_a_config_path_is_rejected_by_the_receiver():
    # CLI 側で空パスを弾く理由 (main.reload の BadParameter) の裏付け。
    with pytest.raises(ValueError):
        WorkerInput.from_command(build_command(EventType.reload))


def test_send_reports_an_unreachable_target_as_failure(monkeypatch):
    class FakeRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "failed to connect"

    class Failing:
        def process_command(self, command, timeout=None):
            raise FakeRpcError()

    monkeypatch.setattr(client, "CommanderStub", lambda channel: Failing())
    result = send("127.0.0.1:1", EventType.ping, timeout=0.1)
    assert result.ok is False
    assert "UNAVAILABLE" in result.detail
    assert "failed to connect" in result.detail


def test_send_reports_success_with_a_round_trip_time(monkeypatch):
    class Ok:
        def process_command(self, command, timeout=None):
            assert timeout is not None  # deadline 無しの呼び出しは UI を固める
            return type("Response", (), {"result": True})()

    monkeypatch.setattr(client, "CommanderStub", lambda channel: Ok())
    result = send("127.0.0.1:8080", EventType.ping)
    assert result == SendResult(ok=True, elapsed_ms=result.elapsed_ms)
    assert result.elapsed_ms >= 0


def test_send_treats_a_false_result_as_failure(monkeypatch):
    class Refusing:
        def process_command(self, command, timeout=None):
            return type("Response", (), {"result": False})()

    monkeypatch.setattr(client, "CommanderStub", lambda channel: Refusing())
    result = send("127.0.0.1:8080", EventType.ping)
    assert result.ok is False
