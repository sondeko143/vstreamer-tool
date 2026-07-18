import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import PIPE  # nosec B404 - launches local vspeech with fixed argv
from subprocess import STDOUT  # nosec B404 - see PIPE import
from subprocess import Popen  # nosec B404 - see PIPE import
from subprocess import TimeoutExpired  # nosec B404 - see PIPE import
from threading import Thread
from time import monotonic

import grpc
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2 import Operand
from vstreamer_protos.commander.commander_pb2 import OperationChain
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.config import RoutesList
from vspeech.logger import logger
from vspeech.shared_context import EventAddress


def build_argv(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "vspeech", "--config", str(config_path)]


def build_text_command(text: str, text_send_operations: RoutesList) -> Command:
    chains = []
    for ops in text_send_operations:
        operations = [EventAddress.from_string(op).to_pb() for op in ops if op]
        if operations:
            chains.append(OperationChain(operations=operations))
    return Command(chains=chains, operand=Operand(text=text.strip()))


class PipelineRunner:
    def __init__(
        self,
        config_path: Path,
        port: int,
        on_log: Callable[[str], None],
        on_exit: Callable[[int], None],
    ):
        self.config_path = config_path
        self.port = port
        self.on_log = on_log
        self.on_exit = on_exit
        self.proc: Popen[str] | None = None
        self.started_at: float | None = None
        # ユーザーが Stop / delete / window close で意図的に止めたか。
        # 即死 (起動失敗) と区別するために要る — terminate の exit code は
        # 非 0 なので、これが無いと通常停止まで「起動失敗」と誤認する。
        self.stopping = False

    def start(self) -> None:
        self.started_at = monotonic()
        self.stopping = False
        self.proc = Popen(  # nosec B603 - fixed argv, no shell, self-created config path
            build_argv(self.config_path),
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                self.on_log(line.rstrip())
        finally:
            self.on_exit(proc.wait())

    def ran_for(self) -> float:
        """start() からの経過秒。未起動なら 0.0。"""
        return 0.0 if self.started_at is None else monotonic() - self.started_at

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request_stop(self) -> None:
        """Send the terminate signal WITHOUT waiting for the process to die.

        Safe to call from the Tk main thread (e.g. on window close) — it never
        blocks the UI. The process dies asynchronously; `_pump` fires `on_exit`
        when it does.
        """
        self.stopping = True
        if self.proc and self.is_running():
            self.proc.terminate()

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the process, then wait up to `timeout`s, escalating to a
        kill if it ignores the terminate signal.

        This BLOCKS for up to `timeout`s, so a caller on the Tk main thread
        should run it off-thread (e.g. in a daemon Thread) to keep the UI
        responsive. `on_exit` is fired by `_pump`, not here.
        """
        self.stopping = True
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except TimeoutExpired:
            logger.warning("pipeline on port %s ignored terminate; killing", self.port)
            proc.kill()

    def send_text(self, text: str, text_send_operations: RoutesList) -> None:
        command = build_text_command(text, text_send_operations)
        with grpc.insecure_channel(f"127.0.0.1:{self.port}") as channel:
            CommanderStub(channel).process_command(command)
            logger.info("send: %s", command)
