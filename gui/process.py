import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import PIPE  # nosec B404 - launches local vspeech with fixed argv
from subprocess import STDOUT  # nosec B404 - see PIPE import
from subprocess import Popen  # nosec B404 - see PIPE import
from threading import Thread

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
    chains = [
        OperationChain(
            operations=[EventAddress.from_string(op).to_pb() for op in ops if op]
        )
        for ops in text_send_operations
        if ops
    ]
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

    def start(self) -> None:
        self.proc = Popen(  # nosec B603 - fixed argv, no shell, self-created config path
            build_argv(self.config_path),
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1,
        )
        Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            self.on_log(line.rstrip())
        self.on_exit(proc.wait())

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc and self.is_running():
            self.proc.terminate()
            self.proc.wait()

    def send_text(self, text: str, text_send_operations: RoutesList) -> None:
        command = build_text_command(text, text_send_operations)
        with grpc.insecure_channel(f"127.0.0.1:{self.port}") as channel:
            CommanderStub(channel).process_command(command)
            logger.info("send: %s", command)
