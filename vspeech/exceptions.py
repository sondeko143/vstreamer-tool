from asyncio import current_task
from dataclasses import dataclass


class ReplaceFilterParseError(ValueError):
    pass


class EventDestinationNotFoundError(BaseException):
    pass


class EventToOperationConvertError(BaseException):
    pass


class WorkerShutdown(BaseException):
    pass


def get_task_name() -> str:
    t = current_task()
    return t.get_name() if t else "unknown"


def shutdown_worker(e: BaseException):
    return WorkerShutdown(get_task_name()).with_traceback(e.__traceback__)


@dataclass(frozen=True)
class ConfigProblem:
    worker: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.worker}] {self.detail}"


class ConfigError(Exception):
    """preflight が集約した致命的な設定不備（タスク spawn 前に送出）。"""

    def __init__(self, problems: list[ConfigProblem]):
        self.problems = problems
        super().__init__("; ".join(str(p) for p in problems))


class WorkerStartupError(Exception):
    """worker が起動時に実リソースを取得できなかった（層B の深層失敗）。"""

    def __init__(self, worker: str, detail: str):
        self.worker = worker
        self.detail = detail
        super().__init__(f"[{worker}] {detail}")


class DeviceNotFoundError(Exception):
    """設定で指定したオーディオデバイスが解決できない。"""

    pass
