from asyncio import current_task
from contextlib import contextmanager
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
    # Dotted path of the offending setting (e.g. "rvc.model_file"). Names the bad
    # setting apart from the prose in `detail`. The only reader today is
    # tests/test_preflight.py (the GUI used it to jump here: ADR-0045, removed by
    # ADR-0061). Kept because it lets tests assert on the field instead of the prose.
    field: str | None = None

    def __str__(self) -> str:
        return f"[{self.worker}] {self.detail}"


class ConfigError(Exception):
    """Fatal config problems aggregated by preflight (raised before spawning tasks)."""

    def __init__(self, problems: list[ConfigProblem]):
        self.problems = problems
        super().__init__("; ".join(str(p) for p in problems))


class WorkerStartupError(Exception):
    """A worker could not acquire a real resource at startup (layer B, deep failure)."""

    def __init__(self, worker: str, detail: str):
        self.worker = worker
        self.detail = detail
        super().__init__(f"[{worker}] {detail}")


class DeviceNotFoundError(Exception):
    """The audio device named in the config cannot be resolved."""

    pass


class DeviceRateUnresolvedError(DeviceNotFoundError):
    """The device's true sample rate could not be determined (ADR-0071).

    A subclass of DeviceNotFoundError so the existing preflight handlers keep
    catching it, while callers that care can tell the two apart.
    """


@contextmanager
def worker_startup(worker: str):
    """Convert a resource-acquisition failure at worker startup into a
    WorkerStartupError (layer B, ADR-0038)."""
    try:
        yield
    except WorkerStartupError:
        raise
    except Exception as e:
        raise WorkerStartupError(worker, str(e)) from e
