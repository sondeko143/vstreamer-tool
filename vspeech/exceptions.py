from asyncio import current_task


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
