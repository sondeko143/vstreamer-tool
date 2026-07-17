from pathlib import Path
from typing import Any
from typing import get_args

from pydantic import SecretStr

from vspeech.config import Config


def get_value(config: Config, path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        node = getattr(node, part)
    if isinstance(node, SecretStr):
        return node.get_secret_value()
    return node


def coerce_value(node: Any, child: str, value: Any) -> Any:
    # None passes through cleanly (read_into only sends None for Optional
    # fields). Blank strings never reach here — read_into intercepts them — so
    # Path/SecretStr always wrap a real value.
    if value is None:
        return None
    annotation = type(node).model_fields[child].annotation
    types = get_args(annotation) or (annotation,)
    if SecretStr in types:
        return SecretStr(str(value))
    if Path in types:
        return Path(str(value))
    return value


def field_types(config: Config, path: str) -> tuple[Any, ...]:
    """The pydantic annotation of `path`'s field, flattened to a tuple of types
    (e.g. `int | None` -> `(int, NoneType)`, a bare `int` -> `(int,)`)."""
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    annotation = type(node).model_fields[child].annotation
    return get_args(annotation) or (annotation,)


def set_value(config: Config, path: str, value: Any) -> None:
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, child, coerce_value(node, child, value))
