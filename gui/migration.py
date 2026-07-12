from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_CONFIG_VERSION = 1
CURRENT_PROFILE_VERSION = 1

type MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Migration:
    to_version: int
    describe: str
    apply: MigrationFn


# Phase 1 baseline == version 1, so there are no real migrations yet. Future
# field additions/renames add steps with to_version >= 2. Write each `apply`
# to be shape-detecting and idempotent (no-op if the old shape is absent) so a
# full re-run from version 0 stays safe when the recorded version is unknown.
CONFIG_MIGRATIONS: list[Migration] = []
PROFILE_MIGRATIONS: list[Migration] = []


def migrate_dict(
    data: dict[str, Any],
    from_version: int,
    migrations: list[Migration],
    current: int,
) -> tuple[dict[str, Any], int]:
    result = dict(data)
    for migration in sorted(migrations, key=lambda m: m.to_version):
        if migration.to_version > from_version:
            result = migration.apply(result)
    return result, current


def quarantine(path: Path) -> Path:
    n = 1
    while True:
        backup = path.with_name(f"{path.name}.bak-{n}")
        if not backup.exists():
            break
        n += 1
    backup.write_bytes(path.read_bytes())
    return backup


@dataclass
class LoadResult[T]:
    ok: bool
    value: T | None = None
    error: str | None = None
    raw_text: str | None = None
    migrated: bool = False
    quarantined_path: Path | None = None
