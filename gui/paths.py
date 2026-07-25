from dataclasses import dataclass
from pathlib import Path

import platformdirs


def default_root() -> Path:
    return Path(
        platformdirs.user_config_dir("vstreamer", appauthor=False, roaming=True)
    )


@dataclass(frozen=True)
class GuiPaths:
    root: Path

    @property
    def targets(self) -> Path:
        return self.root / "targets.toml"


def resolve_paths(root: Path | None = None) -> GuiPaths:
    return GuiPaths(root=root if root is not None else default_root())
