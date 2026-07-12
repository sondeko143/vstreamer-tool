from dataclasses import dataclass
from pathlib import Path

import platformdirs


def default_root() -> Path:
    return Path(
        platformdirs.user_config_dir("vstreamer", appauthor=False, roaming=True)
    )


@dataclass(frozen=True)
class ProfilePaths:
    root: Path

    @property
    def default_config(self) -> Path:
        return self.root / "default.toml"

    @property
    def manifest(self) -> Path:
        return self.root / "pipelines.toml"

    @property
    def pipelines_dir(self) -> Path:
        return self.root / "pipelines"

    def pipeline_config(self, pipeline_id: str) -> Path:
        return self.pipelines_dir / f"{pipeline_id}.toml"


def resolve_paths(root: Path | None = None) -> ProfilePaths:
    return ProfilePaths(root=root if root is not None else default_root())
