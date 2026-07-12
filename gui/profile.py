import toml
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from gui.migration import CONFIG_MIGRATIONS
from gui.migration import CURRENT_CONFIG_VERSION
from gui.migration import CURRENT_PROFILE_VERSION
from gui.migration import PROFILE_MIGRATIONS
from gui.migration import LoadResult
from gui.migration import migrate_dict
from gui.migration import quarantine
from gui.paths import ProfilePaths
from vspeech.config import Config
from vspeech.logger import logger


class PipelineEntry(BaseModel):
    id: str
    name: str
    port: int
    recipe: str
    config_version: int = CURRENT_CONFIG_VERSION


class Profile(BaseModel):
    profile_version: int = CURRENT_PROFILE_VERSION
    default_config_version: int = CURRENT_CONFIG_VERSION
    pipelines: list[PipelineEntry] = Field(default_factory=list)


def save_default_config(paths: ProfilePaths, config: Config) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.default_config.write_text(config.export_to_toml(), encoding="utf-8")


def load_default_config(paths: ProfilePaths) -> Config:
    path = paths.default_config
    if not path.exists():
        config = Config()
        save_default_config(paths, config)
        return config
    text = path.read_text(encoding="utf-8")
    try:
        data, _ = migrate_dict(
            toml.loads(text),
            from_version=0,
            migrations=CONFIG_MIGRATIONS,
            current=CURRENT_CONFIG_VERSION,
        )
        return Config.model_validate(data)
    except (toml.TomlDecodeError, ValidationError) as e:
        backup = quarantine(path)
        logger.warning("default.toml 破損: %s に退避し既定へ fallback (%s)", backup, e)
        config = Config()
        save_default_config(paths, config)
        return config


def save_profile(paths: ProfilePaths, profile: Profile) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(toml.dumps(profile.model_dump()), encoding="utf-8")


def load_profile(paths: ProfilePaths) -> Profile:
    path = paths.manifest
    if not path.exists():
        return Profile()
    text = path.read_text(encoding="utf-8")
    try:
        raw = toml.loads(text)
        from_version = int(raw.get("profile_version", 0))
        data, _ = migrate_dict(
            raw,
            from_version=from_version,
            migrations=PROFILE_MIGRATIONS,
            current=CURRENT_PROFILE_VERSION,
        )
        return Profile.model_validate(data)
    except (toml.TomlDecodeError, ValidationError, ValueError) as e:
        backup = quarantine(path)
        logger.warning(
            "pipelines.toml 破損: %s に退避し空プロファイルへ (%s)", backup, e
        )
        return Profile()


def save_pipeline_config(
    paths: ProfilePaths, entry: PipelineEntry, config: Config
) -> None:
    config.listen_port = entry.port
    paths.pipelines_dir.mkdir(parents=True, exist_ok=True)
    paths.pipeline_config(entry.id).write_text(
        config.export_to_toml(), encoding="utf-8"
    )
    entry.config_version = CURRENT_CONFIG_VERSION


def load_pipeline_config(
    paths: ProfilePaths, entry: PipelineEntry
) -> LoadResult[Config]:
    path = paths.pipeline_config(entry.id)
    if not path.exists():
        return LoadResult(ok=False, error=f"config file not found: {path}", raw_text="")
    text = path.read_text(encoding="utf-8")
    try:
        data, _ = migrate_dict(
            toml.loads(text),
            from_version=entry.config_version,
            migrations=CONFIG_MIGRATIONS,
            current=CURRENT_CONFIG_VERSION,
        )
        config = Config.model_validate(data)
    except (toml.TomlDecodeError, ValidationError) as e:
        backup = quarantine(path)
        return LoadResult(
            ok=False, error=str(e), raw_text=text, quarantined_path=backup
        )
    return LoadResult(
        ok=True,
        value=config,
        raw_text=text,
        migrated=entry.config_version < CURRENT_CONFIG_VERSION,
    )
