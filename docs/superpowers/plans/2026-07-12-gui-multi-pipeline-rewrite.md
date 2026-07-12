# Multi-pipeline GUI Rewrite (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `gui/` so it manages N independent vspeech subprocesses ("pipelines") from a single platformdirs-backed user profile, with preset recipes, a minimal essential-fields form + raw TOML editor, auto free-port allocation, and a GUI-only config migration + corrupt-file quarantine mechanism.

**Architecture:** A profile (fixed platformdirs directory) holds a `default.toml` template, a GUI-only `pipelines.toml` manifest, and one pure-`Config` file per pipeline under `pipelines/`. Pure logic (paths, ports, migration, recipes, profile I/O, process/gRPC command building) is TDD-tested with no Tkinter dependency; the Tk UI layer (widgets, form, raw editor, editor panel, app window) is built on top and verified by manual smoke tests.

**Tech Stack:** Python 3.14, Tkinter via ttkbootstrap, pydantic v2, platformdirs, toml, grpc + vstreamer-protos, pytest.

## Global Constraints

- Python **>=3.14,<3.15** only. Do not lower the floor.
- **pydantic v2 only** — `model_config`/`model_validate`/`model_dump`/`model_copy`/`field_serializer`. Never reintroduce v1 APIs.
- **`Config` is `extra="forbid"`** (verified: `Config.model_validate({...,'config_version':3})` → `extra_forbidden`). Config files (`default.toml`, `pipelines/<id>.toml`) must stay pure `Config` shape. Versions live in the manifest, never in config files.
- Imports are **one-per-line** (ruff `force-single-line = true`) and auto-sorted. Type-checked by **ty** (Python 3.14).
- PEP 695 syntax is available and expected (`type X = ...`, `class C[T]`, `@dataclass class R[T]`).
- Dependency direction is one-way: `gui/` imports `vspeech/`, never the reverse.
- Tests are **Tk-independent pure logic only** under `tests/gui/`. Do not instantiate Tk widgets in tests.
- Health gate: `uv run --all-extras poe check` (ruff format/lint, ty, pytest, lock-check, audit, security, deadcode). Run tests with the `gui` extra synced.
- Machine-specific paths/hosts must not be hardcoded in committed docs; use `<USER>`/platformdirs abstraction.

---

## File Structure

**New pure-logic modules (TDD):**
- `gui/paths.py` — profile directory resolution (platformdirs) + file layout.
- `gui/ports.py` — free-port detection + allocation.
- `gui/migration.py` — versioning, migration chain, quarantine, `LoadResult`.
- `gui/recipes.py` — recipe registry (enable + route wiring).
- `gui/profile.py` — `PipelineEntry`/`Profile` models + safe load/save of manifest, default template, and per-pipeline configs.
- `gui/process.py` — `PipelineRunner` (subprocess + log thread) + `build_argv`/`build_text_command`.

**New Tk UI modules (build + manual verify):**
- `gui/widgets.py` — small reusable widgets (`Checkbutton`, `Textbox`, `Spinbox`, `ScrolledText`, `TextHandler`) salvaged from the old GUI.
- `gui/form.py` — minimal essential-fields form bound to a `Config`.
- `gui/rawedit.py` — raw TOML editor widget.
- `gui/pipeline_editor.py` — right panel: form/raw tabs + Start/Stop/status + text send + log, owning one `PipelineRunner`.
- `gui/app.py` — main window (pipeline list + editor) + click entry `main()`.
- `gui/__main__.py` — `from gui.app import main; main()`.

**Kept:** `gui/autocomplete_combobox.py` (device combos).
**Deleted:** `gui/gui.py`, `gui/dummy_param.py`.

**Tests:** `tests/gui/test_paths.py`, `test_ports.py`, `test_migration.py`, `test_recipes.py`, `test_profile.py`, `test_process.py`.

---

### Task 1: Dependency + profile paths

**Files:**
- Modify: `pyproject.toml` (add `platformdirs` to the `gui` extra)
- Create: `gui/paths.py`
- Test: `tests/gui/test_paths.py`
- Create: `tests/gui/__init__.py` (empty, if needed for discovery)

**Interfaces:**
- Produces:
  - `class ProfilePaths` (frozen dataclass) with `root: Path` and properties `default_config: Path`, `manifest: Path`, `pipelines_dir: Path`, and method `pipeline_config(pipeline_id: str) -> Path`.
  - `default_root() -> Path` — platformdirs `%APPDATA%\vstreamer` (Win) / `~/.config/vstreamer` (Linux).
  - `resolve_paths(root: Path | None = None) -> ProfilePaths`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, change the `gui` extra to include platformdirs:

```toml
gui = [
    "ttkbootstrap>=1.10.1,<2",
    "pillow>=12.1.1,<13",
    "platformdirs>=4,<5",
]
```

- [ ] **Step 2: Sync the extra**

Run: `uv sync --all-extras`
Expected: resolves and installs `platformdirs`. (If it errors with "os error 5" because a vspeech process holds the venv, stop and ask the user before killing live pipelines.)

- [ ] **Step 3: Write the failing test**

Create `tests/gui/__init__.py` (empty). Create `tests/gui/test_paths.py`:

```python
from pathlib import Path

from gui.paths import resolve_paths


def test_resolve_paths_layout(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    assert paths.root == tmp_path
    assert paths.default_config == tmp_path / "default.toml"
    assert paths.manifest == tmp_path / "pipelines.toml"
    assert paths.pipelines_dir == tmp_path / "pipelines"
    assert paths.pipeline_config("ab12") == tmp_path / "pipelines" / "ab12.toml"


def test_default_root_used_when_none(monkeypatch):
    import gui.paths as paths_mod

    monkeypatch.setattr(paths_mod, "default_root", lambda: Path("/fake/root"))
    paths = resolve_paths(None)
    assert paths.root == Path("/fake/root")
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.paths'`.

- [ ] **Step 5: Implement `gui/paths.py`**

```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_paths.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock gui/paths.py tests/gui/__init__.py tests/gui/test_paths.py
git commit -m "feat(gui): profile path resolution via platformdirs"
```

---

### Task 2: Free-port allocation

**Files:**
- Create: `gui/ports.py`
- Test: `tests/gui/test_ports.py`

**Interfaces:**
- Produces:
  - `is_port_free(port: int, host: str = "127.0.0.1") -> bool`.
  - `allocate_free_port(claimed: set[int], base: int = 8080, limit: int = 65535) -> int` — skips `claimed` and OS-busy ports, returns the first free port, raises `RuntimeError` if none.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/test_ports.py`:

```python
import pytest

import gui.ports as ports_mod
from gui.ports import allocate_free_port


def test_allocate_skips_claimed_and_busy(monkeypatch):
    busy = {8081}
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port, host="127.0.0.1": port not in busy)
    # 8080 is claimed, 8081 is OS-busy -> first free is 8082
    assert allocate_free_port(claimed={8080}, base=8080) == 8082


def test_allocate_first_when_all_free(monkeypatch):
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port, host="127.0.0.1": True)
    assert allocate_free_port(claimed=set(), base=8080) == 8080


def test_allocate_raises_when_none(monkeypatch):
    monkeypatch.setattr(ports_mod, "is_port_free", lambda port, host="127.0.0.1": False)
    with pytest.raises(RuntimeError):
        allocate_free_port(claimed=set(), base=8080, limit=8082)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_ports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.ports'`.

- [ ] **Step 3: Implement `gui/ports.py`**

```python
import socket


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def allocate_free_port(
    claimed: set[int], base: int = 8080, limit: int = 65535
) -> int:
    port = base
    while port <= limit:
        if port not in claimed and is_port_free(port):
            return port
        port += 1
    raise RuntimeError(f"no free port available from {base}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_ports.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/ports.py tests/gui/test_ports.py
git commit -m "feat(gui): auto-allocate free ports skipping claimed and OS-busy"
```

---

### Task 3: Migration primitives

**Files:**
- Create: `gui/migration.py`
- Test: `tests/gui/test_migration.py`

**Interfaces:**
- Produces:
  - `CURRENT_CONFIG_VERSION: int = 1`, `CURRENT_PROFILE_VERSION: int = 1`.
  - `type MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]`.
  - `@dataclass class Migration` with `to_version: int`, `describe: str`, `apply: MigrationFn`.
  - `CONFIG_MIGRATIONS: list[Migration]` (empty in phase 1), `PROFILE_MIGRATIONS: list[Migration]` (empty).
  - `migrate_dict(data, from_version, migrations, current) -> tuple[dict[str, Any], int]`.
  - `quarantine(path: Path) -> Path` — non-destructive `.bak-<n>` backup.
  - `@dataclass class LoadResult[T]` with `ok: bool`, `value: T | None = None`, `error: str | None = None`, `raw_text: str | None = None`, `migrated: bool = False`, `quarantined_path: Path | None = None`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/test_migration.py`:

```python
from pathlib import Path

from gui.migration import Migration
from gui.migration import migrate_dict
from gui.migration import quarantine


def test_migrate_dict_applies_only_newer_steps():
    m2 = Migration(2, "add b", lambda d: {**d, "b": 2})
    m3 = Migration(3, "add c", lambda d: {**d, "c": 3})
    out, ver = migrate_dict({"a": 1}, from_version=1, migrations=[m3, m2], current=3)
    assert out == {"a": 1, "b": 2, "c": 3}
    assert ver == 3


def test_migrate_dict_skips_already_applied():
    m2 = Migration(2, "add b", lambda d: {**d, "b": 2})
    m3 = Migration(3, "add c", lambda d: {**d, "c": 3})
    out, _ = migrate_dict({"a": 1}, from_version=2, migrations=[m2, m3], current=3)
    assert out == {"a": 1, "c": 3}


def test_migrate_dict_empty_chain_is_identity():
    out, ver = migrate_dict({"a": 1}, from_version=0, migrations=[], current=1)
    assert out == {"a": 1}
    assert ver == 1


def test_quarantine_is_non_destructive_and_increments(tmp_path: Path):
    p = tmp_path / "x.toml"
    p.write_text("first", encoding="utf-8")
    b1 = quarantine(p)
    assert b1.name == "x.toml.bak-1"
    assert b1.read_text(encoding="utf-8") == "first"
    assert p.read_text(encoding="utf-8") == "first"
    p.write_text("second", encoding="utf-8")
    b2 = quarantine(p)
    assert b2.name == "x.toml.bak-2"
    assert b2.read_text(encoding="utf-8") == "second"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_migration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.migration'`.

- [ ] **Step 3: Implement `gui/migration.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_migration.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/migration.py tests/gui/test_migration.py
git commit -m "feat(gui): config migration chain, quarantine, and LoadResult"
```

---

### Task 4: Recipes

**Files:**
- Create: `gui/recipes.py`
- Test: `tests/gui/test_recipes.py`

**Interfaces:**
- Consumes: `vspeech.config.Config`.
- Produces:
  - `@dataclass class Recipe` with `key: str`, `label: str`, `apply: Callable[[Config], Config]`.
  - `RECIPES: list[Recipe]` and `RECIPES_BY_KEY: dict[str, Recipe]`.
  - Recipe keys: `mic_loopback`, `mic_transcribe_tts`, `mic_vc`, `text_tts`, `blank`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/test_recipes.py`:

```python
from vspeech.config import Config

from gui.recipes import RECIPES_BY_KEY


def test_mic_loopback():
    out = RECIPES_BY_KEY["mic_loopback"].apply(Config())
    assert out.recording.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["playback"]]
    assert out.transcription.enable is False


def test_mic_transcribe_tts():
    out = RECIPES_BY_KEY["mic_transcribe_tts"].apply(Config())
    assert out.recording.enable is True
    assert out.transcription.enable is True
    assert out.tts.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["transcription", "tts", "playback"]]


def test_mic_vc():
    out = RECIPES_BY_KEY["mic_vc"].apply(Config())
    assert out.recording.enable is True
    assert out.vc.enable is True
    assert out.playback.enable is True
    assert out.recording.routes_list == [["vc", "playback"]]


def test_text_tts():
    out = RECIPES_BY_KEY["text_tts"].apply(Config())
    assert out.recording.enable is False
    assert out.tts.enable is True
    assert out.playback.enable is True
    assert out.text_send_operations == [["tts", "playback"]]


def test_blank_leaves_disabled():
    out = RECIPES_BY_KEY["blank"].apply(Config())
    assert out.recording.enable is False
    assert out.tts.enable is False


def test_apply_does_not_mutate_input():
    base = Config()
    RECIPES_BY_KEY["mic_loopback"].apply(base)
    assert base.recording.enable is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_recipes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.recipes'`.

- [ ] **Step 3: Implement `gui/recipes.py`**

```python
from collections.abc import Callable
from dataclasses import dataclass

from vspeech.config import Config


@dataclass
class Recipe:
    key: str
    label: str
    apply: Callable[[Config], Config]


def _mic_loopback(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["playback"]]
    return config


def _mic_transcribe_tts(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.transcription.enable = True
    config.tts.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["transcription", "tts", "playback"]]
    return config


def _mic_vc(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = True
    config.vc.enable = True
    config.playback.enable = True
    config.recording.routes_list = [["vc", "playback"]]
    return config


def _text_tts(base: Config) -> Config:
    config = base.model_copy(deep=True)
    config.recording.enable = False
    config.tts.enable = True
    config.playback.enable = True
    config.text_send_operations = [["tts", "playback"]]
    return config


def _blank(base: Config) -> Config:
    return base.model_copy(deep=True)


RECIPES: list[Recipe] = [
    Recipe("mic_loopback", "マイク→再生 (モニター)", _mic_loopback),
    Recipe("mic_transcribe_tts", "マイク→文字起こし→読み上げ→再生", _mic_transcribe_tts),
    Recipe("mic_vc", "マイク→ボイチェン→再生", _mic_vc),
    Recipe("text_tts", "テキスト→読み上げ→再生", _text_tts),
    Recipe("blank", "空 (default のまま)", _blank),
]

RECIPES_BY_KEY: dict[str, Recipe] = {recipe.key: recipe for recipe in RECIPES}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_recipes.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/recipes.py tests/gui/test_recipes.py
git commit -m "feat(gui): pipeline recipes wiring workers and routes"
```

---

### Task 5: Profile models + safe load/save

**Files:**
- Create: `gui/profile.py`
- Test: `tests/gui/test_profile.py`

**Interfaces:**
- Consumes: `ProfilePaths` (Task 1), migration primitives (Task 3), `vspeech.config.Config`.
- Produces:
  - `class PipelineEntry(BaseModel)`: `id: str`, `name: str`, `port: int`, `recipe: str`, `config_version: int = CURRENT_CONFIG_VERSION`.
  - `class Profile(BaseModel)`: `profile_version: int`, `default_config_version: int`, `pipelines: list[PipelineEntry]`.
  - `load_default_config(paths) -> Config` (creates when missing, quarantine+fallback on corruption).
  - `save_default_config(paths, config) -> None`.
  - `load_profile(paths) -> Profile` (empty when missing, quarantine+empty on corruption).
  - `save_profile(paths, profile) -> None`.
  - `load_pipeline_config(paths, entry) -> LoadResult[Config]`.
  - `save_pipeline_config(paths, entry, config) -> None` (injects `entry.port` into `config.listen_port`, sets `entry.config_version` to current).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/test_profile.py`:

```python
from pathlib import Path

from vspeech.config import Config

from gui.paths import resolve_paths
from gui.profile import PipelineEntry
from gui.profile import Profile
from gui.profile import load_default_config
from gui.profile import load_pipeline_config
from gui.profile import load_profile
from gui.profile import save_pipeline_config
from gui.profile import save_profile


def test_default_config_created_when_missing(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    config = load_default_config(paths)
    assert isinstance(config, Config)
    assert paths.default_config.exists()


def test_default_config_corrupt_falls_back(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.default_config.write_text("this = = not valid toml", encoding="utf-8")
    config = load_default_config(paths)
    assert isinstance(config, Config)
    assert (tmp_path / "default.toml.bak-1").exists()


def test_profile_roundtrip(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    profile = Profile(
        pipelines=[PipelineEntry(id="ab12", name="p1", port=8080, recipe="mic_vc")]
    )
    save_profile(paths, profile)
    loaded = load_profile(paths)
    assert loaded.pipelines[0].id == "ab12"
    assert loaded.pipelines[0].port == 8080
    assert loaded.pipelines[0].recipe == "mic_vc"


def test_profile_missing_is_empty(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    assert load_profile(paths).pipelines == []


def test_profile_corrupt_falls_back_to_empty(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text("= broken", encoding="utf-8")
    assert load_profile(paths).pipelines == []
    assert (tmp_path / "pipelines.toml.bak-1").exists()


def test_pipeline_config_roundtrip_injects_port(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    entry = PipelineEntry(id="ab12", name="p1", port=8085, recipe="blank")
    save_pipeline_config(paths, entry, Config())
    result = load_pipeline_config(paths, entry)
    assert result.ok is True
    assert result.value is not None
    assert result.value.listen_port == 8085


def test_pipeline_config_corrupt_surfaces_raw(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.pipelines_dir.mkdir(parents=True, exist_ok=True)
    entry = PipelineEntry(id="ab12", name="p1", port=8085, recipe="blank")
    paths.pipeline_config("ab12").write_text("listen_port = = broken", encoding="utf-8")
    result = load_pipeline_config(paths, entry)
    assert result.ok is False
    assert result.raw_text == "listen_port = = broken"
    assert result.quarantined_path is not None
    assert result.quarantined_path.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.profile'`.

- [ ] **Step 3: Implement `gui/profile.py`**

```python
import toml
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from vspeech.config import Config
from vspeech.logger import logger

from gui.migration import CONFIG_MIGRATIONS
from gui.migration import CURRENT_CONFIG_VERSION
from gui.migration import CURRENT_PROFILE_VERSION
from gui.migration import LoadResult
from gui.migration import PROFILE_MIGRATIONS
from gui.migration import migrate_dict
from gui.migration import quarantine
from gui.paths import ProfilePaths


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
        logger.warning("pipelines.toml 破損: %s に退避し空プロファイルへ (%s)", backup, e)
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_profile.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/profile.py tests/gui/test_profile.py
git commit -m "feat(gui): profile/manifest models with safe load and migration"
```

---

### Task 6: Pipeline process runner + gRPC command

**Files:**
- Create: `gui/process.py`
- Test: `tests/gui/test_process.py`

**Interfaces:**
- Consumes: `vspeech.config.RoutesList`, `vspeech.shared_context.EventAddress`, vstreamer-protos `Command`/`Operand`/`OperationChain`/`CommanderStub`.
- Produces:
  - `build_argv(config_path: Path) -> list[str]`.
  - `build_text_command(text: str, text_send_operations: RoutesList) -> Command`.
  - `class PipelineRunner` with `__init__(self, config_path: Path, port: int, on_log: Callable[[str], None], on_exit: Callable[[int], None])`, `start()`, `stop()`, `is_running() -> bool`, `send_text(text: str, text_send_operations: RoutesList)`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/test_process.py`:

```python
import sys
from pathlib import Path

from gui.process import build_argv
from gui.process import build_text_command


def test_build_argv():
    argv = build_argv(Path("/x/pipe.toml"))
    assert argv[:3] == [sys.executable, "-m", "vspeech"]
    assert argv[-2:] == ["--config", str(Path("/x/pipe.toml"))]


def test_build_text_command_trims_and_chains():
    command = build_text_command("  hi  ", [["tts", "playback"]])
    assert command.operand.text == "hi"
    assert len(command.chains) == 1
    assert len(command.chains[0].operations) == 2


def test_build_text_command_drops_empty_chains():
    command = build_text_command("x", [["tts"], []])
    assert len(command.chains) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/gui/test_process.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.process'`.

- [ ] **Step 3: Implement `gui/process.py`**

```python
import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import Popen
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --all-extras pytest tests/gui/test_process.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/process.py tests/gui/test_process.py
git commit -m "feat(gui): PipelineRunner subprocess lifecycle and text-send command"
```

---

### Task 7: Reusable Tk widgets

**Files:**
- Create: `gui/widgets.py`

**Interfaces:**
- Produces (all value widgets expose `get_value() -> Any` and `set(value: Any)`):
  - `class Checkbutton(ttkCheckbutton)` (BooleanVar-backed).
  - `class Textbox(Entry)` (StringVar-backed).
  - `class Spinbox(ttkSpinbox)` (`get_value()` returns the string; caller coerces).
  - `class ScrolledText(ttkScrolledText)`.
  - `class TextHandler(logging.Handler)` — appends `LogRecord`s to a `Text`/`ScrolledText` via `after(0, ...)`.

This task has no unit test (Tk widgets need a display). It is verified by later UI tasks that import and use it. Provide the complete module.

- [ ] **Step 1: Implement `gui/widgets.py`**

```python
from logging import Handler
from logging import LogRecord
from tkinter import END
from tkinter import BooleanVar
from tkinter import StringVar
from typing import Any

from ttkbootstrap import Checkbutton as ttkCheckbutton
from ttkbootstrap import Entry
from ttkbootstrap import Spinbox as ttkSpinbox
from ttkbootstrap import Text
from ttkbootstrap.widgets.scrolled import ScrolledText as ttkScrolledText


class ScrolledText(ttkScrolledText):
    pass


class TextHandler(Handler):
    """Log to a Tkinter Text/ScrolledText widget from any thread."""

    def __init__(self, text: "Text | ScrolledText"):
        Handler.__init__(self)
        self.text = text

    def emit(self, record: LogRecord):
        msg = self.format(record)

        def append():
            self.text.configure(state="normal")
            self.text.insert(END, msg + "\n")
            self.text.configure(state="disabled")
            self.text.yview(END)

        self.text.after(0, append)


class Spinbox(ttkSpinbox):
    def get_value(self) -> str:
        return super().get()

    def set(self, value: Any):
        self.delete(0, END)
        self.insert(0, str(value))


class Checkbutton(ttkCheckbutton):
    var: BooleanVar

    def __init__(self, master: Any, **kw: Any):
        self.var = BooleanVar()
        super().__init__(master, variable=self.var, onvalue=True, offvalue=False, **kw)

    def get_value(self) -> bool:
        return self.var.get()

    def set(self, value: Any):
        self.var.set(bool(value))


class Textbox(Entry):
    var: StringVar

    def __init__(self, master: Any, **kw: Any):
        self.var = StringVar()
        super().__init__(master, textvariable=self.var, **kw)

    def get_value(self) -> str:
        return self.var.get()

    def set(self, value: Any):
        self.var.set("" if value is None else str(value))
```

- [ ] **Step 2: Verify it imports**

Run: `uv run --all-extras python -c "import gui.widgets"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add gui/widgets.py
git commit -m "feat(gui): reusable value widgets and log handler"
```

---

### Task 8: Minimal essential-fields form

**Files:**
- Create: `gui/form.py`

**Interfaces:**
- Consumes: `gui/widgets.py`, `gui/autocomplete_combobox.py`, `vspeech.config` (`Config`, `TranscriptionWorkerType`, `TtsWorkerType`).
- Produces:
  - `class PipelineForm(Frame)` with `__init__(self, master, on_change: Callable[[], None])`, `bind_config(config: Config) -> None` (rebuild widgets from a config), `read_into(config: Config) -> None` (write widget values back).

Config-path get/set uses dotted paths (e.g. `recording.input_device_index`). Backend fields for transcription/tts depend on the selected `worker_type`; when that combo changes, the conditional subframe rebuilds. Device combos degrade to empty lists (manual index entry) if the `audio` extra is missing.

This is a build task (no unit test). Provide the complete module.

- [ ] **Step 1: Implement `gui/form.py`**

```python
from functools import partial
from tkinter import EW
from tkinter import W
from tkinter import X
from typing import Any
from typing import Callable

from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Labelframe

from gui.autocomplete_combobox import AutocompleteCombobox
from gui.widgets import Checkbutton
from gui.widgets import Spinbox
from gui.widgets import Textbox
from vspeech.config import Config
from vspeech.config import TranscriptionWorkerType
from vspeech.config import TtsWorkerType

try:
    from vspeech.lib.audio import list_all_devices
except Exception:  # audio extra not installed

    def list_all_devices(input: bool = False, output: bool = False) -> dict[str, int]:
        return {}


def _get(config: Config, path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _set(config: Config, path: str, value: Any) -> None:
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, child, value)


class PipelineForm(Frame):
    def __init__(self, master: Any, on_change: Callable[[], None]):
        super().__init__(master)
        self.on_change = on_change
        self.config: Config | None = None
        # widget -> (config_path, coerce fn from widget value to config value)
        self.bindings: dict[Any, tuple[str, Callable[[Any], Any]]] = {}
        self.body = Frame(self)
        self.body.pack(fill=X)

    def bind_config(self, config: Config) -> None:
        self.config = config
        for child in list(self.body.children.values()):
            child.destroy()
        self.bindings.clear()
        self._section_recording()
        self._section_playback()
        self._section_transcription()
        self._section_tts()
        self._section_vc()

    def read_into(self, config: Config) -> None:
        for widget, (path, coerce) in self.bindings.items():
            try:
                _set(config, path, coerce(widget.get_value()))
            except (ValueError, KeyError):
                continue

    # --- field builders -------------------------------------------------

    def _check(self, parent: Any, path: str, label: str) -> Checkbutton:
        widget = Checkbutton(parent, text=label)
        widget.set(_get(self.config, path))
        widget.configure(command=self.on_change)
        self.bindings[widget] = (path, bool)
        return widget

    def _entry(self, parent: Any, path: str, label: str) -> Frame:
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        widget = Textbox(frame)
        widget.set(_get(self.config, path))
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, str)
        widget.pack(fill=X)
        return frame

    def _spin(
        self, parent: Any, path: str, label: str, from_: float, to: float, inc: float
    ) -> Frame:
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        widget = Spinbox(frame, from_=from_, to=to, increment=inc, wrap=True)
        widget.set(_get(self.config, path))
        coerce = int if float(inc).is_integer() else float
        widget.configure(command=self.on_change)
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, coerce)
        widget.pack(fill=X)
        return frame

    def _device_combo(
        self, parent: Any, path: str, label: str, *, input: bool
    ) -> Frame:
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        combo = AutocompleteCombobox[int](frame)
        combo.set_completion_list(
            list_all_devices(input=input, output=not input)
        )
        current = _get(self.config, path)
        combo_label = combo.get_label_for_item_value(current) if current is not None else None
        if combo_label:
            combo.set(combo_label)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.on_change())
        self.bindings[combo] = (path, lambda v: v)
        combo.pack(fill=X)
        return frame

    def _enum_combo(
        self, parent: Any, path: str, label: str, enum_cls: Any
    ) -> AutocompleteCombobox[Any]:
        Label(parent, text=label).pack(fill=X)
        combo = AutocompleteCombobox[Any](parent)
        combo.set_completion_list({member.name: member for member in enum_cls})
        current = _get(self.config, path)
        combo.set(current.name)
        self.bindings[combo] = (path, lambda v: v)
        combo.pack(fill=X)
        return combo

    # --- worker sections ------------------------------------------------

    def _section_recording(self) -> None:
        box = Labelframe(self.body, text="recording")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "recording.enable", "enable recording").pack(anchor=W)
        self._device_combo(box, "recording.input_device_index", "input device", input=True).pack(fill=X)
        self._spin(box, "recording.rate", "rate", 8000, 48000, 1).pack(fill=X)
        self._spin(box, "recording.silence_threshold", "silence threshold (dBFS)", -120, 0, 1).pack(fill=X)

    def _section_playback(self) -> None:
        box = Labelframe(self.body, text="playback")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "playback.enable", "enable playback").pack(anchor=W)
        self._device_combo(box, "playback.output_device_index", "output device", input=False).pack(fill=X)
        self._spin(box, "playback.volume", "volume", 0, 100, 1).pack(fill=X)

    def _section_transcription(self) -> None:
        box = Labelframe(self.body, text="transcription")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "transcription.enable", "enable transcription").pack(anchor=W)
        combo = self._enum_combo(box, "transcription.worker_type", "worker_type", TranscriptionWorkerType)
        backend = Frame(box)
        backend.pack(fill=X)
        combo.bind(
            "<<ComboboxSelected>>",
            partial(self._rebuild_transcription_backend, backend, combo),
            add="+",
        )
        self._rebuild_transcription_backend(backend, combo, None)

    def _rebuild_transcription_backend(self, backend: Frame, combo: Any, _event: Any) -> None:
        for child in list(backend.children.values()):
            self.bindings.pop(child, None)
            child.destroy()
        self.on_change()
        try:
            worker_type = combo.get_value()
        except KeyError:
            return
        if worker_type == TranscriptionWorkerType.WHISPER:
            self._entry(backend, "whisper.model", "whisper model").pack(fill=X)
            self._spin(backend, "whisper.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
        elif worker_type == TranscriptionWorkerType.GCP:
            self._entry(backend, "gcp.service_account_file_path", "gcp key.json path").pack(fill=X)
        elif worker_type == TranscriptionWorkerType.ACP:
            self._entry(backend, "ami.appkey", "ami appkey").pack(fill=X)
            self._entry(backend, "ami.engine_uri", "ami engine_uri").pack(fill=X)
            self._entry(backend, "ami.engine_name", "ami engine_name").pack(fill=X)
            self._entry(backend, "ami.service_id", "ami service_id").pack(fill=X)

    def _section_tts(self) -> None:
        box = Labelframe(self.body, text="tts")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "tts.enable", "enable tts").pack(anchor=W)
        combo = self._enum_combo(box, "tts.worker_type", "worker_type", TtsWorkerType)
        backend = Frame(box)
        backend.pack(fill=X)
        combo.bind(
            "<<ComboboxSelected>>",
            partial(self._rebuild_tts_backend, backend, combo),
            add="+",
        )
        self._rebuild_tts_backend(backend, combo, None)

    def _rebuild_tts_backend(self, backend: Frame, combo: Any, _event: Any) -> None:
        for child in list(backend.children.values()):
            self.bindings.pop(child, None)
            child.destroy()
        self.on_change()
        try:
            worker_type = combo.get_value()
        except KeyError:
            return
        if worker_type == TtsWorkerType.VOICEVOX:
            self._entry(backend, "voicevox.openjtalk_dir", "openjtalk_dir").pack(fill=X)
            self._entry(backend, "voicevox.model_dir", "model_dir").pack(fill=X)
            self._entry(backend, "voicevox.onnxruntime_path", "onnxruntime_path").pack(fill=X)
            self._spin(backend, "voicevox.speaker_id", "speaker_id", 0, 100, 1).pack(fill=X)
        elif worker_type == TtsWorkerType.VR2:
            self._entry(backend, "vr2.voice_name", "voice_name").pack(fill=X)

    def _section_vc(self) -> None:
        box = Labelframe(self.body, text="vc")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "vc.enable", "enable vc").pack(anchor=W)
        self._entry(box, "rvc.model_file", "rvc model_file").pack(fill=X)
        self._entry(box, "rvc.hubert_model_file", "hubert asset dir").pack(fill=X)
        self._entry(box, "rvc.rmvpe_model_file", "rmvpe model_file").pack(fill=X)
        self._spin(box, "rvc.f0_up_key", "f0_up_key", -64, 64, 1).pack(fill=X)
        self._spin(box, "rvc.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
```

Note on `_entry` for path/`SecretStr` fields: `read_into` sets the raw string; pydantic coerces `Path` fields from strings, and `SecretStr` fields (`ami.appkey`) accept a plain string on `model_validate`/attribute set. Empty string for `whisper.gpu_id`/`onnxruntime_path` etc. is handled because those numeric fields use `_spin` (int coerce) and path fields accept `""` → falls to the config default only if left blank; leave blank fields as-is (they set the literal value).

- [ ] **Step 2: Verify it imports**

Run: `uv run --all-extras python -c "import gui.form"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add gui/form.py
git commit -m "feat(gui): minimal essential-fields pipeline form"
```

---

### Task 9: Raw TOML editor

**Files:**
- Create: `gui/rawedit.py`

**Interfaces:**
- Consumes: `gui/widgets.py` (`ScrolledText`), migration (`migrate_dict`, `CONFIG_MIGRATIONS`, `CURRENT_CONFIG_VERSION`), `vspeech.config.Config`.
- Produces:
  - `class RawTomlEditor(Frame)` with `__init__(self, master)`, `set_text(text: str) -> None`, `set_config(config: Config) -> None`, `parse() -> tuple[Config | None, str | None]` (returns `(config, None)` on success or `(None, error_message)` on failure).

Build task. Provide the complete module.

- [ ] **Step 1: Implement `gui/rawedit.py`**

```python
from tkinter import END
from typing import Any

import toml
from pydantic import ValidationError
from ttkbootstrap import Frame

from gui.migration import CONFIG_MIGRATIONS
from gui.migration import CURRENT_CONFIG_VERSION
from gui.migration import migrate_dict
from gui.widgets import ScrolledText
from vspeech.config import Config


class RawTomlEditor(Frame):
    def __init__(self, master: Any):
        super().__init__(master)
        self.text = ScrolledText(self)
        self.text.pack(fill="both", expand=True)

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", END)
        self.text.insert("1.0", text)

    def set_config(self, config: Config) -> None:
        self.set_text(config.export_to_toml())

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def parse(self) -> tuple[Config | None, str | None]:
        try:
            data, _ = migrate_dict(
                toml.loads(self.get_text()),
                from_version=0,
                migrations=CONFIG_MIGRATIONS,
                current=CURRENT_CONFIG_VERSION,
            )
            return Config.model_validate(data), None
        except (toml.TomlDecodeError, ValidationError) as e:
            return None, str(e)
```

- [ ] **Step 2: Verify it imports**

Run: `uv run --all-extras python -c "import gui.rawedit"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add gui/rawedit.py
git commit -m "feat(gui): raw TOML editor with validating parse"
```

---

### Task 10: Pipeline editor panel

**Files:**
- Create: `gui/pipeline_editor.py`

**Interfaces:**
- Consumes: `PipelineForm` (Task 8), `RawTomlEditor` (Task 9), `PipelineRunner` (Task 6), `ScrolledText`/`Textbox`/`TextHandler` (Task 7), profile save/load (Task 5), ports (Task 2), `Config`.
- Produces:
  - `class PipelineEditor(Frame)` with `__init__(self, master, paths, on_dirty)`, `load_entry(entry: PipelineEntry) -> None` (safe-loads and shows the pipeline; on corrupt shows the raw broken text + error banner and disables start), `save() -> bool`, `sync_form_to_config() -> None`.

Build task. Provide the complete module. This panel owns a `PipelineRunner` per loaded entry and drives Start/Stop/text-send/log.

- [ ] **Step 1: Implement `gui/pipeline_editor.py`**

```python
from tkinter import BOTH
from tkinter import DISABLED
from tkinter import END
from tkinter import LEFT
from tkinter import NORMAL
from tkinter import X
from typing import Any
from typing import Callable

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Notebook

from gui.form import PipelineForm
from gui.paths import ProfilePaths
from gui.process import PipelineRunner
from gui.profile import PipelineEntry
from gui.profile import load_pipeline_config
from gui.profile import save_pipeline_config
from gui.ports import is_port_free
from gui.rawedit import RawTomlEditor
from gui.widgets import ScrolledText
from gui.widgets import Textbox
from vspeech.config import Config
from vspeech.logger import logger


class PipelineEditor(Frame):
    def __init__(
        self, master: Any, paths: ProfilePaths, on_dirty: Callable[[], None]
    ):
        super().__init__(master)
        self.paths = paths
        self.on_dirty = on_dirty
        self.entry: PipelineEntry | None = None
        self.config: Config | None = None
        self.runner: PipelineRunner | None = None
        self.broken = False

        self.banner = Label(self, text="", bootstyle="danger")
        self.banner.pack(fill=X)

        notebook = Notebook(self)
        notebook.pack(fill=BOTH, expand=True)
        self.form = PipelineForm(notebook, on_change=self.on_dirty)
        self.raw = RawTomlEditor(notebook)
        notebook.add(self.form, text="Form")
        notebook.add(self.raw, text="Raw TOML")
        self.notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        controls = Frame(self)
        controls.pack(fill=X)
        self.start_bt = Button(controls, text="Start", command=self.start)
        self.start_bt.pack(side=LEFT, padx=4, pady=4)
        self.stop_bt = Button(controls, text="Stop", command=self.stop, state=DISABLED)
        self.stop_bt.pack(side=LEFT, padx=4, pady=4)
        self.status = Label(controls, text="■ stopped")
        self.status.pack(side=LEFT, padx=8)
        self.apply_raw_bt = Button(controls, text="Apply Raw", command=self.apply_raw)
        self.apply_raw_bt.pack(side=LEFT, padx=4)
        self.save_bt = Button(controls, text="Save", command=self.save)
        self.save_bt.pack(side=LEFT, padx=4)

        send_frame = Frame(self)
        send_frame.pack(fill=X)
        self.send_entry = Textbox(send_frame)
        self.send_entry.pack(side=LEFT, fill=X, expand=True, padx=4)
        Button(send_frame, text="send", command=self.send_text).pack(side=LEFT, padx=4)

        self.log = ScrolledText(self, height=10, state=DISABLED)
        self.log.pack(fill=BOTH, expand=True)

    # --- loading --------------------------------------------------------

    def load_entry(self, entry: PipelineEntry) -> None:
        self.entry = entry
        result = load_pipeline_config(self.paths, entry)
        if result.ok and result.value is not None:
            self.broken = False
            self.banner.configure(text="")
            self.config = result.value
            self.form.bind_config(self.config)
            self.raw.set_config(self.config)
            self.start_bt.configure(state=NORMAL)
            if result.migrated:
                save_pipeline_config(self.paths, entry, self.config)
                self.on_dirty()
        else:
            self.broken = True
            self.config = None
            self.banner.configure(
                text=f"❗ config 読込失敗: {result.quarantined_path} に退避。生TOMLで修正してください — {result.error}"
            )
            self.raw.set_text(result.raw_text or "")
            self.notebook.select(self.raw)
            self.start_bt.configure(state=DISABLED)

    # --- form/raw sync --------------------------------------------------

    def _on_tab_changed(self, _event: Any) -> None:
        if self.notebook.select() == str(self.raw) and self.config is not None:
            self.sync_form_to_config()
            self.raw.set_config(self.config)

    def sync_form_to_config(self) -> None:
        if self.config is not None:
            self.form.read_into(self.config)

    def apply_raw(self) -> None:
        config, error = self.raw.parse()
        if config is None:
            self.banner.configure(text=f"❗ TOML エラー: {error}")
            return
        self.broken = False
        self.banner.configure(text="")
        self.config = config
        self.form.bind_config(config)
        self.start_bt.configure(state=NORMAL)
        self.on_dirty()

    def save(self) -> bool:
        if self.entry is None:
            return False
        if self.broken or self.config is None:
            config, error = self.raw.parse()
            if config is None:
                self.banner.configure(text=f"❗ 保存不可 (TOML エラー): {error}")
                return False
            self.config = config
            self.broken = False
            self.banner.configure(text="")
        else:
            self.sync_form_to_config()
        save_pipeline_config(self.paths, self.entry, self.config)
        logger.info("saved pipeline %s", self.entry.id)
        return True

    # --- runtime --------------------------------------------------------

    def start(self) -> None:
        if self.entry is None or not self.save():
            return
        if not is_port_free(self.entry.port):
            self._append_log(f"port {self.entry.port} is busy; cannot start")
            return
        self.runner = PipelineRunner(
            config_path=self.paths.pipeline_config(self.entry.id),
            port=self.entry.port,
            on_log=lambda line: self.log.after(0, self._append_log, line),
            on_exit=lambda code: self.log.after(0, self._on_exit, code),
        )
        self.runner.start()
        self.start_bt.configure(state=DISABLED)
        self.stop_bt.configure(state=NORMAL)
        self.status.configure(text="● running")

    def stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def _on_exit(self, code: int) -> None:
        self._append_log(f"process exited: {code}")
        self.start_bt.configure(state=NORMAL)
        self.stop_bt.configure(state=DISABLED)
        self.status.configure(text="■ stopped")

    def send_text(self) -> None:
        if self.runner and self.runner.is_running() and self.config is not None:
            self.runner.send_text(
                self.send_entry.get_value(), self.config.text_send_operations
            )

    def _append_log(self, line: str) -> None:
        self.log.configure(state=NORMAL)
        self.log.insert(END, line + "\n")
        self.log.configure(state=DISABLED)
        self.log.yview(END)
```

- [ ] **Step 2: Verify it imports**

Run: `uv run --all-extras python -c "import gui.pipeline_editor"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add gui/pipeline_editor.py
git commit -m "feat(gui): pipeline editor panel with form/raw, run, log, send"
```

---

### Task 11: Main window + entry point

**Files:**
- Create: `gui/app.py`
- Modify: `gui/__main__.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `class App(Frame)` — pipeline list (left) + `PipelineEditor` (right), new/delete buttons.
  - `main()` — click command with `--profile-dir` and `--theme` options; no config-file argument.

Build task + manual smoke. Provide complete modules.

- [ ] **Step 1: Implement `gui/app.py`**

```python
from pathlib import Path
from tkinter import BOTH
from tkinter import LEFT
from tkinter import RIGHT
from tkinter import END
from tkinter import Listbox
from tkinter import Y
from typing import Any
from uuid import uuid4

import click
from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Window
from ttkbootstrap.dialogs import Querybox
from ttkbootstrap.themes.standard import STANDARD_THEMES

from gui.paths import resolve_paths
from gui.pipeline_editor import PipelineEditor
from gui.ports import allocate_free_port
from gui.profile import PipelineEntry
from gui.profile import load_default_config
from gui.profile import load_profile
from gui.profile import save_pipeline_config
from gui.profile import save_profile
from gui.recipes import RECIPES
from gui.recipes import RECIPES_BY_KEY
from gui.migration import quarantine
from vspeech.logger import logger


class App(Frame):
    def __init__(self, master: Any, profile_dir: Path | None):
        super().__init__(master)
        self.pack(fill=BOTH, expand=True)
        self.paths = resolve_paths(profile_dir)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.default_config = load_default_config(self.paths)
        self.profile = load_profile(self.paths)

        left = Frame(self)
        left.pack(side=LEFT, fill=Y)
        self.listbox = Listbox(left, width=32)
        self.listbox.pack(fill=Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        Button(left, text="+ new", command=self.new_pipeline).pack(fill="x")
        Button(left, text="del", command=self.delete_pipeline).pack(fill="x")

        self.editor = PipelineEditor(self, self.paths, on_dirty=lambda: None)
        self.editor.pack(side=RIGHT, fill=BOTH, expand=True)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, END)
        for entry in self.profile.pipelines:
            self.listbox.insert(END, f"{entry.name}  :{entry.port}")

    def _on_select(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.profile.pipelines[selection[0]]
        self.editor.load_entry(entry)

    def new_pipeline(self) -> None:
        labels = [recipe.label for recipe in RECIPES]
        chosen = Querybox.get_string(
            prompt="レシピを選んでください:\n" + "\n".join(labels),
            title="new pipeline",
        )
        recipe = next((r for r in RECIPES if r.label == chosen or r.key == chosen), RECIPES_BY_KEY["blank"])
        claimed = {entry.port for entry in self.profile.pipelines}
        port = allocate_free_port(claimed)
        entry = PipelineEntry(
            id=uuid4().hex[:8],
            name=recipe.label,
            port=port,
            recipe=recipe.key,
        )
        config = recipe.apply(self.default_config)
        save_pipeline_config(self.paths, entry, config)
        self.profile.pipelines.append(entry)
        save_profile(self.paths, self.profile)
        self._refresh_list()

    def delete_pipeline(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.profile.pipelines.pop(selection[0])
        config_path = self.paths.pipeline_config(entry.id)
        if config_path.exists():
            quarantine(config_path)
            config_path.unlink()
        save_profile(self.paths, self.profile)
        self._refresh_list()
        logger.info("deleted pipeline %s", entry.id)


@click.command()
@click.option("--profile-dir", "profile_dir", type=click.Path(path_type=Path), default=None)
@click.option("-t", "--theme", default="cosmo", type=click.Choice(list(STANDARD_THEMES.keys())))
def main(profile_dir: Path | None, theme: str):
    root = Window(themename=theme)
    root.title("vspeech pipelines")
    root.geometry("900x760")
    App(root, profile_dir)
    root.mainloop()
```

- [ ] **Step 2: Update `gui/__main__.py`**

```python
from gui.app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual smoke — launch and create a loopback pipeline**

Run: `uv run --all-extras python -m gui --profile-dir ./.gui-smoke`
Then in the GUI:
1. Click `+ new`, choose "マイク→再生 (モニター)".
2. Select the pipeline; on the Form tab pick an input + output device.
3. Click Start; speak into the mic and confirm you hear yourself (or see log lines). Click Stop.

Expected: `./.gui-smoke/` contains `default.toml`, `pipelines.toml`, `pipelines/<id>.toml`; process starts and logs stream into the log pane.

- [ ] **Step 4: Commit**

```bash
git add gui/app.py gui/__main__.py
git commit -m "feat(gui): main window with pipeline list, recipes, and free-port assignment"
```

---

### Task 12: Remove old GUI, docs, and health gate

**Files:**
- Delete: `gui/gui.py`, `gui/dummy_param.py`
- Modify: `CLAUDE.md` (Run the GUI command)
- Modify: `.gitignore` (ignore the smoke dir)

- [ ] **Step 1: Delete the old modules**

```bash
git rm gui/gui.py gui/dummy_param.py
```

- [ ] **Step 2: Update the CLAUDE.md GUI command**

In `CLAUDE.md`, replace the GUI run line:

```sh
# Run the GUI control panel (reads the user profile from the OS config dir)
uv run python -m gui                    # optionally: --profile-dir PATH  -t THEME
```

- [ ] **Step 3: Ignore the smoke directory**

Add to `.gitignore`:

```
.gui-smoke/
```

- [ ] **Step 4: Run the full health gate**

Run: `uv run --all-extras poe check`
Expected: green except the two pre-accepted findings noted in `docs/follow-ups.md` (torch CVE, vr2_config deadcode). Fix any new ruff/ty failures introduced by the GUI (e.g. import ordering, missing annotations). Re-run until only the accepted findings remain.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(gui): remove the single-config GUI and update docs"
```

---

## Self-Review

**Spec coverage:**
- Requirement 1 (always read the user default profile) → Task 1 (`paths`), Task 5 (`load_default_config`/`load_profile`), Task 11 (`App` loads on startup, no config-file arg).
- Requirement 2 (multiple pipelines start/stop) → Task 6 (`PipelineRunner`), Task 10 (Start/Stop), Task 11 (list of pipelines).
- Requirement 3 (manage each pipeline's settings) → Task 8 (form), Task 9 (raw editor), Task 10 (editor).
- Requirement 4 (config migration) → Task 3 (chain/quarantine), Task 5 (safe load + fallback), Task 10 (corrupt → raw repair).
- Requirement 5 (works out of the box) → Task 4 (recipes incl. `mic_loopback`), Task 11 (recipe on new).
- Auto free-port allocation → Task 2, used in Task 11, re-checked in Task 10.
- Deletion of old GUI + docs → Task 12.

**Placeholder scan:** No TBD/TODO; every code step contains complete code. UI tasks (7–11) are build tasks with import-smoke or manual verification, which is the deliberate testing boundary from the spec (Tk-independent tests only).

**Type consistency:** `ProfilePaths.pipeline_config(pipeline_id)`, `PipelineEntry(id/name/port/recipe/config_version)`, `load_pipeline_config -> LoadResult[Config]`, `PipelineRunner(config_path, port, on_log, on_exit)`, `Recipe.apply(Config) -> Config`, `RawTomlEditor.parse() -> tuple[Config | None, str | None]`, `PipelineForm.bind_config/read_into` are used with the same names/signatures across tasks.

**Known implementation risks to watch during execution (not plan gaps):**
- `ttkbootstrap.dialogs.Querybox.get_string` is a simple prompt; if recipe selection UX is poor, replace with a small custom `Toplevel` + radio list. Behavior (return the chosen recipe) is unchanged.
- `Labelframe`/`Notebook` tab identity comparison (`self.notebook.select() == str(self.raw)`) may need adjusting to `.index()` on some ttkbootstrap versions; verify during Task 10 smoke.
- `Config` numeric fields left blank in the form: `_spin` coerces to int/float and will raise `ValueError` on empty, which `read_into` swallows (keeps prior value) — acceptable.
