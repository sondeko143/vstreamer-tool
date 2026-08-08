# config をファイル1本に統一する 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** config の入力経路を `--config` ファイル 1 本に統一し、pydantic-settings 依存と死んだコンテナ配備経路を撤去する。

**Architecture:** `Config` を `BaseSettings` から `BaseModel` に載せ替え、`BaseSettings` 由来の暗黙の `extra="forbid"` は `ConfigDict` で明示して引き継ぐ。`--config` を click の required オプションにし、環境変数フォールバックを削除する。pydantic-settings が二度と戻らないよう、既存の `tests/test_forbidden_imports.py` の構造ゲートに載せる。コンテナ配備は成果物とそれ専用の uv 依存解決設定をまとめて撤去する。

**Tech Stack:** Python 3.14 / uv / pydantic v2 / click / pytest (`asyncio_mode = "auto"`) / ruff / ty / poethepoet

**ADR:** [ADR-0066](../../adr/0066-config-input-file-only.md)（config 入力の一本化）、[ADR-0067](../../adr/0067-drop-container-deploy-path.md)（コンテナ配備の撤去）。どちらも `Proposed`。Task 6 で `Accepted` に昇格させる。

**Spec:** [2026-08-09-config-file-only-design.md](../specs/2026-08-09-config-file-only-design.md)

## Global Constraints

- Python は **3.14 のみ**（`requires-python = ">=3.14,<3.15"`）。floor を下げない。
- **コメントと docstring は英語**、ユーザーが読む文字列（ログ・例外メッセージ・click の `help=` とコマンド docstring・`config.py` の `description=`）は日本語のまま（ADR-0064）。日本語コメントが残っているファイルを触ったら、そのファイルのコメントは同じ変更で英語化する。
- **import は 1 行 1 つ**（ruff `force-single-line = true`）。
- pydantic は **v2 API のみ**。`parse_obj` / `.dict()` / `.json()` / `root_validator` / `orm_mode` / `Field(env=)` / `json_encoders` を復活させない。
- 依存を同期するときは **`uv sync --all-extras`**。単独の `--extra` は他の extra を deselect する。
- スコープ外と判断した指摘は、**その指摘が関係するコードの場所にコメントで残す**（別途の追跡ドキュメントを作らない）。
- 検証コマンドの終了コードはパイプ越しに読まない。`uv run pytest` は完全な node ID で指定する。

## File Structure

| ファイル | 責務 | 操作 |
|---|---|---|
| `vspeech/config.py` | config スキーマ。`Config` の基底クラス・`model_config`・`listen_port`・import を変更 | Modify |
| `vspeech/main.py` | エントリポイント。`--config` を必須化し env フォールバックを削除 | Modify |
| `tests/test_config_input.py` | config の入力契約（env 非読込 / 未知キー / ファイル解決）の回帰テスト | Create |
| `tests/test_main.py` | エントリポイントの回帰テスト。既存テストの入力修正 + 必須化テスト追加 | Modify |
| `tests/test_forbidden_imports.py` | 構造ゲート。`pydantic_settings` を追加 | Modify |
| `pyproject.toml` | 依存から pydantic-settings を削除、解決対象を win32 のみに | Modify |
| `poe_tasks.toml` | `requirements-pod` / `clean` タスクを削除 | Modify |
| `.vscode/launch.json` | Cloud Run 構成と、削除済みモジュールを指す `gui` 構成を削除 | Modify |
| `Dockerfile` / `.dockerignore` / `requirements-pod.txt` | コンテナ配備の成果物 | Delete |
| `CLAUDE.md` | config 入力・Docker・per-platform ピンの記述を実態に合わせる | Modify |
| `docs/adr/0066-*.md` / `0067-*.md` / `README.md` | Status を Accepted に昇格 | Modify |

---

### Task 1: `Config` を `BaseModel` 化し、環境変数の読み込みを止める

**Files:**
- Create: `tests/test_config_input.py`
- Modify: `vspeech/config.py:11`, `vspeech/config.py:18-19`, `vspeech/config.py:601`, `vspeech/config.py:619-621`, `vspeech/config.py:626-629`

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces: `Config` は `pydantic.BaseModel` のサブクラスになる。`Config.model_config` は `ConfigDict(extra="forbid")`。`Config.listen_port: int` の既定は `8080` で alias なし。`Config.read_config_from_file(file: IO[bytes]) -> Config` は変更なし。

- [ ] **Step 1: Write the failing tests**

`tests/test_config_input.py` を新規作成する。

```python
"""Regression tests for how configuration gets into the pipeline (ADR-0066).

The pipeline reads its configuration from a `--config` file and from nothing
else. These pin the parts of that contract that are cheap to regress by
accident: the environment must not leak in, an unknown top-level key must still
be rejected, and a file must resolve to the values it names.
"""

import pytest
from pydantic import ValidationError

from vspeech.config import Config


def test_environment_variables_do_not_reach_the_config(monkeypatch):
    monkeypatch.setenv("vspeech_listen_port", "9999")
    monkeypatch.setenv("vspeech_transcription__enable", "true")

    config = Config()

    assert config.listen_port == 8080
    assert config.transcription.enable is False


def test_a_top_level_unknown_key_is_rejected():
    # Before ADR-0066 this came for free from SettingsConfigDict; it is now
    # stated outright on the model. A typo in config.toml must not be swallowed.
    with pytest.raises(ValidationError):
        Config.model_validate({"listen_prot": 9999})


def test_the_port_key_no_longer_sets_the_listen_port():
    # `PORT` was an alias serving the Cloud Run contract only (ADR-0067).
    with pytest.raises(ValidationError):
        Config.model_validate({"PORT": 9999})


def test_a_config_file_resolves_to_the_values_it_names(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "listen_port = 9100\n\n[recording]\nenable = true\n", encoding="utf-8"
    )

    with config_file.open("rb") as opened:
        config = Config.read_config_from_file(opened)

    assert config.listen_port == 9100
    assert config.recording.enable is True
```

- [ ] **Step 2: Run the tests and confirm which fail**

Run: `uv run pytest tests/test_config_input.py -v`

Expected: 4 件中 **2 件が FAIL**。

- `test_environment_variables_do_not_reach_the_config` → FAIL（`assert 9999 == 8080`。現状は env を読む）
- `test_the_port_key_no_longer_sets_the_listen_port` → FAIL（`DID NOT RAISE ValidationError`。現状 `PORT` は alias で通る）
- 残り 2 件は PASS。これらは変更で壊してはいけない挙動を固定するガードなので、赤くならないのが正しい。

- [ ] **Step 3: Move `Config` off `BaseSettings`**

`vspeech/config.py` の import を修正する。11 行目の `AliasChoices` は `listen_port` が唯一の利用箇所なので一緒に削る。18-19 行目を削除する。

```python
# 削除する 3 行
from pydantic import AliasChoices
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
```

`BaseModel` と `ConfigDict` は既に import 済み（12-13 行目）なので追加は不要。

601 行目の基底クラスを差し替える。

```python
class Config(BaseModel):
```

619-621 行目の `listen_port` から alias を外す。

```python
    listen_port: int = 8080
```

626-629 行目の `model_config` を差し替える。ネストされたセクション内の未知キーが今も素通りする件は、スコープ外と判断した根拠ごとその場に残す。

```python
    # `extra="forbid"` was inherited implicitly from SettingsConfigDict before
    # ADR-0066. Restate it here or a typo'd key in config.toml starts being
    # swallowed: plain BaseModel defaults to extra="ignore".
    #
    # [Open, deferred 2026-08-09] This only guards the top level. The nested
    # section models (RecordingConfig and friends) are plain BaseModel, so
    # `[recording] enabel = true` is still ignored silently. Tightening them
    # was left out of ADR-0066 because any stray key in an existing config.toml
    # would then stop the pipeline from starting.
    model_config = ConfigDict(extra="forbid")
```

- [ ] **Step 4: Run the tests and verify they all pass**

Run: `uv run pytest tests/test_config_input.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the wider config tests to catch collateral damage**

Run: `uv run pytest tests/test_config_secret.py tests/test_config_bounds.py tests/test_config_stream_vc.py tests/test_telemetry_config.py tests/test_voicevox_config.py -v`
Expected: all passed

`export_to_toml` は `model_dump()` 経由なので基底クラスの変更に影響されないはずだが、`tests/test_config_secret.py` がそれを守る唯一のテストなので必ず通す。

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff format . ; uv run ruff check . ; uv run ty check`
Expected: 3 つとも clean。終了コードはコマンドごとに確認する（パイプ越しの `$?` を見ない）。

- [ ] **Step 7: Commit**

```bash
git add vspeech/config.py tests/test_config_input.py
git commit -m "refactor(config)!: read configuration from the file only (ADR-0066)"
```

---

### Task 2: `--config` を必須にする

**Files:**
- Modify: `vspeech/main.py:87-100`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: Task 1 の `Config`（`BaseModel` 版）
- Produces: `cmd` の callback シグネチャが `def cmd(config_file: IO[bytes])` になる（`| None` が外れる）。`--config` 未指定時は click が usage error を出して exit 2。

- [ ] **Step 1: Write the failing test**

`tests/test_main.py` の末尾に追加する。ファイル冒頭の import も併せて足す。

```python
import subprocess
import sys
from pathlib import Path
```

```python
def test_the_entry_point_requires_a_config_file():
    """`python -m vspeech` with no --config must fail as a usage error (ADR-0066).

    Run as a real process on purpose: click enforces `required=True` inside
    `main()`, which the callback-level test above bypasses entirely.
    """
    result = subprocess.run(
        [sys.executable, "-m", "vspeech"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )

    assert result.returncode == 2, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "--config" in result.stderr
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_main.py::test_the_entry_point_requires_a_config_file -v`

Expected: FAIL。ただし落ち方に注意する。現状は env フォールバックで全デフォルトの `Config` が組まれ、`receiver` が `[::]:8080` で gRPC サーバを実際に起動して**そのまま待ち続ける**。つまり returncode 2 で失敗するのではなく、`subprocess.TimeoutExpired` で 120 秒後にエラーになる。これがこのタスクで潰す挙動そのもの（無意味なプロセスが黙って立つ）なので、赤の形として正しい。120 秒待ちたくなければ、この 1 回だけ `timeout=10` に落として確認し、実装後に `timeout=120` へ戻す。

- [ ] **Step 3: Make `--config` required and drop the fallback**

`vspeech/main.py` の 87-100 行目を差し替える。

```python
@click.command()
@click.option(
    "--config",
    "--json-config",
    "config_file",
    type=click.File("rb"),
    required=True,
)
def cmd(config_file: IO[bytes]):
    config = Config.read_config_from_file(config_file)
    config_file.close()
```

- [ ] **Step 4: Run it to make sure it passes**

Run: `uv run pytest tests/test_main.py::test_the_entry_point_requires_a_config_file -v`
Expected: PASS

- [ ] **Step 5: Fix the existing event-loop test, which passed `config_file=None`**

`tests/test_main.py::test_cmd_creates_its_own_event_loop_without_a_current_one` は callback を直接呼んでいるので click の必須チェックを迂回するが、`config_file=None` は `read_config_from_file(None)` に落ちて `AttributeError` になる。空の TOML を渡す形に変える。テストの主旨（3.14 のループ取得回帰）は変えない。

シグネチャに `tmp_path` を足す。

```python
def test_cmd_creates_its_own_event_loop_without_a_current_one(tmp_path):
```

`cmd.callback` の呼び出しを差し替える。

```python
    config_file = tmp_path / "config.toml"
    config_file.write_text("", encoding="utf-8")

    # No current event loop in this thread -> the removed get_event_loop()
    # behaviour would raise RuntimeError here on 3.14.
    asyncio.set_event_loop(None)
    try:
        with (
            patch("vspeech.main.vspeech_coro", _noop_coro),
            patch("vspeech.main.configure_logger"),
            patch("vspeech.main.telemetry.configure"),
        ):
            # click types Command.callback as Optional; it is set here.
            assert cmd.callback is not None
            with config_file.open("rb") as opened:
                with pytest.raises(SystemExit):
                    cmd.callback(config_file=opened)
    finally:
```

- [ ] **Step 6: Run the whole entry-point file**

Run: `uv run pytest tests/test_main.py -v`
Expected: 2 passed

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff format . ; uv run ruff check . ; uv run ty check`
Expected: 3 つとも clean。`cmd` から `| None` が外れたので、`ty` が `config_file` の Optional 扱いに文句を言わないことを確認する。

- [ ] **Step 8: Commit**

```bash
git add vspeech/main.py tests/test_main.py
git commit -m "feat(cli)!: require --config and drop the env fallback (ADR-0066)"
```

---

### Task 3: pydantic-settings を依存とランタイムの両方から外す

**Files:**
- Modify: `tests/test_forbidden_imports.py:1-10`, `tests/test_forbidden_imports.py:21`
- Modify: `pyproject.toml:17`

**Interfaces:**
- Consumes: Task 1 で `vspeech/config.py` から消えた `pydantic_settings` の import
- Produces: `FORBIDDEN` タプルに `"pydantic_settings"` が入る。`uv.lock` から pydantic-settings（と推移依存の python-dotenv）が消える。

- [ ] **Step 1: Extend the structural gate**

`tests/test_forbidden_imports.py` の docstring に 1 項目足す。

```python
- pydantic_settings: importing it costs a resident pipeline +13.7 MB RSS / +176 modules /
  473 ms at startup, because its provider barrel imports every backend (AWS / Azure / GCP
  Secret Manager, CLI, dotenv, YAML) unconditionally. Removed in ADR-0066 by taking
  configuration from the `--config` file only.
```

21 行目のタプルに追加する。

```python
FORBIDDEN = ("fairseq", "transformers", "pydantic_settings")
```

同ファイル末尾に、実際にロードされないことを別プロセスで確かめるテストを足す。`test_consumer_path_is_torch_free` と同じ手口。

```python
def test_the_entry_point_never_loads_pydantic_settings():
    """Nothing on the startup path drags the env-config machinery back in (ADR-0066).

    The AST gate above only sees `vspeech/`; this catches a transitive import
    through a dependency. A sys.modules check inside the test process would be
    contaminated by test order, so it runs in a pristine child process.
    """
    code = (
        "import sys\n"
        "import vspeech.main\n"
        "assert 'pydantic_settings' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
```

- [ ] **Step 2: Run the gate**

Run: `uv run pytest tests/test_forbidden_imports.py -v`
Expected: all passed。Task 1 で import を消しているのでここは緑になるのが正しい。ゲートは「二度と戻らない」ことを守るためのもので、赤から入る必要はない。

- [ ] **Step 3: Drop the dependency**

`pyproject.toml` の 17 行目 `"pydantic-settings>=2,<3",` を削除する。16 行目の `"pydantic>=2,<3",` は残す。

- [ ] **Step 4: Re-lock**

Run: `uv lock`
Expected: 成功する。`uv.lock` から `pydantic-settings` と `python-dotenv` のエントリが消える。

Run: `git diff --stat uv.lock`
Expected: uv.lock に削除行が出ている。

- [ ] **Step 5: Sync the environment**

Run: `uv sync --all-extras`
Expected: 成功し、pydantic-settings がアンインストールされる。稼働中のパイプラインがあると os error 5 で失敗するので、その場合は先に止める。

- [ ] **Step 6: Record the actual saving**

Run: `uv run python -c "import sys; import vspeech.main; print(len(sys.modules))"`
Expected: Task 開始前の測定値 **685** より減っている。pydantic-settings 単体の寄与は 176 modules だが、その一部（`ssl` / `importlib.metadata` 等）は grpc や google 系からも引かれるため、実際の減少幅はそれ以下になりうる。**実測値をこのステップの結果として記録する**（見込み値で置き換えない）。

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest`
Expected: all passed。`uv sync` 後の初回なので、extra 由来の収集エラーが無いことも確認する。

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock tests/test_forbidden_imports.py
git commit -m "build(deps)!: drop pydantic-settings and gate it out of the runtime (ADR-0066)"
```

---

### Task 4: コンテナ配備の成果物を削除する

**Files:**
- Delete: `Dockerfile`, `.dockerignore`, `requirements-pod.txt`
- Modify: `poe_tasks.toml:24-30`
- Modify: `.vscode/launch.json:7-110`, `.vscode/launch.json:126-136`

**Interfaces:**
- Consumes: なし
- Produces: `uv run poe` の一覧から `requirements-pod` と `clean` が消える。

- [ ] **Step 1: Check for other references before deleting**

Run: `git grep -n 'requirements-pod\|Dockerfile\|dockerignore'`
Expected: ヒットするのは `CLAUDE.md`（Task 6 で直す）、`poe_tasks.toml`、`docs/adr/0067-*.md`、`docs/superpowers/` 配下のみ。CI ワークフロー（`.github/workflows/codeql.yml`）にヒットが出た場合は、そこも本 Task で直す。

- [ ] **Step 2: Delete the artifacts**

```bash
git rm Dockerfile .dockerignore requirements-pod.txt
```

- [ ] **Step 3: Remove the poe tasks**

`poe_tasks.toml` の 24-30 行目、`requirements-pod` の日本語コメント 4 行とタスク定義、`clean` のコメント 1 行とタスク定義をまとめて削除する。`clean` は `requirements-pod.txt` しか消さないタスクなので道連れにする。

- [ ] **Step 4: Verify the task list**

Run: `uv run poe`
Expected: 一覧に `requirements-pod` と `clean` が無く、他のタスク（`check` / `fix` / `voicevox-assets` / `convert-hubert` / `export-hubert-onnx` / `export-fcpe-onnx` / `metrics`）は残っている。

- [ ] **Step 5: Remove the dead launch configurations**

`.vscode/launch.json` から 2 つの構成を削除する。

- `"name": "Cloud Run: Run/Debug Locally"`（7-110 行目）— 撤去する配備そのもの
- `"name": "gui"`（126-136 行目）— 2026-07-26 に削除済みの `vspeech.gui` を指している（ADR-0061）

`"test"` と `"vspeech"` の 2 構成は残す。`"vspeech"` は `module: vspeech` を引数なしで起動するので、Task 2 で `--config` 必須にした以上そのままでは usage error になる。`args` を足して実行可能な状態にする。

```json
        {
            "name": "vspeech",
            "type": "python",
            "request": "launch",
            "module": "vspeech",
            "args": [
                "--config",
                "config.toml"
            ],
            "justMyCode": true
        }
```

- [ ] **Step 6: Verify the JSON still parses**

Run: `uv run python -c "import json, pathlib; d = json.loads(pathlib.Path('.vscode/launch.json').read_text(encoding='utf-8')); print([c['name'] for c in d['configurations']])"`
Expected: `['test', 'vspeech']`

`launch.json` は行コメントを含む JSONC なので、`json.loads` が落ちる場合はコメント行が原因。その場合はコメントを保ったまま目視で構造を確認する。

- [ ] **Step 7: Commit**

```bash
git add -A Dockerfile .dockerignore requirements-pod.txt poe_tasks.toml .vscode/launch.json
git commit -m "chore(deploy)!: remove the dead container deploy artifacts (ADR-0067)"
```

---

### Task 5: uv の Linux 依存解決を落として再ロックする

**Files:**
- Modify: `pyproject.toml:90-99`（`[tool.uv] environments`）, `pyproject.toml:113-116`（`voicevox-core`）

**Interfaces:**
- Consumes: Task 4 でコンテナ成果物が消えていること
- Produces: `uv.lock` が Windows のみの解決になる。

- [ ] **Step 1: Narrow the resolution target**

`pyproject.toml` の `[tool.uv]` 内、`environments` のコメントと配列を差し替える。

```toml
# voicevox-core / torch / torchaudio / pyvcroid2 are Windows-only wheels, and
# onnxruntime-gpu has no macOS wheel. This project targets Windows only (ADR-0067
# removed the Linux container deploy), so resolve for that platform alone.
environments = [
    "sys_platform == 'win32'",
]
```

- [ ] **Step 2: Drop the manylinux wheel pin**

`[tool.uv.sources]` の `voicevox-core` を単一の URL に戻す。marker 付きリストは Docker イメージのためだけに存在していた。

```toml
voicevox-core = { url = "https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-win_amd64.whl" }
```

- [ ] **Step 3: Re-lock**

Run: `uv lock`
Expected: 成功する。

Run: `git diff --stat uv.lock`
Expected: Linux 専用に解決されていたパッケージが落ちて、`uv.lock` が縮む。

- [ ] **Step 4: Sync and verify nothing broke**

Run: `uv sync --all-extras`
Expected: 成功する。

Run: `uv run pytest`
Expected: all passed

- [ ] **Step 5: Verify the audit surface**

Run: `uv run poe audit`
Expected: 既知の受容済み所見（torch GHSA-rrmf-rvhw-rf47）のみ。新規の所見が出た場合は commit せず報告する。

タスク名が `audit` でない場合は `uv run poe` の一覧で確認する。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): resolve for Windows only now that the Linux image is gone (ADR-0067)"
```

---

### Task 6: ドキュメントを実態に合わせ、ADR を Accepted に昇格する

**Files:**
- Modify: `CLAUDE.md:42-43`, `CLAUDE.md:46`, `CLAUDE.md:121`, `CLAUDE.md:124`
- Modify: `docs/adr/0066-config-input-file-only.md:3`, `docs/adr/0067-drop-container-deploy-path.md:3`, `docs/adr/README.md`

**Interfaces:**
- Consumes: Task 1-5 の全変更
- Produces: なし（最終タスク）

- [ ] **Step 1: Remove the Docker requirements section from the command list**

`CLAUDE.md` の 42-43 行目、以下の 2 行を削除する。

```
# Regenerate the Docker requirements file (Linux deploy image, voicevox extra)
uv run poe requirements-pod
```

- [ ] **Step 2: Fix the retired-Makefile sentence**

46 行目の末尾、`その `requirements-pod` / `voicevox-assets` / `clean` targets are now poe tasks.` を、残っているタスクだけを指すよう直す。

```
The former `Makefile` is retired — its `voicevox-assets` target is now a poe task, and its `requirements-pod` / `clean` targets went away with the container deploy (ADR-0067).
```

- [ ] **Step 3: Rewrite the config-loading line**

121 行目を差し替える。

```
- **Config loading** (`Config` in `config.py`): from a **required** `--config` file (TOML, or JSON if the name ends in `.json`) and nothing else — the environment-variable path and its pydantic-settings dependency were removed in [ADR-0066](docs/adr/0066-config-input-file-only.md), so `python -m vspeech` with no `--config` is a usage error. `Config` sets `extra="forbid"` explicitly (it used to inherit it from `SettingsConfigDict`); note this guards the top level only. Secrets (`ami.appkey`, `gcp.service_account_info`) are `SecretStr`.
```

- [ ] **Step 4: Fix the platform-constraints paragraph**

124 行目のうち、コンテナ配備を前提にした 2 箇所を置換する。長い 1 段落なので、以下の文字列を厳密に置き換える。

置換 1 — 以下を、

```
`voicevox-core` is pinned per-platform there (a `marker`ed list: `win_amd64` for dev, `manylinux_2_34_x86_64` for the Docker image) — don't put a `sys_platform` marker back on the `voicevox` extra itself, or `uv export` silently drops it from `requirements-pod.txt`.
```

こう直す。

```
`voicevox-core` is pinned there to the single `win_amd64` wheel; the per-platform `marker`ed list existed only for the Docker image and went away with it ([ADR-0067](docs/adr/0067-drop-container-deploy-path.md)).
```

置換 2 — 以下を、

```
Dev target is Windows; the Docker image targets Linux.
```

こう直す。

```
Dev and deploy target is Windows only — `[tool.uv] environments` resolves for `sys_platform == 'win32'` alone since [ADR-0067](docs/adr/0067-drop-container-deploy-path.md) removed the Linux container image.
```

- [ ] **Step 5: Verify no stale references remain**

Run: `git grep -n 'requirements-pod\|Dockerfile\|dockerignore\|manylinux\|pydantic-settings\|pydantic_settings'`
Expected: ヒットするのは `docs/adr/0066-*.md` / `0067-*.md`（決定の記録なので残るのが正しい）、`docs/superpowers/specs/` と `docs/superpowers/plans/` 配下（スナップショットなので残るのが正しい）、`tests/test_forbidden_imports.py`（ゲートなので残るのが正しい）のみ。`CLAUDE.md` にヒットが残っていたら直す。

- [ ] **Step 6: Promote both ADRs to Accepted**

実装が決定を裏づけたので、`docs/adr/0066-config-input-file-only.md` と `docs/adr/0067-drop-container-deploy-path.md` の 3 行目をそれぞれ 1 行だけ書き換える。本文には触れない。

```
- Status: Accepted
```

`docs/adr/README.md` の索引で、0066 と 0067 の Status 列を `Proposed` から `Accepted` に変える。

- [ ] **Step 7: Run the full health gate**

Run: `uv run poe check`
Expected: 既知の受容済み所見（torch の audit 所見、`vr2_config` の deadcode）以外は clean。新規の所見が出たら commit せず報告する。

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md docs/adr/0066-config-input-file-only.md docs/adr/0067-drop-container-deploy-path.md docs/adr/README.md
git commit -m "docs: align the guide with file-only config and promote ADR-0066/0067"
```

---

## 実機確認（マージ前に必須）

テストは `python -m vspeech` を `--config` 無しで叩く経路しか通っていない。**設定ファイルを渡して実際にパイプラインが上がることを、この計画の完了前に必ず 1 回確認する。** 過去に「エントリポイントを動かさずテストだけ通して壊れていた」事例が 2 回ある（3.14 の `get_event_loop`、logger の cp1252/PIPE）。

```sh
uv run python -m vspeech --config ./config.toml
```

worker の起動ログが出て、`vsctl ping` が通ることまで見る。
