# GUI run-readiness 情報整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GUI を開いた技術者が、起動前に「この pipeline が動くのに何が足りないか」を見て、直して、自信を持って Start できるようにする。

**Architecture:** 「何が必須か」の権威は既存の `vspeech.preflight`（ADR-0038）に置いたまま、GUI はそれを再利用して起動前に描画する（ADR-0045）。GUI 側に必須項目の知識を複製しない。純ロジック（`gui/readiness.py` / `gui/shared_paths.py`）は Tk 非依存で TDD し、Tk 層は手動 smoke 検証に留める（ADR-0032 の既定路線）。マシン共通の素材パスは `default.toml` で一度編集し明示 propagate する（ADR-0046）。

**Tech Stack:** Python 3.14 / pydantic v2 / ttkbootstrap (Tk) / pytest (`asyncio_mode = "auto"`) / uv / ruff / ty

**ADR:** [ADR-0045](../../adr/0045-gui-readiness-reuses-preflight.md)（readiness は preflight を再利用）, [ADR-0046](../../adr/0046-gui-shared-asset-paths-explicit-propagate.md)（共有素材パスは明示 propagate）。どちらも `Proposed` — 実装が裏づけたら最終レビュー時に `Accepted` へ昇格する。

**Spec:** [2026-07-16-gui-run-readiness-design.md](../specs/2026-07-16-gui-run-readiness-design.md)

## Global Constraints

- Python **3.14 のみ**（`>=3.14,<3.15`）。依存操作は `uv`。
- **import は 1 行 1 つ**（ruff `force-single-line = true`）。型検査は `ty`（pyright ではない）。
- **pydantic v2 API のみ**。v1 API（`parse_obj` / `.dict()` / `root_validator` / `json_encoders`）を持ち込まない。
- **依存方向は gui → vspeech の一方向のみ**（ADR-0032）。`vspeech/` から `gui/` を import しない。
- **必須項目の知識を GUI に複製しない**（ADR-0045）。「何が必須か」の判断は `vspeech.preflight` にしか置かない。
- **pipeline config は自己完結**（ADR-0032 / ADR-0046）。起動時に別ファイルと合成しない。`python -m vspeech --config pipelines/<id>.toml` 単体で起動できること。
- **`ConfigProblem.__str__` と既存の起動時ログ出力は不変**（ADR-0045 の変更は加算的）。
- **純ロジックは TDD、Tk UI 層は手動 smoke 検証**（ADR-0032）。Tk widget に単体テストを書かない。
- **対象ユーザーは技術者**。ラベルの平易語化・ウィザード化はしない（spec 非ゴール）。config のフィールド名はそのまま出してよい。
- 検証コマンドは `uv run pytest <完全な node ID>` / `uv run ty check`（**プロジェクト全体**。ファイル指定はテストファイルの型エラーを見逃す）/ `uv run ruff format .` / `uv run ruff check .`。
- **コミットは全て**メッセージ末尾に trailer を付ける。Bash ツールから打つときは heredoc を使う:

```bash
git commit -F - <<'EOF'
<subject>

<body>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

## File Structure

| ファイル | 責務 |
|---|---|
| `vspeech/exceptions.py`（変更） | `ConfigProblem` に省略可能な `field` を足す |
| `vspeech/preflight.py`（変更） | `collect_problems()` を分離し、各問題に `field` を付ける |
| `gui/readiness.py`（新規） | `collect_problems` を呼び worker 単位に整形するだけの純ロジック。必須知識は持たない |
| `gui/readiness_panel.py`（新規） | readiness を描く Tk widget。flow 行 + ✓/✗ 行 + 「→修正」 |
| `gui/pipeline_editor.py`（変更） | パネルを載せ、Start をゲートし、失敗バナーを出す |
| `gui/form.py`（変更） | `focus_field(path)` と問題フィールドの ✗ 印、資産/資格フィールドを上へ |
| `gui/process.py`（変更） | 即死（起動直後の異常終了）を呼び出し側へ伝える |
| `gui/app.py`（変更） | 即死バナーの配線、共有パス編集の導線と propagate |
| `gui/config_paths.py`（新規） | ドット path での get/set。`form.py` の private ヘルパを抽出して共用 |
| `gui/shared_paths.py`（新規） | 共有素材パスの定義と propagate の純ロジック |
| `tests/test_preflight.py`（変更） | `collect_problems` と `field` の検証 |
| `tests/test_gui_readiness.py`（新規） | readiness 整形の検証 |
| `tests/test_gui_shared_paths.py`（新規） | 共有パス propagate の検証 |

---

### Task 1: `collect_problems` の分離と `ConfigProblem.field`

「何が必須か」の権威を GUI から例外なしで呼べるようにし、問題から設定箇所へ飛べるよう構造化フィールドを足す（ADR-0045）。挙動とログ出力は不変。

**Files:**
- Modify: `vspeech/exceptions.py:31-37`
- Modify: `vspeech/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: 既存の `Config` / `_CHECKERS`
- Produces:
  - `ConfigProblem(worker: str, detail: str, field: str | None = None)` — frozen dataclass
  - `vspeech.preflight.collect_problems(config: Config) -> list[ConfigProblem]`
  - `vspeech.preflight.preflight(config: Config) -> None`（従来どおり非空なら `ConfigError` 送出）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_preflight.py` の末尾に追記する:

```python
def test_collect_problems_returns_list_without_raising():
    problems = collect_problems(_acp())  # ACP 4 フィールドすべて空
    assert [p.worker for p in problems] == ["transcription"] * 4
    assert {p.field for p in problems} == {
        "ami.appkey",
        "ami.engine_uri",
        "ami.engine_name",
        "ami.service_id",
    }


def test_collect_problems_empty_for_clean_config():
    assert collect_problems(Config()) == []


def test_preflight_still_raises_on_problems():
    with pytest.raises(ConfigError):
        preflight(_acp())


def test_config_problem_str_is_unchanged_by_field():
    problem = ConfigProblem("vc", "rvc.model_file '' が存在しません", "rvc.model_file")
    assert str(problem) == "[vc] rvc.model_file '' が存在しません"


def test_vc_problems_carry_their_field():
    config = Config(vc=VcConfig(enable=True))
    fields = {p.field for p in collect_problems(config)}
    assert "rvc.model_file" in fields
    assert "rvc.hubert_model_file" in fields
    assert "rvc.rmvpe_model_file" in fields  # f0_extractor_type の既定は rmvpe
```

同ファイル冒頭の import に 3 行足す（1 行 1 import）:

```python
from vspeech.config import VcConfig
from vspeech.exceptions import ConfigProblem
from vspeech.preflight import collect_problems
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_preflight.py::test_collect_problems_returns_list_without_raising -v`
Expected: FAIL — `ImportError: cannot import name 'collect_problems' from 'vspeech.preflight'`

- [ ] **Step 3: `ConfigProblem` に `field` を足す**

`vspeech/exceptions.py` の `ConfigProblem` を置き換える。`field` は**既定値付きの第3フィールド**なので、既存の `ConfigProblem(w, "msg")` 位置引数呼び出しはすべてそのまま通る。`__str__` は変えない（ADR-0045: 既存ログ出力は不変）:

```python
@dataclass(frozen=True)
class ConfigProblem:
    worker: str
    detail: str
    field: str | None = None
    """問題の設定箇所のドット path (例 "rvc.model_file")。GUI がそこへ移動するために使う (ADR-0045)。"""

    def __str__(self) -> str:
        return f"[{self.worker}] {self.detail}"
```

- [ ] **Step 4: `collect_problems` を分離する**

`vspeech/preflight.py` 末尾の `preflight` を置き換える:

```python
def collect_problems(config: Config) -> list[ConfigProblem]:
    """enable 済み worker の設定不備を集約して返す（送出しない）。

    GUI の起動前 readiness がこれを単一の権威として再利用する (ADR-0045)。
    「何が必須か」の判断をこの module の外に複製しないこと。
    """
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    return problems


def preflight(config: Config) -> None:
    problems = collect_problems(config)
    if problems:
        raise ConfigError(problems)
```

- [ ] **Step 5: 各チェッカーに `field` を付ける**

`vspeech/preflight.py` の各 `ConfigProblem(...)` に `field=` を足す。detail 文言は変えない。

`_check_gcp_credentials`:

```python
            return [
                ConfigProblem(
                    worker,
                    f"gcp.service_account_file_path '{path}' が存在しません",
                    field="gcp.service_account_file_path",
                )
            ]
```

`_check_vad_gate`（`worker` は `"transcription"` か `"vc"` なので path はそこから組める）:

```python
        return [
            ConfigProblem(
                worker,
                f"vad_gate=true ですが vad_model_file '{path}' が存在しません",
                field=f"{worker}.vad_model_file",
            )
        ]
```

`_check_transcription` の ACP ループと mozc:

```python
        for name, value in required:
            if not value:
                problems.append(
                    ConfigProblem(
                        w,
                        f"ACP バックエンドには {name} が必須ですが空です",
                        field=name,
                    )
                )
        if tc.transliterate_with_mozc and find_spec("mozcpy") is None:
            problems.append(
                ConfigProblem(
                    w,
                    "transliterate_with_mozc=true ですが mozcpy が未インストールです",
                    field="transcription.transliterate_with_mozc",
                )
            )
```

`_check_recording`:

```python
    try:
        resolve_input_device(config.recording)
    except DeviceNotFoundError as e:
        problems.append(
            ConfigProblem(w, str(e), field="recording.input_device_index")
        )
    try:
        WorkerOutput.from_routes_list(config.recording.routes_list)
    except Exception as e:
        problems.append(
            ConfigProblem(
                w,
                f"recording.routes_list が不正です: {e}",
                field="recording.routes_list",
            )
        )
```

`_check_playback`:

```python
    try:
        resolve_output_device(config.playback)
    except DeviceNotFoundError as e:
        return [
            ConfigProblem("playback", str(e), field="playback.output_device_index")
        ]
```

`_check_tts`:

```python
    for name, path in (
        ("voicevox.openjtalk_dir", vv.openjtalk_dir),
        ("voicevox.model_dir", vv.model_dir),
    ):
        if not path.expanduser().is_dir():
            problems.append(
                ConfigProblem(w, f"{name} '{path}' が存在しません", field=name)
            )
    if (
        vv.onnxruntime_path is not None
        and not vv.onnxruntime_path.expanduser().is_file()
    ):
        problems.append(
            ConfigProblem(
                w,
                f"voicevox.onnxruntime_path '{vv.onnxruntime_path}' が存在しません",
                field="voicevox.onnxruntime_path",
            )
        )
```

`_check_vc`:

```python
    if not rvc.model_file.expanduser().is_file():
        problems.append(
            ConfigProblem(
                w,
                f"rvc.model_file '{rvc.model_file}' が存在しません",
                field="rvc.model_file",
            )
        )
    hubert = rvc.hubert_model_file
    if hubert == Path() or not hubert.expanduser().is_dir():
        problems.append(
            ConfigProblem(
                w,
                f"rvc.hubert_model_file '{hubert}' (資産ディレクトリ) が存在しません",
                field="rvc.hubert_model_file",
            )
        )
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        if not rvc.rmvpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    w,
                    f"rvc.rmvpe_model_file '{rvc.rmvpe_model_file}' が存在しません",
                    field="rvc.rmvpe_model_file",
                )
            )
```

`_check_subtitle`（url / text_source / 色）:

```python
    if not obs.url:
        problems.append(
            ConfigProblem(
                w,
                "OBS バックエンドには subtitle.obs.url が必須ですが空です",
                field="subtitle.obs.url",
            )
        )
    elif not obs.url.startswith(("ws://", "wss://")):
        problems.append(
            ConfigProblem(
                w,
                f"subtitle.obs.url '{obs.url}' は ws:// か wss:// で始まる必要があります",
                field="subtitle.obs.url",
            )
        )
```

```python
    if not obs.text_source:
        problems.append(
            ConfigProblem(
                w,
                "OBS バックエンドには subtitle.obs.text_source が必須ですが空です",
                field="subtitle.obs.text_source",
            )
        )
```

```python
    for name, value in (
        ("subtitle.text.font_color", subtitle.text.font_color),
        ("subtitle.text.outline_color", subtitle.text.outline_color),
        ("subtitle.translated.font_color", subtitle.translated.font_color),
        ("subtitle.translated.outline_color", subtitle.translated.outline_color),
    ):
        try:
            hex_color_to_obs_int(value)
        except ValueError as e:
            problems.append(ConfigProblem(w, f"{name}: {e}", field=name))
    if subtitle.bg_color != TRANSPARENT_BG_COLOR:
        try:
            hex_color_to_obs_int(subtitle.bg_color)
        except ValueError as e:
            problems.append(
                ConfigProblem(w, f"subtitle.bg_color: {e}", field="subtitle.bg_color")
            )
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: PASS（新規 5 件 + 既存すべて。既存テストは `p.detail` / `p.worker` しか見ていないので影響なし）

- [ ] **Step 7: 全体の型検査と整形**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: すべて exit 0。`ty check` は**プロジェクト全体**で走らせる（ファイル指定だとテストファイルの型エラーを見逃す）。

- [ ] **Step 8: コミット**

```bash
git add vspeech/exceptions.py vspeech/preflight.py tests/test_preflight.py
git commit -F - <<'EOF'
feat(preflight): collect_problems を分離し ConfigProblem に field を足す (ADR-0045)

GUI の起動前 readiness が「何が必須か」の権威である preflight を例外制御
フロー無しで再利用できるようにする。field は問題から設定箇所へ移動するための
構造化 path。どちらも加算的変更で、起動時の挙動と ConfigProblem.__str__ の
ログ出力は不変。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: readiness の整形（純ロジック）

`collect_problems` の結果を worker 単位へ整形し、配線（flow）を導出する。**必須項目の知識はここに持たない** — 呼ぶだけ。

**Files:**
- Create: `gui/readiness.py`
- Test: `tests/test_gui_readiness.py`

**Interfaces:**
- Consumes: `vspeech.preflight.collect_problems`, `vspeech.exceptions.ConfigProblem`（Task 1）
- Produces:
  - `gui.readiness.WORKER_NAMES: tuple[str, ...]`
  - `gui.readiness.enabled_workers(config: Config) -> list[str]`
  - `gui.readiness.flow_of(config: Config) -> list[list[str]]`
  - `gui.readiness.WorkerReadiness(worker: str, problems: list[ConfigProblem])`、`.ok -> bool`
  - `gui.readiness.Readiness(workers: list[WorkerReadiness], flow: list[list[str]], error: str | None)`、`.ok -> bool`、`.problem_count -> int`
  - `gui.readiness.evaluate(config: Config) -> Readiness`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gui_readiness.py` を新規作成:

```python
import pytest

from gui.readiness import enabled_workers
from gui.readiness import evaluate
from gui.readiness import flow_of
from vspeech.config import Config
from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import VcConfig


def test_nothing_enabled_is_ready():
    readiness = evaluate(Config())
    assert readiness.ok
    assert readiness.problem_count == 0
    assert readiness.workers == []


def test_enabled_workers_follows_enable_flags():
    config = Config(
        recording=RecordingConfig(enable=True), playback=PlaybackConfig(enable=True)
    )
    assert enabled_workers(config) == ["recording", "playback"]


def test_vc_without_assets_is_not_ready_and_groups_by_worker():
    readiness = evaluate(Config(vc=VcConfig(enable=True)))
    assert not readiness.ok
    assert [w.worker for w in readiness.workers] == ["vc"]
    fields = {p.field for p in readiness.workers[0].problems}
    assert "rvc.model_file" in fields
    assert not readiness.workers[0].ok


def test_flow_starts_at_recording_when_recording_seeds():
    config = Config(
        recording=RecordingConfig(enable=True, routes_list=[["vc", "playback"]])
    )
    assert flow_of(config) == [["recording", "vc", "playback"]]


def test_flow_uses_text_send_operations_without_recording():
    config = Config(text_send_operations=[["tts", "playback"]])
    assert flow_of(config) == [["(text)", "tts", "playback"]]


def test_evaluation_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    def boom(_config: Config) -> list[object]:
        raise ImportError("sounddevice missing")

    monkeypatch.setattr("gui.readiness.collect_problems", boom)
    readiness = evaluate(Config(vc=VcConfig(enable=True)))
    assert not readiness.ok
    assert readiness.error is not None
    assert "sounddevice missing" in readiness.error
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_gui_readiness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.readiness'`

- [ ] **Step 3: `gui/readiness.py` を実装する**

```python
"""起動前 readiness の評価 (ADR-0045)。

「何が必須か」の権威は vspeech.preflight (ADR-0038) にあり、この module は
それを呼んで表示用に整形するだけ。必須項目の知識をここへ複製しないこと —
複製すると GUI が緑なのに起動して落ちる (逆も) という形で drift が出る。
"""

from dataclasses import dataclass

from vspeech.config import Config
from vspeech.exceptions import ConfigProblem
from vspeech.preflight import collect_problems

# config に <name>.enable を持つ worker。preflight のチェッカーと対象を揃える。
WORKER_NAMES: tuple[str, ...] = (
    "recording",
    "transcription",
    "translation",
    "tts",
    "vc",
    "playback",
    "subtitle",
)


@dataclass(frozen=True)
class WorkerReadiness:
    worker: str
    problems: list[ConfigProblem]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class Readiness:
    workers: list[WorkerReadiness]
    flow: list[list[str]]
    error: str | None = None
    """readiness の評価自体が失敗した理由 (成立していれば None)。"""

    @property
    def ok(self) -> bool:
        return self.error is None and all(worker.ok for worker in self.workers)

    @property
    def problem_count(self) -> int:
        return sum(len(worker.problems) for worker in self.workers)


def enabled_workers(config: Config) -> list[str]:
    return [name for name in WORKER_NAMES if getattr(config, name).enable]


def flow_of(config: Config) -> list[list[str]]:
    """この pipeline の配線。recording が種なら routes_list、テキスト起点なら
    text_send_operations が実際に使われる鎖。どちらも鎖には種自身が入らない
    ので、表示用に先頭へ足す。"""
    if config.recording.enable:
        return [["recording", *chain] for chain in config.recording.routes_list]
    return [["(text)", *chain] for chain in config.text_send_operations]


def evaluate(config: Config) -> Readiness:
    workers = enabled_workers(config)
    try:
        problems = collect_problems(config)
    except Exception as e:
        # preflight 自体が評価不能 (例: audio extra 未導入でデバイス検査が
        # import 段で失敗)。readiness の失敗で GUI を落とさない (ADR-0045)。
        return Readiness(
            workers=[WorkerReadiness(worker, []) for worker in workers],
            flow=flow_of(config),
            error=f"readiness を評価できませんでした: {e}",
        )
    by_worker: dict[str, list[ConfigProblem]] = {worker: [] for worker in workers}
    for problem in problems:
        # enable 済みの worker しか問題を出さないはずだが、取りこぼしを
        # 握り潰さず末尾に見せる。
        by_worker.setdefault(problem.worker, []).append(problem)
    return Readiness(
        workers=[
            WorkerReadiness(worker, problems) for worker, problems in by_worker.items()
        ],
        flow=flow_of(config),
    )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_gui_readiness.py -v`
Expected: PASS（6 件）

- [ ] **Step 5: 全体の型検査と整形、コミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: すべて exit 0

```bash
git add gui/readiness.py tests/test_gui_readiness.py
git commit -F - <<'EOF'
feat(gui): preflight を再利用する readiness 整形を足す (ADR-0045)

collect_problems の結果を worker 単位へ束ね、配線を導出するだけの純ロジック。
必須項目の知識は持たず vspeech.preflight に委ねる。評価自体が失敗しても
例外を出さず error として返し、GUI を落とさない。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: readiness パネル（Tk widget）

readiness を描く widget。ADR-0032 の既定路線どおり、Tk 層は単体テストを書かず手動 smoke で見る。

**Files:**
- Create: `gui/readiness_panel.py`

**Interfaces:**
- Consumes: `gui.readiness.Readiness` / `WorkerReadiness`（Task 2）
- Produces:
  - `gui.readiness_panel.ReadinessPanel(master, on_fix: Callable[[str], None])`
  - `.show(readiness: Readiness) -> None`
  - `.clear() -> None`

- [ ] **Step 1: `gui/readiness_panel.py` を実装する**

```python
from collections.abc import Callable
from tkinter import BOTH
from tkinter import LEFT
from tkinter import W
from tkinter import X
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label

from gui.readiness import Readiness


class ReadinessPanel(Frame):
    """「この pipeline は起動できるか」を起動前に見せるパネル (ADR-0045)。

    行の中身は vspeech.preflight が出した問題そのもので、この widget は
    判断をしない — 描くだけ。
    """

    def __init__(self, master: Any, on_fix: Callable[[str], None]):
        super().__init__(master, padding=(0, 0, 0, 6))
        self.on_fix = on_fix
        self.flow = Label(self, text="", bootstyle="secondary")
        self.flow.pack(fill=X, anchor=W)
        self.rows = Frame(self)
        self.rows.pack(fill=BOTH, expand=True)

    def clear(self) -> None:
        self.flow.configure(text="")
        for child in list(self.rows.children.values()):
            child.destroy()

    def show(self, readiness: Readiness) -> None:
        self.clear()
        self.flow.configure(text=self._flow_text(readiness))
        if readiness.error is not None:
            Label(self.rows, text=f"⚠ {readiness.error}", bootstyle="warning").pack(
                fill=X, anchor=W
            )
            return
        if not readiness.workers:
            Label(
                self.rows,
                text="有効な worker がありません (この pipeline は何もしません)",
                bootstyle="warning",
            ).pack(fill=X, anchor=W)
            return
        for worker in readiness.workers:
            if worker.ok:
                Label(
                    self.rows, text=f"✓ {worker.worker}", bootstyle="success"
                ).pack(fill=X, anchor=W)
                continue
            for problem in worker.problems:
                self._problem_row(worker.worker, problem.detail, problem.field)

    def _problem_row(self, worker: str, detail: str, field: str | None) -> None:
        row = Frame(self.rows)
        row.pack(fill=X, anchor=W)
        Label(row, text=f"✗ {worker}  {detail}", bootstyle="danger").pack(
            side=LEFT, anchor=W
        )
        if field is not None:
            Button(
                row,
                text="→修正",
                bootstyle="link-danger",
                command=lambda: self.on_fix(field),
            ).pack(side=LEFT)

    def _flow_text(self, readiness: Readiness) -> str:
        if not readiness.flow:
            return ""
        return "  |  ".join(" → ".join(chain) for chain in readiness.flow)
```

- [ ] **Step 2: 型検査と整形**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: すべて exit 0

- [ ] **Step 3: コミット**

```bash
git add gui/readiness_panel.py
git commit -F - <<'EOF'
feat(gui): readiness を描く Tk パネルを足す (ADR-0045)

flow 行 + worker 毎の ✓ / 問題毎の ✗ 行 + 設定箇所へ飛ぶ「→修正」。
判断は一切せず preflight が出した問題をそのまま描く。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 4: エディタ統合・Start ゲート・「→修正」の着地

パネルを載せ、Start を readiness でゲートし（未充足なら無効化＋理由、明示操作の escape hatch あり）、「→修正」でフォームの該当欄へ飛ばす。

**Files:**
- Modify: `gui/form.py`（`focus_field` を足す）
- Modify: `gui/pipeline_editor.py`

**Interfaces:**
- Consumes: `gui.readiness.evaluate`（Task 2）, `gui.readiness_panel.ReadinessPanel`（Task 3）
- Produces:
  - `gui.form.PipelineForm.focus_field(path: str) -> bool`
  - `gui.pipeline_editor.PipelineEditor.readiness_ok -> bool`
  - `gui.pipeline_editor.PipelineEditor.refresh_readiness() -> None`

- [ ] **Step 1: `PipelineForm.focus_field` を足す**

`gui/form.py` の `read_into` の直前に追記する。`self.bindings` が widget → (path, coerce) を持っているので逆引きで足りる:

```python
    def focus_field(self, path: str) -> bool:
        """`path` に束ねた widget へスクロールしてフォーカスする。

        見つかれば True。worker が無効で畳まれている等で widget が無ければ
        False（呼び出し側がその旨を出す）。
        """
        for widget, (bound_path, _coerce) in self.bindings.items():
            if bound_path != path or not widget.winfo_exists():
                continue
            # ScrolledFrame の中身なので、まず可視域へ入れてからフォーカス。
            self.body.update_idletasks()
            try:
                self.body.yview_moveto(
                    max(0.0, widget.winfo_rooty() - self.body.winfo_rooty())
                    / max(1, self.body.winfo_height())
                )
            except TclError:
                pass
            widget.focus_set()
            return True
        return False
```

`gui/form.py` 冒頭の import に 1 行足す:

```python
from tkinter import TclError
```

- [ ] **Step 2: エディタにパネルと readiness 状態を持たせる**

`gui/pipeline_editor.py` の import に足す（1 行 1 つ）:

```python
from gui.readiness import Readiness
from gui.readiness import evaluate
from gui.readiness_panel import ReadinessPanel
```

`__init__` の `self.banner` を作った直後（`notebook` を作る前）に挿す:

```python
        self.readiness: Readiness | None = None
        # 「preflight は不足と言うが承知の上で起動する」を 1 回だけ通すフラグ。
        self.force_start = False
        self.panel = ReadinessPanel(self, on_fix=self._focus_field)
        self.panel.pack(fill=X)
```

**注**: `self.panel` は `notebook` より前に作るが、`_focus_field` が参照する `self.notebook` / `self.form` はボタンを押した時にしか要らないので順序問題は起きない。`__init__` 末尾の既存の `self._refresh_start_button()` は、後段の Step 4 で置く `self.start_hint` / `self.force_bt` を参照する — どちらも `controls` フレーム（`notebook` の後）で作るので、その時点では存在している。

- [ ] **Step 3: Start ゲートと再評価を実装する**

`gui/pipeline_editor.py` の `_refresh_start_button` を置き換える:

```python
    @property
    def readiness_ok(self) -> bool:
        return self.readiness is None or self.readiness.ok

    def refresh_readiness(self) -> None:
        """フォームの現在値を config へ書き戻してから再評価する。

        判断は vspeech.preflight が持つ (ADR-0045) ので、ここは呼ぶだけ。
        preflight はデバイス列挙などの I/O を伴うため、呼ぶのは
        load / save / Start / タブ切替の時だけにする（毎キーストロークでは呼ばない）。
        """
        if self.config is None:
            self.readiness = None
            self.panel.clear()
        else:
            self.sync_form_to_config()
            self.readiness = evaluate(self.config)
            self.panel.show(self.readiness)
        self._refresh_start_button()

    def _focus_field(self, path: str) -> None:
        self.notebook.select(self.form)
        if not self.form.focus_field(path):
            self.banner.configure(
                text=f"⚠ {path} はフォームに出ていません（Raw TOML で編集してください）"
            )

    def _refresh_start_button(self) -> None:
        can_start = self.entry is not None and not self.broken and not self.running
        gated = can_start and not self.readiness_ok and not self.force_start
        self.start_bt.configure(state=DISABLED if gated or not can_start else NORMAL)
        if gated and self.readiness is not None:
            self.start_hint.configure(
                text=f"{self.readiness.problem_count} 件未充足"
            )
            self.force_bt.pack(side=LEFT, padx=4)
        else:
            self.start_hint.configure(text="")
            self.force_bt.pack_forget()
```

- [ ] **Step 4: 未充足件数ラベルと escape hatch ボタンを置く**

`gui/pipeline_editor.py` の `__init__` で、`self.stop_bt.pack(...)` の直後・`self.status` の前に挿す:

```python
        self.start_hint = Label(controls, text="", bootstyle="danger")
        self.start_hint.pack(side=LEFT, padx=(0, 4))
        # 未充足でも起動する escape hatch。技術者が「preflight は不足と言うが
        # 承知の上で起動する」を選べるように小さく残す（既定は無効化側）。
        self.force_bt = Button(
            controls,
            text="未充足でも起動",
            bootstyle="link-danger",
            command=self._force_start_click,
        )
```

（`self.force_bt` はここで `pack` しない — `_refresh_start_button` が出し入れする。）

`_start_click` の直後に足す:

```python
    def _force_start_click(self) -> None:
        self.force_start = True
        try:
            self.on_start()
        finally:
            self.force_start = False
        self._refresh_start_button()
```

- [ ] **Step 5: readiness を再評価する箇所を配線する**

`load_entry` の末尾 `self._refresh_start_button()` を置き換える:

```python
        self.refresh_readiness()
```

`apply_raw` の末尾 `self.on_dirty()` の直前の `self._refresh_start_button()` を置き換える:

```python
        self.refresh_readiness()
```

`save` の末尾、`logger.info("saved pipeline %s", self.entry.id)` の直前に足す:

```python
        self.refresh_readiness()
```

`clear` の `self._refresh_start_button()` の直前に足す:

```python
        self.readiness = None
        self.panel.clear()
```

- [ ] **Step 6: 手動 smoke 検証**

Run: `uv run python -m gui`

確認すること（ADR-0032 どおり Tk 層は手動検証）:
1. 「+ new」→「マイク→ボイチェン→再生」を作る。パネルに `recording → vc → playback` の flow 行が出る。
2. `rvc.model_file` 等が未設定なので `✗ vc  rvc.model_file '' が存在しません` が出て、**Start が無効**・`3 件未充足`・「未充足でも起動」が現れる。
3. `→修正` を押すと Form タブへ切り替わり `rvc model_file` 欄にフォーカスが当たる。
4. 「マイク→再生 (モニター)」を作ると `✓ recording` / `✓ playback` が出て **Start が有効**になる。
5. Start して起動することを確認し、Stop する。

Expected: 上記 5 点すべて。

- [ ] **Step 7: 型検査・整形・全テスト・コミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: すべて exit 0

```bash
git add gui/form.py gui/pipeline_editor.py
git commit -F - <<'EOF'
feat(gui): 起動前 readiness をエディタに載せ Start をゲートする (ADR-0045)

pipeline を選んだ時点で「起動できるか / 何が足りないか」がパネルに出る。
未充足なら Start を無効化して件数を出し、「→修正」でその設定箇所へ飛ぶ。
承知の上で起動する技術者向けに escape hatch を小さく残す。preflight は I/O を
伴うので再評価は load/save/apply/Start の時だけ。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: 即死の失敗バナー

起動直後に落ちた場合、`process exited: N` で終わらせず理由を目立たせる（摩擦#2）。readiness が起動前に config 不備を止めるので、ここに落ちてくるのは主に層B（実ロード・GPU・VOICEROID2・OBS 接続）の失敗。

**Files:**
- Modify: `gui/process.py`
- Modify: `gui/app.py`
- Modify: `gui/pipeline_editor.py`

**Interfaces:**
- Consumes: `gui.pipeline_editor.PipelineEditor`（Task 4）
- Produces:
  - `gui.process.PipelineRunner.started_at: float | None`
  - `gui.process.PipelineRunner.stopping: bool`
  - `gui.process.PipelineRunner.ran_for() -> float`
  - `gui.pipeline_editor.PipelineEditor.show_launch_failure(lines: list[str]) -> None`

- [ ] **Step 1: runner に起動からの経過と「停止中」フラグを持たせる**

`gui/process.py` の import に足す:

```python
from time import monotonic
```

`PipelineRunner.__init__` の末尾に足す:

```python
        self.started_at: float | None = None
        # ユーザーが Stop / delete / window close で意図的に止めたか。
        # 即死 (起動失敗) と区別するために要る — terminate の exit code は
        # 非 0 なので、これが無いと通常停止まで「起動失敗」と誤認する。
        self.stopping = False
```

`start()` の `self.proc = Popen(...)` の直前に足す:

```python
        self.started_at = monotonic()
        self.stopping = False
```

`request_stop()` の `if self.proc and self.is_running():` の直前に足す:

```python
        self.stopping = True
```

`stop()` の `proc = self.proc` の直前に足す:

```python
        self.stopping = True
```

`is_running` の直前に足す:

```python
    def ran_for(self) -> float:
        """start() からの経過秒。未起動なら 0.0。"""
        return 0.0 if self.started_at is None else monotonic() - self.started_at
```

- [ ] **Step 2: エディタに失敗バナーを足す**

`gui/pipeline_editor.py` の `show_launch_failure` を `append_log` の直後に足す:

```python
    def show_launch_failure(self, lines: list[str]) -> None:
        """起動直後に落ちた理由をログを開かずに見せる。

        vspeech は設定不備なら「起動中止: 設定不備 N 件」+ 各問題 を整形して
        吐いて exit する (ADR-0038) ので、末尾数行がそのまま理由になる。
        """
        reason = " / ".join(line for line in lines if line.strip())
        self.banner.configure(text=f"❗ 起動に失敗しました: {reason}"[:400])
```

- [ ] **Step 3: app 側で即死を判定して回す**

`gui/app.py` の先頭付近、`LOG_BUFFER_MAX = 2000` の下に足す:

```python
# これより早く落ちたら「起動に失敗した」とみなす。正常起動なら worker が
# 走り続けるので、この窓で終わるのは起動失敗だけ。
QUICK_EXIT_SEC = 10.0
# 失敗バナーに載せるログ末尾の行数。
FAILURE_TAIL_LINES = 4
```

`_on_exit` を置き換える:

```python
    def _on_exit(self, pipeline_id: str, code: int) -> None:
        runner = self.runners.get(pipeline_id)
        if runner is None:
            return  # deleted while its process was terminating — nothing to show
        message = f"process exited: {code}"
        log = self.logs.setdefault(pipeline_id, deque(maxlen=LOG_BUFFER_MAX))
        log.append(message)
        # 意図的な停止 (runner.stopping) は即死とみなさない。terminate の
        # exit code は非 0 なので、これが無いと Stop 直後に誤って失敗バナーが出る。
        quick = (
            code != 0 and not runner.stopping and runner.ran_for() < QUICK_EXIT_SEC
        )
        if self.editor.entry is not None and self.editor.entry.id == pipeline_id:
            self.editor.append_log(message)
            self.editor.set_running(False)
            if quick:
                self.editor.show_launch_failure(list(log)[-FAILURE_TAIL_LINES:])
        self._refresh_list()
```

- [ ] **Step 4: 手動 smoke 検証**

Run: `uv run python -m gui`

1. 「マイク→ボイチェン→再生」を作り、`✗` が出ている状態で「未充足でも起動」を押す。
2. vspeech が `起動中止: 設定不備 N 件` を吐いて exit 1 する。
3. **赤いバナーにその理由が出る**（ログを読まずに分かる）ことを確認する。
4. 「マイク→再生 (モニター)」を Start し、**数秒以内に Stop** する。terminate の
   exit code は非 0 だが、**失敗バナーは出ない**（意図的停止を即死と誤認しない）ことを確認する。

Expected: 上記 4 点。

- [ ] **Step 5: 型検査・整形・全テスト・コミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: すべて exit 0

```bash
git add gui/process.py gui/app.py gui/pipeline_editor.py
git commit -F - <<'EOF'
feat(gui): 起動直後に落ちた理由をバナーに出す

process exited: N で終わらせず、起動から QUICK_EXIT_SEC 未満の異常終了を
起動失敗とみなしてログ末尾の理由をバナーへ出す。readiness が config 不備を
起動前に止めるので、ここに来るのは主に層B (実ロード/GPU/外部接続) の失敗。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: 問題フィールドの ✗ 印と資産フィールドの前出し

spec の「いま有効な worker の必須項目が、任意設定より前面に表示される」を満たす。**どのフィールドが必須かの判断は持たない** — readiness が出した `field` を印として使うだけ。並び順は純粋な体裁（資産・資格 → 調整つまみ）。

**Files:**
- Modify: `gui/form.py`
- Modify: `gui/pipeline_editor.py`

**Interfaces:**
- Consumes: `gui.readiness.Readiness`（Task 2）
- Produces: `gui.form.PipelineForm.mark_problems(fields: set[str]) -> None`

- [ ] **Step 1: 問題フィールドに印を付ける**

`gui/form.py` の `PipelineForm.__init__` の `self.bindings` 初期化の直後に足す:

```python
        # path -> その欄の見出し Label。readiness の ✗ 印を付けるために持つ。
        self.labels: dict[str, Any] = {}
```

`_entry` / `_spin` / `_device_combo` の `Label(frame, text=label)` を、label を捕まえる形へ置き換える。`_entry`:

```python
    def _entry(self, parent: Any, path: str, label: str) -> Frame:
        assert self.config is not None  # nosec B101
        frame = Frame(parent)
        self.labels[path] = Label(frame, text=label)
        self.labels[path].pack(fill=X, pady=(6, 0))
        widget = Textbox(frame)
        widget.set(_get(self.config, path))
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, str)
        widget.pack(fill=X)
        return frame
```

`_spin`:

```python
    def _spin(
        self, parent: Any, path: str, label: str, from_: float, to: float, inc: float
    ) -> Frame:
        assert self.config is not None  # nosec B101
        frame = Frame(parent)
        self.labels[path] = Label(frame, text=label)
        self.labels[path].pack(fill=X, pady=(6, 0))
        widget = Spinbox(frame, from_=from_, to=to, increment=inc, wrap=True)
        widget.set(_get(self.config, path))
        coerce = int if float(inc).is_integer() else float
        widget.configure(command=self.on_change)
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, coerce)
        widget.pack(fill=X)
        return frame
```

`_device_combo` の `Label(frame, text=label).pack(fill=X, pady=(6, 0))` を置き換える:

```python
        self.labels[path] = Label(frame, text=label)
        self.labels[path].pack(fill=X, pady=(6, 0))
```

`bind_config` の `self.bindings.clear()` の直後に足す:

```python
        self.labels.clear()
```

`focus_field` の直前に足す:

```python
    def mark_problems(self, fields: set[str]) -> None:
        """readiness が問題ありとした欄の見出しに ✗ を付ける。

        「必須か」の判断はしない — 渡された path をそのまま印付けるだけ
        (権威は vspeech.preflight, ADR-0045)。
        """
        for path, label in self.labels.items():
            if not label.winfo_exists():
                continue
            text = label.cget("text").removeprefix("✗ ")
            if path in fields:
                label.configure(text=f"✗ {text}", bootstyle="danger")
            else:
                label.configure(text=text, bootstyle="default")
```

- [ ] **Step 2: 資産・資格フィールドを調整つまみより前に出す**

`gui/form.py` の `_section_vc` を置き換える（資産パス 3 本が先、`f0_up_key` / `gpu_id` は後ろ。中身は同じで順序だけ）:

```python
    def _section_vc(self) -> None:
        box = self._section_box("vc")
        enable = self._check(box, "vc.enable", "enable vc")
        enable.pack(anchor=W)
        content = Frame(box)
        self._register_section(box, enable, "vc", content)
        # 資産パス (無いと起動しない) を先に、調整つまみを後ろに置く。
        self._entry(content, "rvc.model_file", "rvc model_file").pack(fill=X)
        self._entry(content, "rvc.hubert_model_file", "hubert asset dir").pack(fill=X)
        self._entry(content, "rvc.rmvpe_model_file", "rvc rmvpe_model_file").pack(
            fill=X
        )
        self._spin(content, "rvc.f0_up_key", "f0_up_key", -64, 64, 1).pack(fill=X)
        self._spin(content, "rvc.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
```

`_section_recording` を置き換える（デバイスが先。既存もそうなので `rate` と `silence_threshold` の順は変えない）:

```python
    def _section_recording(self) -> None:
        box = self._section_box("recording")
        enable = self._check(box, "recording.enable", "enable recording")
        enable.pack(anchor=W)
        content = Frame(box)
        self._register_section(box, enable, "recording", content)
        # デバイス (無い/違うと無音になる) を先に、調整つまみを後ろに置く。
        self._device_combo(
            content, "recording.input_device_index", "input device", input=True
        ).pack(fill=X)
        self._spin(content, "recording.rate", "rate", 8000, 48000, 1).pack(fill=X)
        self._spin(
            content,
            "recording.silence_threshold",
            "silence threshold (dBFS)",
            -120,
            0,
            1,
        ).pack(fill=X)
```

**注**: `_section_vc` の `rvc.rmvpe_model_file` はもともとフォームに在ったが、`f0_extractor_type` の既定が `rmvpe` なので readiness が必ず要求する。ラベルを `rvc rmvpe_model_file` に揃えて path と対応させる。

- [ ] **Step 3: エディタから印を配線する**

`gui/pipeline_editor.py` の `refresh_readiness` の `self.panel.show(self.readiness)` の直後に足す:

```python
            self.form.mark_problems(
                {
                    problem.field
                    for worker in self.readiness.workers
                    for problem in worker.problems
                    if problem.field is not None
                }
            )
```

- [ ] **Step 4: 手動 smoke 検証**

Run: `uv run python -m gui`

1. 「マイク→ボイチェン→再生」を選ぶ。Form タブの `rvc model_file` / `hubert asset dir` / `rvc rmvpe_model_file` の見出しに **✗ が付き赤くなる**。
2. vc セクションで資産パス 3 本が `f0_up_key` / `gpu_id` より上にある。
3. 実在するパスを入れて Save すると、その欄の ✗ が消える。

Expected: 上記 3 点。

- [ ] **Step 5: 型検査・整形・全テスト・コミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: すべて exit 0

```bash
git add gui/form.py gui/pipeline_editor.py
git commit -F - <<'EOF'
feat(gui): 問題フィールドに ✗ を付け資産パスを調整つまみより前に出す

readiness が出した field をそのまま印にするだけで、必須かの判断は持たない
(権威は preflight, ADR-0045)。並び順は体裁として資産/資格 → 調整つまみへ。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 7: 共有素材パスの編集と propagate

マシン共通の素材パスを `default.toml` で一度編集し、明示操作で既存の全 pipeline へ反映する（ADR-0046）。pipeline config は自己完結のまま。

**Files:**
- Create: `gui/config_paths.py`
- Create: `gui/shared_paths.py`
- Modify: `gui/form.py`（private ヘルパを `config_paths` へ移す）
- Modify: `gui/app.py`
- Create: `gui/shared_dialog.py`
- Test: `tests/test_gui_shared_paths.py`

**Interfaces:**
- Consumes: `gui.profile.load_pipeline_config` / `save_pipeline_config` / `Profile` / `PipelineEntry`, `gui.paths.ProfilePaths`
- Produces:
  - `gui.config_paths.get_value(config: Config, path: str) -> Any`
  - `gui.config_paths.set_value(config: Config, path: str, value: Any) -> None`
  - `gui.config_paths.field_types(config: Config, path: str) -> tuple[Any, ...]`
  - `gui.shared_paths.SHARED_ASSET_FIELDS: tuple[str, ...]`
  - `gui.shared_paths.shared_values(config: Config) -> dict[str, Any]`
  - `gui.shared_paths.apply_shared(source: Config, target: Config) -> list[str]`
  - `gui.shared_dialog.SharedPathsDialog(master, paths: ProfilePaths, default_config: Config)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gui_shared_paths.py` を新規作成:

```python
from pathlib import Path

from gui.shared_paths import SHARED_ASSET_FIELDS
from gui.shared_paths import apply_shared
from gui.shared_paths import shared_values
from vspeech.config import Config
from vspeech.config import RvcConfig


def test_shared_values_reads_every_shared_field():
    config = Config(rvc=RvcConfig(model_file=Path("m.pth")))
    values = shared_values(config)
    assert set(values) == set(SHARED_ASSET_FIELDS)
    assert values["rvc.model_file"] == Path("m.pth")


def test_apply_shared_copies_and_reports_changed_fields():
    source = Config(rvc=RvcConfig(model_file=Path("new.pth")))
    target = Config(rvc=RvcConfig(model_file=Path("old.pth")))
    changed = apply_shared(source, target)
    assert changed == ["rvc.model_file"]
    assert target.rvc.model_file == Path("new.pth")


def test_apply_shared_is_noop_when_already_equal():
    source = Config(rvc=RvcConfig(model_file=Path("same.pth")))
    target = Config(rvc=RvcConfig(model_file=Path("same.pth")))
    assert apply_shared(source, target) == []


def test_apply_shared_does_not_touch_non_shared_fields():
    source = Config(rvc=RvcConfig(model_file=Path("m.pth"), f0_up_key=12))
    target = Config(rvc=RvcConfig(model_file=Path("old.pth"), f0_up_key=-3))
    apply_shared(source, target)
    assert target.rvc.f0_up_key == -3  # 調整つまみは pipeline 固有なので保つ
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_gui_shared_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.shared_paths'`

- [ ] **Step 3: ドット path ヘルパを `gui/config_paths.py` へ抽出する**

`gui/config_paths.py` を新規作成（中身は `gui/form.py` の `_get` / `_coerce_value` / `_field_types` / `_set` をそのまま移したもの）:

```python
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
```

`gui/form.py` から `_get` / `_coerce_value` / `_field_types` / `_set` の 4 関数と、それだけが使っていた import（`from pathlib import Path` / `from typing import get_args` / `from pydantic import SecretStr` のうち他で使われていないもの）を削除し、以下を import に足す（1 行 1 つ）:

```python
from gui.config_paths import field_types as _field_types
from gui.config_paths import get_value as _get
from gui.config_paths import set_value as _set
```

**注**: `form.py` の `read_into` は `SecretStr` を型判定にまだ使っているので、`from pydantic import SecretStr` は残す。

- [ ] **Step 4: `gui/shared_paths.py` を実装する**

```python
"""マシン共通の素材パス (ADR-0046)。

マシンに 1 セットしかない重い資産のパス。default.toml で一度編集し、明示
操作で既存の全 pipeline へ propagate する。pipeline config は自己完結のまま
なので (ADR-0032)、値は各 pipeline へ実際に書き込む — 起動時合成はしない。
"""

from typing import Any

from gui.config_paths import get_value
from gui.config_paths import set_value
from vspeech.config import Config

SHARED_ASSET_FIELDS: tuple[str, ...] = (
    "rvc.model_file",
    "rvc.hubert_model_file",
    "rvc.rmvpe_model_file",
    "voicevox.openjtalk_dir",
    "voicevox.model_dir",
    "voicevox.onnxruntime_path",
)


def shared_values(config: Config) -> dict[str, Any]:
    return {path: get_value(config, path) for path in SHARED_ASSET_FIELDS}


def apply_shared(source: Config, target: Config) -> list[str]:
    """`source` の共有素材パスを `target` へ写し、実際に変わった path を返す。

    共有指定のフィールドだけ触る。pipeline 固有の調整つまみは保つ。
    """
    changed: list[str] = []
    for path in SHARED_ASSET_FIELDS:
        value = get_value(source, path)
        if get_value(target, path) == value:
            continue
        set_value(target, path, value)
        changed.append(path)
    return changed
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `uv run pytest tests/test_gui_shared_paths.py -v`
Expected: PASS（4 件）

Run: `uv run pytest tests/ -v`
Expected: PASS（`form.py` のヘルパ移動で既存が壊れていないこと）

- [ ] **Step 6: 共有パス編集ダイアログを作る**

`gui/shared_dialog.py` を新規作成:

```python
from pathlib import Path
from tkinter import W
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Toplevel

from gui.config_paths import get_value
from gui.config_paths import set_value
from gui.paths import ProfilePaths
from gui.profile import save_default_config
from gui.shared_paths import SHARED_ASSET_FIELDS
from gui.widgets import Textbox
from vspeech.config import Config


class SharedPathsDialog(Toplevel):
    """マシン共通の素材パスを default.toml 上で一度だけ編集する (ADR-0046)。

    `propagate` が True で閉じたら、呼び出し側が既存の全 pipeline へ反映する。
    """

    def __init__(self, master: Any, paths: ProfilePaths, default_config: Config):
        super().__init__(master)
        self.title("共有 (既定) の素材パス")
        self.transient(master)
        self.paths = paths
        self.default_config = default_config
        self.propagate = False
        self.saved = False

        Label(
            self,
            text="マシンに 1 セットしかない資産のパス。ここで編集すると新規 pipeline が継承します。",
        ).pack(anchor=W, padx=12, pady=(12, 6))

        self.entries: dict[str, Textbox] = {}
        for path in SHARED_ASSET_FIELDS:
            row = Frame(self)
            row.pack(fill="x", padx=12, pady=2)
            Label(row, text=path, width=28).pack(side="left", anchor=W)
            entry = Textbox(row)
            entry.set(get_value(default_config, path))
            entry.pack(side="left", fill="x", expand=True)
            self.entries[path] = entry

        buttons = Frame(self)
        buttons.pack(anchor="e", padx=12, pady=12)
        Button(
            buttons, text="保存", command=self._save, bootstyle="primary"
        ).pack(side="left", padx=4)
        Button(
            buttons,
            text="保存して全 pipeline へ反映",
            command=self._save_and_propagate,
            bootstyle="warning",
        ).pack(side="left", padx=4)
        Button(
            buttons, text="Cancel", command=self.destroy, bootstyle="secondary"
        ).pack(side="left", padx=4)

        self.grab_set()
        self.wait_window()

    def _write_back(self) -> None:
        for path, entry in self.entries.items():
            text = entry.get_value().strip()
            # 空欄は「未設定」。Optional (voicevox.onnxruntime_path) は None、
            # それ以外の Path は Path() 番兵に戻す — preflight がそれを不在と
            # して報告する。
            if not text:
                set_value(
                    self.default_config,
                    path,
                    None if path == "voicevox.onnxruntime_path" else Path(),
                )
            else:
                set_value(self.default_config, path, text)
        save_default_config(self.paths, self.default_config)
        self.saved = True

    def _save(self) -> None:
        self._write_back()
        self.destroy()

    def _save_and_propagate(self) -> None:
        self._write_back()
        self.propagate = True
        self.destroy()
```

- [ ] **Step 7: app に導線と propagate を足す**

`gui/app.py` の import に足す（1 行 1 つ）:

```python
from gui.shared_dialog import SharedPathsDialog
from gui.shared_paths import apply_shared
```

`__init__` の `Button(left, text="del", ...)` の直後に足す:

```python
        Button(left, text="共有パス", command=self.edit_shared_paths).pack(fill="x")
```

`new_pipeline` の直前に足す:

```python
    def edit_shared_paths(self) -> None:
        dialog = SharedPathsDialog(self, self.paths, self.default_config)
        if not dialog.saved:
            return
        if dialog.propagate:
            self._propagate_shared_paths()
        # 編集中の pipeline は propagate で config が変わりうるので読み直す。
        selection = self.listbox.curselection()
        if selection:
            self.editor.load_entry(self.profile.pipelines[selection[0]])
            self.editor.refresh_readiness()

    def _propagate_shared_paths(self) -> None:
        """共有素材パスを既存の全 pipeline へ書き込む (ADR-0046)。

        pipeline config は自己完結を保つので、値は実際に各ファイルへ書く。
        壊れて読めない pipeline は飛ばして名指しで報告する — 黙って捨てない。
        """
        skipped: list[str] = []
        for entry in self.profile.pipelines:
            result = load_pipeline_config(self.paths, entry)
            if not result.ok or result.value is None:
                skipped.append(entry.name)
                continue
            changed = apply_shared(self.default_config, result.value)
            if changed:
                save_pipeline_config(self.paths, entry, result.value)
                logger.info(
                    "propagated %s to pipeline %s", ", ".join(changed), entry.id
                )
        save_profile(self.paths, self.profile)
        if skipped:
            self.editor.banner.configure(
                text=f"⚠ 読めない pipeline へは反映できませんでした: {', '.join(skipped)}"
            )
```

`gui/app.py` の import に足す（`load_pipeline_config` はまだ import されていない）:

```python
from gui.profile import load_pipeline_config
```

- [ ] **Step 8: 手動 smoke 検証**

Run: `uv run python -m gui`

1. 「共有パス」を押すと 6 フィールドが `default.toml` の現在値で出る。
2. `rvc.model_file` に実在するパスを入れて「保存して全 pipeline へ反映」。
3. 既存の vc pipeline を選ぶと、その欄の ✗ が消えて readiness が 1 件減っている。
4. `~/.config/vstreamer/pipelines/<id>.toml` を開き、値が**実際に書かれている**ことを確認する（自己完結、ADR-0046）。
5. `uv run python -m vspeech --config <その toml>` が GUI 抜きで起動する。

Expected: 上記 5 点。

- [ ] **Step 9: 型検査・整形・全テスト・コミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: すべて exit 0

```bash
git add gui/config_paths.py gui/shared_paths.py gui/shared_dialog.py gui/form.py gui/app.py tests/test_gui_shared_paths.py
git commit -F - <<'EOF'
feat(gui): 共有素材パスを default.toml で編集し明示 propagate する (ADR-0046)

マシンに 1 セットしかない資産パスを一度だけ編集し、明示操作で既存の全
pipeline へ書き込む。pipeline config は自己完結のまま (ADR-0032) なので
起動時合成はせず実際に書く。読めない pipeline は名指しで報告して飛ばす。
ドット path ヘルパは form.py から config_paths.py へ抽出して共用。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 8: ドキュメント整合と ADR の昇格

**Files:**
- Modify: `docs/adr/0045-gui-readiness-reuses-preflight.md:3`
- Modify: `docs/adr/0046-gui-shared-asset-paths-explicit-propagate.md:3`
- Modify: `docs/adr/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: spec の受入基準を 1 つずつ実物で確認する**

`docs/superpowers/specs/2026-07-16-gui-run-readiness-design.md` の 8 項目を、起動中の GUI で 1 つずつ確認してチェックを付ける。**動かして確かめる**（テストが通ることは確認にならない — 過去に「テストは通るのにエントリポイントが落ちる」を 2 度出している）。

Run: `uv run python -m gui`

- [ ] **Step 2: ADR を Accepted へ昇格する**

実装が両 ADR の決定を裏づけたので、各ファイルの `- Status: Proposed` を書き換える（本文は触らない）:

```markdown
- Status: Accepted
```

`docs/adr/README.md` の 0045 / 0046 の行の `Proposed` を `Accepted` にする。

**実装が決定を覆していた場合は昇格させず、supersede する新 ADR を起こすこと。**

- [ ] **Step 3: CLAUDE.md に readiness の不変条件を書く**

`CLAUDE.md` の Architecture 節、`### SharedContext` の直前に足す:

```markdown
### GUI の起動前 readiness
GUI は「この pipeline が起動できるか」を起動前に見せるが、**「何が必須か」の判断は持たない** — [`vspeech/preflight.py`](vspeech/preflight.py) の `collect_problems` を呼ぶだけで、そこが唯一の権威（[ADR-0045](docs/adr/0045-gui-readiness-reuses-preflight.md)）。必須リソースを増やすときは preflight だけを直せば GUI は自動で追随する。GUI 側に必須項目の一覧を複製しないこと — 複製すると「GUI は緑なのに起動して落ちる」形で drift する。`ConfigProblem.field` は GUI がその設定箇所へ飛ぶための公開契約。マシン共通の素材パスは `default.toml` で編集して明示 propagate し、pipeline config は単体で起動できる自己完結を保つ（[ADR-0046](docs/adr/0046-gui-shared-asset-paths-explicit-propagate.md)、[ADR-0032](docs/adr/0032-gui-multi-pipeline-rewrite.md)）。
```

- [ ] **Step 4: 最終確認とコミット**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: すべて exit 0

```bash
git add docs/adr/ CLAUDE.md
git commit -F - <<'EOF'
docs: ADR-0045/0046 を Accepted へ昇格し readiness の不変条件を CLAUDE.md へ

実装が両決定を裏づけたので Proposed から昇格 (Status 行のみ)。GUI に必須項目を
複製しない不変条件を CLAUDE.md に明記する。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

## Self-Review

**1. Spec coverage** — spec の受入基準 8 項目と task の対応:

| 受入基準 | Task |
|---|---|
| 有効 worker ごとに必要項目と充足状況が起動前に表示される | 2, 3, 4 |
| 生 TOML 手編集にも一覧が追随する | 2（`enabled_workers` は実 enable フラグ駆動、`collect_problems` も同様） |
| 未充足項目からワンアクションで設定箇所へ移動しフォーカス | 1（`field`）, 4（`focus_field` / `_focus_field`） |
| 未充足なら Start 無効化＋件数/理由、明示操作の起動口は残る | 4 |
| 起動直後の異常終了理由が生ログを開かずに分かる | 5 |
| マシン共通の素材パスを一箇所で編集し全 pipeline へ反映 | 7 |
| 反映後も pipeline 設定ファイル単体で起動できる | 7（Step 8-5 で実起動を確認） |
| 有効 worker の必須項目が任意設定より前面 | 6 |

漏れなし。

**2. Placeholder scan** — 「TBD」「後で」「適切に」「Task N と同様」なし。コードを変える全 step に実コードあり。

**3. Type consistency** — `collect_problems(config) -> list[ConfigProblem]`（Task 1 定義 / Task 2 使用）、`ConfigProblem.field: str | None`（Task 1 定義 / Task 3・4・6 使用）、`evaluate(config) -> Readiness`（Task 2 定義 / Task 4 使用）、`Readiness.workers: list[WorkerReadiness]` と `.problem_count`（Task 2 定義 / Task 3・4・6 使用）、`ReadinessPanel(master, on_fix)` / `.show()` / `.clear()`（Task 3 定義 / Task 4 使用）、`PipelineForm.focus_field(path) -> bool`（Task 4 定義 / Task 4 使用）、`mark_problems(fields: set[str])`（Task 6 定義 / Task 6 使用）、`apply_shared(source, target) -> list[str]`（Task 7 定義 / Task 7 使用）。名前・シグネチャは task 間で一致。
