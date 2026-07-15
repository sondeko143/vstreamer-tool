# Worker 設定 preflight (fail-loud) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** enable した worker の致命的な設定不備を起動時 preflight で集約検出してプロセス全体を明確な理由付きで停止し、機能しない設定を凌ぐ防御コードを減らす。

**Architecture:** 2 層。層A = TaskGroup を開く前に `preflight(config)` が全 enable worker の安価な設定検査（必須フィールド・ファイル/ディレクトリ存在・デバイス発見可否・依存の有無）を走らせ、検出した全問題を 1 つの `ConfigError` に集約して送出、`cmd()` が traceback 無しの整形ログで停止。層B = worker 起動時の実リソース取得失敗（モデルロード・GPU・ストリーム open）を `WorkerStartupError` に変換し、`vspeech_coro` の `except* WorkerStartupError` が worker と原因を名指しして停止。

**Tech Stack:** Python 3.14, pydantic v2, asyncio TaskGroup / `except*`, sounddevice, onnxruntime, pytest (`asyncio_mode="auto"`)。

## Global Constraints

- Python **3.14 のみ**（`>=3.14,<3.15`）。`TaskGroup` / `except*`（3.11+）、PEP 695 `type X =`（3.12+）を使用。floor を下げない。
- import は **1 行 1 つ**（ruff `force-single-line=true`、保存時自動ソート）。
- 型チェックは **ty**（`uv run ty check`、whole-project）。ruff format/lint（`uv run ruff format .` / `uv run ruff check .`）。
- pydantic は **v2 API のみ**（`Field(...)`, `model_validator(mode="after")`）。v1 API 禁止。
- テストは `tests/`、`pytest`（`asyncio_mode="auto"` なので async テストにマーカ不要）。実ハードウェア・ネットワーク・GPU に依存しないこと（受入基準）。
- 秘密・環境 PII をコミットに含めない（gitleaks pre-commit がステージ差分を走査）。`C:\Users\<name>` パス等は使わない。
- 変更後は `uv run poe check`（ruff+ty+pytest 等）を緑にする。既存の受理済み 2 件（torch CVE / deadcode vr2_config）以外の新規指摘を出さない。
- ブランチ: `feat/worker-config-preflight`（作成済み）。decision は [ADR-0038](../../adr/0038-worker-config-preflight-fail-loud.md)、spec は [2026-07-16-worker-config-preflight-design.md](../specs/2026-07-16-worker-config-preflight-design.md)。

---

## File Structure

- **Create** `vspeech/preflight.py` — 層A の集中 preflight。`preflight(config)` ランナー + 各 `_check_<worker>(config)` チェッカ + 共有ヘルパ（`_check_gcp_credentials` / `_check_vad_gate`）。
- **Modify** `vspeech/exceptions.py` — `ConfigProblem` / `ConfigError` / `WorkerStartupError` / `DeviceNotFoundError` と `worker_startup` コンテキストマネージャを追加。
- **Modify** `vspeech/main.py` — `cmd()` で preflight 呼び出し + `ConfigError` 整形ログ、`vspeech_coro` に `except* WorkerStartupError`。
- **Modify** `vspeech/lib/audio.py` — 共有 `resolve_input_device` / `resolve_output_device`。
- **Modify** `vspeech/worker/recording.py` / `playback.py` — 独自デバイスガードを resolver 呼び出しへ置換。
- **Modify** `vspeech/worker/transcription.py` / `tts.py` / `translation.py` / `vc.py` — 層B の `worker_startup` で setup を包む。Enum `else` 除去（transcription/tts）。recording_log の DEGRADE 化（transcription）。
- **Modify** `vspeech/config.py` — `RecordingConfig` の `rate`/`channels`/`chunk` に `gt=0`。
- **Create** `tests/test_preflight.py` — 層A の HW 非依存テスト。
- **Modify** `tests/test_recording_metrics.py` 系 — `denom` ガード除去に伴う既存テスト確認（新規は test_preflight）。

---

## Task 1: preflight の骨格 + transcription チェッカ + main 配線

層Aの縦切り一枚。例外型・ランナー・transcription チェッカ（ACP 4 必須 / GCP key.json 存在 / mozc 依存 / VAD モデル存在）・`main.py` 配線までを通し、集約と整形ログを成立させる。

**Files:**
- Modify: `vspeech/exceptions.py`
- Create: `vspeech/preflight.py`
- Modify: `vspeech/main.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Produces:
  - `ConfigProblem(worker: str, detail: str)` — frozen dataclass、`str()` は `"[worker] detail"`。
  - `ConfigError(problems: list[ConfigProblem])` — `.problems` 属性を持つ `Exception`。
  - `WorkerStartupError(worker: str, detail: str)` — `.worker` / `.detail` を持つ `Exception`（Task 5 で使用）。
  - `preflight(config: Config) -> None` — 問題があれば `ConfigError` を raise、無ければ何もしない。
  - `_check_gcp_credentials(gcp: GcpConfig, worker: str) -> list[ConfigProblem]` / `_check_vad_gate(cfg, worker: str) -> list[ConfigProblem]` — Task 3 が再利用。

- [ ] **Step 1: Write the failing test**

`tests/test_preflight.py` を新規作成:

```python
from pathlib import Path

import pytest

from vspeech.config import AmiConfig
from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.exceptions import ConfigError
from vspeech.preflight import preflight


def _acp(**ami_kw):
    return Config(
        transcription=TranscriptionConfig(
            enable=True, worker_type=TranscriptionWorkerType.ACP
        ),
        ami=AmiConfig(**ami_kw),
    )


def _full_acp():
    return _acp(
        appkey="k", engine_uri="https://e", engine_name="g", service_id="s"
    )


def test_disabled_worker_is_not_checked():
    # transcription 無効なら ami 空でも問題なし
    preflight(Config())


def test_acp_missing_fields_are_all_reported():
    with pytest.raises(ConfigError) as ei:
        preflight(_acp())  # 4 フィールドすべて空
    details = [p.detail for p in ei.value.problems]
    assert any("ami.appkey" in d for d in details)
    assert any("ami.engine_uri" in d for d in details)
    assert any("ami.engine_name" in d for d in details)
    assert any("ami.service_id" in d for d in details)
    assert all(p.worker == "transcription" for p in ei.value.problems)


def test_acp_complete_config_passes():
    preflight(_full_acp())


def test_gcp_missing_key_file_is_reported():
    cfg = Config(
        transcription=TranscriptionConfig(
            enable=True, worker_type=TranscriptionWorkerType.GCP
        ),
        gcp=GcpConfig(service_account_file_path=Path("/no/such/key.json")),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("service_account_file_path" in p.detail for p in ei.value.problems)


def test_vad_gate_missing_model_is_reported():
    cfg = Config(
        transcription=TranscriptionConfig(
            enable=True,
            worker_type=TranscriptionWorkerType.ACP,
            vad_gate=True,
            vad_model_file=Path("/no/such/silero_vad.onnx"),
        ),
        ami=AmiConfig(
            appkey="k", engine_uri="https://e", engine_name="g", service_id="s"
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("vad_model_file" in p.detail for p in ei.value.problems)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preflight.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'vspeech.preflight'` / `ImportError: cannot import name 'ConfigError'`）。

- [ ] **Step 3: exceptions.py に型を追加**

`vspeech/exceptions.py` の末尾（`shutdown_worker` の後）に追記:

```python
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigProblem:
    worker: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.worker}] {self.detail}"


class ConfigError(Exception):
    """preflight が集約した致命的な設定不備（タスク spawn 前に送出）。"""

    def __init__(self, problems: list["ConfigProblem"]):
        self.problems = problems
        super().__init__("; ".join(str(p) for p in problems))


class WorkerStartupError(Exception):
    """worker が起動時に実リソースを取得できなかった（層B の深層失敗）。"""

    def __init__(self, worker: str, detail: str):
        self.worker = worker
        self.detail = detail
        super().__init__(f"[{worker}] {detail}")
```

（`from contextlib import contextmanager` は Task 5 の `worker_startup` で使う。ここで足しておく。import はファイル先頭へ移すのが規約なので、既存 import 群のアルファベット順位置へ置く: `from asyncio import current_task` の前に `from contextlib import contextmanager` と `from dataclasses import dataclass` を single-line で。ruff が並べ替える。）

- [ ] **Step 4: preflight.py を作成**

`vspeech/preflight.py`:

```python
"""起動時の設定 preflight (層A, ADR-0038)。

enable した各 worker の「実リソースを取得せずに安価に判定できる」設定不備
(必須フィールド・参照ファイル/ディレクトリの存在・デバイス発見可否・依存の
有無) を検査する。検出した全問題を集約して ConfigError で送出する。実ロードで
しか分からない失敗は worker 起動時に扱う (層B)。
"""

from collections.abc import Callable
from importlib.util import find_spec

from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.config import VcConfig
from vspeech.exceptions import ConfigProblem
from vspeech.exceptions import ConfigError

Checker = Callable[[Config], list[ConfigProblem]]


def _check_gcp_credentials(gcp: GcpConfig, worker: str) -> list[ConfigProblem]:
    # 認証の実成立は層B。ここでは指定した key.json の存在だけ安価に見る。
    if gcp.service_account_file_path is not None:
        path = gcp.service_account_file_path.expanduser()
        if not path.is_file():
            return [
                ConfigProblem(
                    worker,
                    f"gcp.service_account_file_path '{path}' が存在しません",
                )
            ]
    return []


def _check_vad_gate(
    cfg: TranscriptionConfig | VcConfig, worker: str
) -> list[ConfigProblem]:
    if not cfg.vad_gate:
        return []
    path = cfg.vad_model_file.expanduser()
    if not path.is_file():
        return [
            ConfigProblem(
                worker,
                f"vad_gate=true ですが vad_model_file '{path}' が存在しません",
            )
        ]
    return []


def _check_transcription(config: Config) -> list[ConfigProblem]:
    if not config.transcription.enable:
        return []
    w = "transcription"
    tc = config.transcription
    problems: list[ConfigProblem] = []
    if tc.worker_type == TranscriptionWorkerType.ACP:
        ami = config.ami
        required = (
            ("ami.appkey", ami.appkey.get_secret_value()),
            ("ami.engine_uri", ami.engine_uri),
            ("ami.engine_name", ami.engine_name),
            ("ami.service_id", ami.service_id),
        )
        for name, value in required:
            if not value:
                problems.append(
                    ConfigProblem(w, f"ACP バックエンドには {name} が必須ですが空です")
                )
        if tc.transliterate_with_mozc and find_spec("mozcpy") is None:
            problems.append(
                ConfigProblem(
                    w,
                    "transliterate_with_mozc=true ですが mozcpy が未インストールです",
                )
            )
    elif tc.worker_type == TranscriptionWorkerType.GCP:
        problems.extend(_check_gcp_credentials(config.gcp, w))
    # WHISPER のモデル/GPU ロードは層B（起動時取得）。
    problems.extend(_check_vad_gate(tc, w))
    return problems


_CHECKERS: list[Checker] = [
    _check_transcription,
]


def preflight(config: Config) -> None:
    problems: list[ConfigProblem] = []
    for checker in _CHECKERS:
        problems.extend(checker(config))
    if problems:
        raise ConfigError(problems)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_preflight.py -q`
Expected: PASS（6 tests）。

- [ ] **Step 6: main.py に配線**

`vspeech/main.py` の `cmd()` を編集。`telemetry.configure(...)` の直後、`loop = new_event_loop()` の**前**に preflight を挿入:

```python
    configure_logger(config)
    telemetry.configure(
        enabled=config.telemetry.enable,
        max_samples=config.telemetry.max_samples,
        jsonl_path=config.telemetry.jsonl_path,
    )
    try:
        preflight(config)
    except ConfigError as e:
        logger.error("起動中止: 設定不備 %d 件", len(e.problems))
        for problem in e.problems:
            logger.error("  %s", problem)
        exit(1)
    # 3.14 で get_event_loop() は running loop が無いと RuntimeError を投げる
    loop = new_event_loop()
    ...
```

ファイル先頭の import に追加（single-line, ruff がソート）:

```python
from vspeech.exceptions import ConfigError
from vspeech.preflight import preflight
```

そして `vspeech_coro` の TaskGroup に `except* WorkerStartupError` を追加（`except* WorkerShutdown` の前）:

```python
    except* WorkerStartupError as eg:
        for e in eg.exceptions:
            logger.error("worker startup failed: %s", e)
    except* WorkerShutdown as eg:
        for e in eg.exceptions:
            logger.warning("workers shutdown: %s", e.args)
            logger.debug("".join(format_exception(e)))
    finally:
        telemetry.log_summary()
```

import に追加:

```python
from vspeech.exceptions import WorkerStartupError
```

- [ ] **Step 7: main の回帰テストを足す**

`tests/test_preflight.py` に追記（`cmd()` が ConfigError で SystemExit すること）:

```python
def test_cmd_exits_on_config_error(monkeypatch):
    import asyncio

    from vspeech.main import cmd
    from vspeech.exceptions import ConfigError
    from vspeech.exceptions import ConfigProblem

    def _boom(config):
        raise ConfigError([ConfigProblem("transcription", "boom")])

    monkeypatch.setattr("vspeech.main.preflight", _boom)
    monkeypatch.setattr("vspeech.main.configure_logger", lambda config: None)
    monkeypatch.setattr("vspeech.main.telemetry.configure", lambda **kw: None)
    asyncio.set_event_loop(None)
    assert cmd.callback is not None
    with pytest.raises(SystemExit) as ei:
        cmd.callback(config_file=None)
    assert ei.value.code == 1
```

- [ ] **Step 8: Run full new tests + lint/type**

Run: `uv run pytest tests/test_preflight.py -q && uv run ruff format vspeech/preflight.py vspeech/exceptions.py vspeech/main.py && uv run ruff check vspeech/preflight.py vspeech/exceptions.py vspeech/main.py && uv run ty check`
Expected: pytest PASS（7 tests）、ruff clean、ty exit 0。

- [ ] **Step 9: Commit**

```bash
git add vspeech/preflight.py vspeech/exceptions.py vspeech/main.py tests/test_preflight.py
git commit -m "feat(preflight): aggregate transcription config problems + fail-loud wiring"
```

---

## Task 2: 共有デバイス resolver + recording/playback ガード削除 + デバイス preflight

デバイス名→index 解決を `lib/audio.py` の共有 resolver へ一本化し、preflight と worker が同一経路を通る。worker 側の独自「未発見」ガードを削除する（ゴール2）。

**Files:**
- Modify: `vspeech/exceptions.py`（`DeviceNotFoundError`）
- Modify: `vspeech/lib/audio.py`（`resolve_input_device` / `resolve_output_device`）
- Modify: `vspeech/worker/recording.py`（`open_input_stream`）
- Modify: `vspeech/worker/playback.py`（`get_output_device`）
- Modify: `vspeech/preflight.py`（`_check_recording` / `_check_playback`）
- Test: `tests/test_preflight.py`, `tests/test_device_resolver.py`

**Interfaces:**
- Consumes: `ConfigProblem`, `preflight` の `_CHECKERS` 登録。
- Produces:
  - `DeviceNotFoundError(Exception)` — resolver が投げる。
  - `resolve_input_device(config: RecordingConfig) -> DeviceInfo`
  - `resolve_output_device(config: PlaybackConfig) -> DeviceInfo`

- [ ] **Step 1: Write the failing test**

`tests/test_device_resolver.py` を新規作成:

```python
import pytest

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.exceptions import DeviceNotFoundError
from vspeech.lib.audio import DeviceInfo


def _device(index: int = 3):
    return DeviceInfo(
        host_api=0,
        max_input_channels=2,
        max_output_channels=2,
        name="Line 4",
        index=index,
    )


def test_input_resolver_returns_found_device(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: _device(3))
    cfg = RecordingConfig(input_device_name="Line 4")
    assert audio.resolve_input_device(cfg).index == 3


def test_input_resolver_raises_named_error_when_missing(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = RecordingConfig(input_host_api_name="MME", input_device_name="Ghost")
    with pytest.raises(DeviceNotFoundError) as ei:
        audio.resolve_input_device(cfg)
    assert "input_device_name" in str(ei.value)
    assert "Ghost" in str(ei.value)


def test_output_resolver_raises_named_error_when_missing(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = PlaybackConfig(output_device_name="Ghost")
    with pytest.raises(DeviceNotFoundError) as ei:
        audio.resolve_output_device(cfg)
    assert "output_device_name" in str(ei.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_device_resolver.py -q`
Expected: FAIL（`ImportError: cannot import name 'DeviceNotFoundError'`）。

- [ ] **Step 3: DeviceNotFoundError を追加**

`vspeech/exceptions.py` に追記:

```python
class DeviceNotFoundError(Exception):
    """設定で指定したオーディオデバイスが解決できない。"""

    pass
```

- [ ] **Step 4: resolver を lib/audio.py に追加**

`vspeech/lib/audio.py` の import に追加（single-line, ruff ソート）:

```python
from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.exceptions import DeviceNotFoundError
```

`get_sd_dtype` の後などに追加:

```python
def resolve_input_device(config: RecordingConfig) -> DeviceInfo:
    """録音入力デバイスを解決する。見つからなければ DeviceNotFoundError。

    preflight と recording worker が同じ経路を通る (ADR-0038)。index 指定が
    あれば優先し、無ければ host_api / name で検索する。
    """
    if config.input_device_index is not None:
        try:
            return get_device_info(config.input_device_index)
        except Exception as e:
            raise DeviceNotFoundError(
                f"recording.input_device_index={config.input_device_index} "
                f"が無効です: {e}"
            ) from e
    device = search_device(
        host_api_type=config.input_host_api_name,
        name=config.input_device_name,
        input=True,
    )
    if device is None:
        raise DeviceNotFoundError(
            "入力デバイスが見つかりません "
            f"(recording.input_host_api_name={config.input_host_api_name!r}, "
            f"recording.input_device_name={config.input_device_name!r})"
        )
    return device


def resolve_output_device(config: PlaybackConfig) -> DeviceInfo:
    """再生出力デバイスを解決する。見つからなければ DeviceNotFoundError。"""
    if config.output_device_index is not None:
        try:
            return get_device_info(config.output_device_index)
        except Exception as e:
            raise DeviceNotFoundError(
                f"playback.output_device_index={config.output_device_index} "
                f"が無効です: {e}"
            ) from e
    device = search_device(
        host_api_type=config.output_host_api_name,
        name=config.output_device_name,
        output=True,
    )
    if device is None:
        raise DeviceNotFoundError(
            "出力デバイスが見つかりません "
            f"(playback.output_host_api_name={config.output_host_api_name!r}, "
            f"playback.output_device_name={config.output_device_name!r})"
        )
    return device
```

（注: `lib/audio.py` が `vspeech.config` から `RecordingConfig`/`PlaybackConfig` を import しても循環しない。config.py は audio を import しない。）

- [ ] **Step 5: Run resolver tests**

Run: `uv run pytest tests/test_device_resolver.py -q`
Expected: PASS（3 tests）。

- [ ] **Step 6: recording worker のガードを resolver へ置換**

`vspeech/worker/recording.py` の `open_input_stream` を差し替え:

```python
def open_input_stream(config: RecordingConfig) -> sd.RawInputStream:
    device = resolve_input_device(config)
    logger.info("use input device %s: %s", device.index, device.name)
    stream = sd.RawInputStream(
        samplerate=config.rate,
        blocksize=config.chunk,
        device=device.index,
        channels=config.channels,
        dtype=get_sd_dtype(config.format),
    )
    stream.start()
    return stream
```

import を更新: `from vspeech.lib.audio import get_device_name` と `from vspeech.lib.audio import search_device` を削除し、`from vspeech.lib.audio import resolve_input_device` を追加（`get_sd_dtype` は残す。`get_device_name` が他で使われていないことを確認: `grep get_device_name vspeech/worker/recording.py`）。

- [ ] **Step 7: playback worker のガードを resolver へ置換**

`vspeech/worker/playback.py` の `get_output_device` を差し替え:

```python
def get_output_device(config: PlaybackConfig):
    return resolve_output_device(config)
```

import に `from vspeech.lib.audio import resolve_output_device` を追加。`search_device` が playback.py の他所（`search_appropriate_device` は `search_device_by_name` を使う別物）で未使用になったら import を削除（`grep "search_device\b" vspeech/worker/playback.py` で確認。`search_device_by_name` は残す）。

- [ ] **Step 8: preflight に device チェッカを追加**

`vspeech/preflight.py` に追記:

```python
def _check_recording(config: Config) -> list[ConfigProblem]:
    if not config.recording.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_input_device
    from vspeech.shared_context import WorkerOutput

    w = "recording"
    problems: list[ConfigProblem] = []
    try:
        resolve_input_device(config.recording)
    except DeviceNotFoundError as e:
        problems.append(ConfigProblem(w, str(e)))
    try:
        WorkerOutput.from_routes_list(config.recording.routes_list)
    except Exception as e:
        problems.append(ConfigProblem(w, f"recording.routes_list が不正です: {e}"))
    return problems


def _check_playback(config: Config) -> list[ConfigProblem]:
    if not config.playback.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_output_device

    try:
        resolve_output_device(config.playback)
    except DeviceNotFoundError as e:
        return [ConfigProblem("playback", str(e))]
    return []
```

`_CHECKERS` を更新:

```python
_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_recording,
    _check_playback,
]
```

- [ ] **Step 9: preflight の device テストを追加**

`tests/test_preflight.py` に追記:

```python
def test_recording_device_not_found_is_reported(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = Config(
        recording=RecordingConfig(enable=True, input_device_name="Ghost")
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any(p.worker == "recording" for p in ei.value.problems)


def test_recording_bad_route_is_reported(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "search_device", lambda **kw: __import__("tests.test_device_resolver", fromlist=["_device"])._device(1)
    )
    cfg = Config(
        recording=RecordingConfig(
            enable=True, input_device_index=1, routes_list=[["not_an_event"]]
        )
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("routes_list" in p.detail for p in ei.value.problems)
```

（`input_device_index=1` を渡すと resolver は `get_device_info(1)` を呼ぶ。これは sounddevice に触れるので、テストでは `audio.get_device_info` も monkeypatch する。簡潔にするため下の形へ差し替え:）

```python
def test_recording_bad_route_is_reported(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio
    from tests.test_device_resolver import _device

    monkeypatch.setattr(audio, "get_device_info", lambda i: _device(i))
    cfg = Config(
        recording=RecordingConfig(
            enable=True, input_device_index=1, routes_list=[["not_an_event"]]
        )
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("routes_list" in p.detail for p in ei.value.problems)
```

- [ ] **Step 10: Run tests + lint/type**

Run: `uv run pytest tests/test_preflight.py tests/test_device_resolver.py -q && uv run ruff check vspeech/lib/audio.py vspeech/worker/recording.py vspeech/worker/playback.py vspeech/preflight.py && uv run ty check`
Expected: PASS、ruff clean、ty exit 0。

- [ ] **Step 11: Commit**

```bash
git add vspeech/exceptions.py vspeech/lib/audio.py vspeech/worker/recording.py vspeech/worker/playback.py vspeech/preflight.py tests/test_device_resolver.py tests/test_preflight.py
git commit -m "feat(preflight): shared device resolver + recording/playback device checks"
```

---

## Task 3: translation / tts / vc の preflight チェッカ

残りの安価な設定検査を追加する。

**Files:**
- Modify: `vspeech/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `_check_gcp_credentials`, `_check_vad_gate`, `_CHECKERS`。

- [ ] **Step 1: Write the failing test**

`tests/test_preflight.py` に追記:

```python
def test_translation_missing_gcp_key_is_reported():
    from vspeech.config import GcpConfig
    from vspeech.config import TranslationConfig

    cfg = Config(
        translation=TranslationConfig(enable=True),
        gcp=GcpConfig(service_account_file_path=Path("/no/such/key.json")),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any(p.worker == "translation" for p in ei.value.problems)


def test_voicevox_missing_dirs_reported():
    from vspeech.config import TtsConfig
    from vspeech.config import TtsWorkerType
    from vspeech.config import VoicevoxConfig

    cfg = Config(
        tts=TtsConfig(enable=True, worker_type=TtsWorkerType.VOICEVOX),
        voicevox=VoicevoxConfig(
            openjtalk_dir=Path("/no/dict"), model_dir=Path("/no/models")
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    details = [p.detail for p in ei.value.problems]
    assert any("voicevox.openjtalk_dir" in d for d in details)
    assert any("voicevox.model_dir" in d for d in details)


def test_vr2_tts_passes_without_files():
    # VR2 は実初期化が層B。preflight は通す。
    from vspeech.config import TtsConfig

    preflight(Config(tts=TtsConfig(enable=True)))  # 既定 worker_type=VR2


def test_vc_missing_model_files_reported():
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    cfg = Config(
        vc=VcConfig(enable=True),
        rvc=RvcConfig(
            model_file=Path("/no/model.onnx"),
            hubert_model_file=Path("/no/hubert"),
            rmvpe_model_file=Path("/no/rmvpe.onnx"),
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    details = [p.detail for p in ei.value.problems]
    assert any("rvc.model_file" in d for d in details)
    assert any("rvc.hubert_model_file" in d for d in details)
    assert any("rvc.rmvpe_model_file" in d for d in details)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preflight.py -q -k "translation or voicevox or vr2 or vc_missing"`
Expected: FAIL（該当ワーカーがまだチェックされず ConfigError が上がらない）。

- [ ] **Step 3: チェッカを追加**

`vspeech/preflight.py` に追記:

```python
def _check_translation(config: Config) -> list[ConfigProblem]:
    if not config.translation.enable:
        return []
    return _check_gcp_credentials(config.gcp, "translation")


def _check_tts(config: Config) -> list[ConfigProblem]:
    if not config.tts.enable:
        return []
    from vspeech.config import TtsWorkerType

    if config.tts.worker_type != TtsWorkerType.VOICEVOX:
        return []  # VR2 の実初期化は層B
    w = "tts"
    vv = config.voicevox
    problems: list[ConfigProblem] = []
    for name, path in (
        ("voicevox.openjtalk_dir", vv.openjtalk_dir),
        ("voicevox.model_dir", vv.model_dir),
    ):
        if not path.expanduser().is_dir():
            problems.append(ConfigProblem(w, f"{name} '{path}' が存在しません"))
    if vv.onnxruntime_path is not None and not vv.onnxruntime_path.expanduser().is_file():
        problems.append(
            ConfigProblem(
                w, f"voicevox.onnxruntime_path '{vv.onnxruntime_path}' が存在しません"
            )
        )
    return problems


def _check_vc(config: Config) -> list[ConfigProblem]:
    if not config.vc.enable:
        return []
    from vspeech.config import F0ExtractorType

    w = "vc"
    rvc = config.rvc
    problems: list[ConfigProblem] = []
    if not rvc.model_file.expanduser().is_file():
        problems.append(
            ConfigProblem(w, f"rvc.model_file '{rvc.model_file}' が存在しません")
        )
    if not rvc.hubert_model_file.expanduser().is_dir():
        problems.append(
            ConfigProblem(
                w,
                f"rvc.hubert_model_file '{rvc.hubert_model_file}' "
                "(資産ディレクトリ) が存在しません",
            )
        )
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        if not rvc.rmvpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    w, f"rvc.rmvpe_model_file '{rvc.rmvpe_model_file}' が存在しません"
                )
            )
    problems.extend(_check_vad_gate(config.vc, w))
    return problems
```

`_CHECKERS` を更新:

```python
_CHECKERS: list[Checker] = [
    _check_transcription,
    _check_translation,
    _check_tts,
    _check_vc,
    _check_recording,
    _check_playback,
]
```

- [ ] **Step 4: Run tests + type**

Run: `uv run pytest tests/test_preflight.py -q && uv run ty check`
Expected: PASS、ty exit 0。

- [ ] **Step 5: Commit**

```bash
git add vspeech/preflight.py tests/test_preflight.py
git commit -m "feat(preflight): translation/tts/vc config checks"
```

---

## Task 4: 防御コード削減 + recording_log の DEGRADE 化

preflight が前提を断言することで到達不能になった防御を削り、config バリデーションを足す。recording_log の保存失敗を fatal から DEGRADE へ。

**Files:**
- Modify: `vspeech/config.py`（`RecordingConfig` に `gt=0`）
- Modify: `vspeech/worker/recording.py`（`denom<=0` ガード削除）
- Modify: `vspeech/worker/transcription.py`（Enum `else` を `assert_never`、`log_transcribed` の DEGRADE 化）
- Modify: `vspeech/worker/tts.py`（Enum `else` を `assert_never`）
- Test: `tests/test_preflight.py`（新規 `tests/test_config_bounds.py`）

**Interfaces:**
- Consumes: なし（内部整理）。

- [ ] **Step 1: Write the failing test**

`tests/test_config_bounds.py` を新規作成:

```python
import pytest
from pydantic import ValidationError

from vspeech.config import RecordingConfig


def test_recording_positive_bounds_reject_zero():
    with pytest.raises(ValidationError):
        RecordingConfig(rate=0)
    with pytest.raises(ValidationError):
        RecordingConfig(channels=0)
    with pytest.raises(ValidationError):
        RecordingConfig(chunk=0)
    # 既定は許容
    RecordingConfig()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_bounds.py -q`
Expected: FAIL（`rate=0` などが通ってしまい `ValidationError` が上がらない）。

- [ ] **Step 3: config に gt=0 を追加**

`vspeech/config.py` の `RecordingConfig`:

```python
    channels: int = Field(default=1, gt=0, description="recording channels")
    rate: int = Field(default=16000, gt=0, description="recording rate")
    chunk: int = Field(default=1024, gt=0, description="recording block size")
```

- [ ] **Step 4: denom ガードを削除**

`vspeech/worker/recording.py` の `utterance_capture_sec`:

```python
def utterance_capture_sec(frames: bytes, config: RecordingConfig) -> float:
    denom = get_sample_size(config.format) * config.channels * config.rate
    return len(frames) / denom
```

（rate/channels は `gt=0`、`get_sample_size` は既定 INT16 で ≥1 なので denom>0。）

- [ ] **Step 5: Run config test**

Run: `uv run pytest tests/test_config_bounds.py tests/test_recording_metrics.py -q`
Expected: PASS。

- [ ] **Step 6: Enum else を assert_never へ（transcription）**

`vspeech/worker/transcription.py` の `transcription_worker`。`if/elif/elif ... else: raise ValueError("transcription worker type unknown.")` を差し替え:

```python
            if config.worker_type == TranscriptionWorkerType.ACP:
                generator = partial(
                    transcript_worker_ami, ami_config=context.config.ami
                )
            elif config.worker_type == TranscriptionWorkerType.GCP:
                generator = partial(
                    transcript_worker_google, gcp_config=context.config.gcp
                )
            elif config.worker_type == TranscriptionWorkerType.WHISPER:
                generator = partial(
                    transcript_worker_whisper, whisper_config=context.config.whisper
                )
            else:
                assert_never(config.worker_type)
```

import に `from typing import assert_never` を追加。

- [ ] **Step 7: Enum else を assert_never へ（tts）**

`vspeech/worker/tts.py` の `tts_worker` の末尾 `else: raise ValueError("tts worker type unknown.")` を:

```python
            else:
                assert_never(config.worker_type)
```

import に `from typing import assert_never` を追加。

- [ ] **Step 8: recording_log を DEGRADE 化**

`vspeech/worker/transcription.py` の `log_transcribed` を、保存失敗で警告して素通しする形へ。関数冒頭に module-level dedup を用意（既存 `_vad_gate_warned` パターンに倣う）:

```python
_rec_log_warned: set[str] = set()


async def log_transcribed(log_dir_parent: Path, wav_file: BytesIO, text: str):
    try:
        now = datetime.now()
        log_dir = Path(log_dir_parent.expanduser() / now.strftime("%Y%m%d"))
        log_dir.mkdir(exist_ok=True, parents=True)
        log_wav_name = now.strftime("%Y%m%d%H%M%S.wav")
        log_txt_name = now.strftime("%Y%m%d%H%M%S.txt")
        async with aio_open(log_dir / log_wav_name, "wb") as log:
            wav_file.seek(0)
            await log.write(wav_file.read())
        if not text:
            return
        async with aio_open(
            log_dir / log_txt_name, "w", encoding="utf-8", errors="backslashreplace"
        ) as log:
            await log.write(text)
    except OSError as e:
        # 録音ログは補助機能: 保存先が書込不可でもパイプラインは止めない (DEGRADE)。
        key = str(log_dir_parent)
        if key in _rec_log_warned:
            logger.debug("recording_log 保存失敗 (継続): %s", e)
        else:
            _rec_log_warned.add(key)
            logger.warning(
                "recording_log を保存できません (%s) — ログ保存のみ無効化して継続: %s",
                log_dir_parent,
                e,
            )
```

- [ ] **Step 9: DEGRADE テストを追加**

`tests/test_transcription_helpers.py` に追記（存在すれば。無ければ `tests/test_preflight.py` に置く。ここでは `tests/test_transcription_helpers.py` を対象とする）:

```python
async def test_recording_log_write_failure_degrades(monkeypatch, tmp_path):
    from io import BytesIO

    from vspeech.worker import transcription as tx

    def _boom(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(tx.Path, "mkdir", _boom)
    tx._rec_log_warned.clear()
    # 例外を投げずに返る（プロセスを止めない）
    await tx.log_transcribed(tmp_path, BytesIO(b"data"), "text")
    assert str(tmp_path) in tx._rec_log_warned
```

（`tx.Path` は `transcription.py` が import した `Path`。`monkeypatch.setattr(tx.Path, "mkdir", ...)` は Path 全体に効くため、テスト内限定でよいが安全のため `Path.mkdir` の復元は monkeypatch が自動で行う。）

- [ ] **Step 10: Run tests + lint/type**

Run: `uv run pytest tests/test_config_bounds.py tests/test_transcription_helpers.py tests/test_tts_worker.py -q && uv run ruff check vspeech/config.py vspeech/worker/recording.py vspeech/worker/transcription.py vspeech/worker/tts.py && uv run ty check`
Expected: PASS、ruff clean、ty exit 0（`assert_never` により worker_type が網羅されていることを ty が確認）。

- [ ] **Step 11: Commit**

```bash
git add vspeech/config.py vspeech/worker/recording.py vspeech/worker/transcription.py vspeech/worker/tts.py tests/test_config_bounds.py tests/test_transcription_helpers.py
git commit -m "refactor: drop unreachable config guards; degrade recording_log write failure"
```

---

## Task 5: 層B — worker 起動時の深層失敗を WorkerStartupError へ変換

モデルロード・GPU 確保・ストリーム open など「試すまで分からない」失敗を、worker と原因を名指しする `WorkerStartupError` に変換する。

**Files:**
- Modify: `vspeech/exceptions.py`（`worker_startup` コンテキストマネージャ）
- Modify: `vspeech/worker/vc.py` / `transcription.py` / `tts.py` / `translation.py` / `playback.py`（setup を包む）
- Test: `tests/test_worker_startup.py`

**Interfaces:**
- Consumes: `WorkerStartupError`（Task 1）。
- Produces: `worker_startup(worker: str)` — コンテキストマネージャ。中の任意例外を `WorkerStartupError(worker, str(e))` に変換（既に `WorkerStartupError` なら素通し）。

- [ ] **Step 1: Write the failing test**

`tests/test_worker_startup.py` を新規作成:

```python
import pytest

from vspeech.exceptions import WorkerStartupError
from vspeech.exceptions import worker_startup


def test_translates_arbitrary_exception():
    with pytest.raises(WorkerStartupError) as ei:
        with worker_startup("vc"):
            raise RuntimeError("no CUDA provider")
    assert ei.value.worker == "vc"
    assert "no CUDA provider" in ei.value.detail


def test_passes_through_worker_startup_error():
    with pytest.raises(WorkerStartupError) as ei:
        with worker_startup("tts"):
            raise WorkerStartupError("voicevox", "boom")
    assert ei.value.worker == "voicevox"  # 内側を保持


def test_no_error_is_transparent():
    with worker_startup("vc"):
        x = 1 + 1
    assert x == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_worker_startup.py -q`
Expected: FAIL（`ImportError: cannot import name 'worker_startup'`）。

- [ ] **Step 3: worker_startup を追加**

`vspeech/exceptions.py` に追記:

```python
@contextmanager
def worker_startup(worker: str):
    """worker 起動時のリソース取得失敗を WorkerStartupError へ変換する (層B, ADR-0038)。"""
    try:
        yield
    except WorkerStartupError:
        raise
    except Exception as e:
        raise WorkerStartupError(worker, str(e)) from e
```

- [ ] **Step 4: Run context-manager tests**

Run: `uv run pytest tests/test_worker_startup.py -q`
Expected: PASS（3 tests）。

- [ ] **Step 5: vc の setup を包む**

`vspeech/worker/vc.py` の `rvc_worker`。`device, device_name = get_device(...)` から `f0_enabled = metadata["f0"]` までの setup ブロック（warmup の直前まで）を `with worker_startup("vc"):` でインデント。warmup（既存 try/except）とその後の `while True:` は with の**外**に置く。import に `from vspeech.exceptions import worker_startup` を追加。

具体例（抜粋・インデント +4）:

```python
    with worker_startup("vc"):
        device, device_name = get_device(rvc_config.gpu_id, rvc_config.gpu_name)
        logger.info("vc worker device: %s, %s", device, device_name)
        half_available = half_precision_available(id=device.index)
        hubert_model = load_hubert_model(
            file_name=rvc_config.hubert_model_file,
            device=device,
            is_half=half_available,
        )
        session = create_session(rvc_config.model_file, device)
        check_cuda_provider(session.get_providers())
        if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
            rmvpe_session = create_session(rvc_config.rmvpe_model_file, device)
        else:
            rmvpe_session = None
        if vc_config.vad_gate:
            vad_session = create_vad_session(vc_config.vad_model_file)
            logger.info("vad gate enabled: %s", vc_config.vad_model_file)
        else:
            vad_session = None
        modelmeta: Any = session.get_modelmeta()
        metadata: dict[str, Any] = json.loads(
            modelmeta.custom_metadata_map["metadata"]
        )
        target_sample_rate = metadata["samplingRate"]
        f0_enabled = metadata["f0"]
    # warmup (DEGRADE) と本ループは with の外
    try:
        await to_thread(change_voice, ...)  # 既存のまま
        ...
```

- [ ] **Step 6: transcription の setup を包む**

`vspeech/worker/transcription.py`:
- `transcript_worker_whisper`: `device, device_name = get_device(...)` と `model = WhisperModel(...)` を `with worker_startup("transcription"):` で包む（warmup と `vad_session = create_transcription_vad_session(config)` の後の `while True:` は外。ただし VAD セッション生成も setup なので with 内に含める — 現状 warmup 後にあるので、`vad_session = create_transcription_vad_session(config)` を with 内・warmup の前へ移動して包む）。
- `transcript_worker_google`: `credentials, _ = get_credentials(gcp_config)` と `client = SpeechAsyncClient(...)` と `vad_session = create_transcription_vad_session(config)` を `with worker_startup("transcription"):` で包む。
- `transcript_worker_ami`: `vad_session = create_transcription_vad_session(config)` を `with worker_startup("transcription"):` で包む（ami は他に startup リソース無し）。

import に `from vspeech.exceptions import worker_startup`。

- [ ] **Step 7: tts / translation / playback の setup を包む**

- `vspeech/worker/tts.py` `voicevox_worker`: `vvox = Voicevox(...)` を `with worker_startup("tts"):` で包む。`vroid2_worker` は `tts_worker` 側で VR2 初期化しているので、`tts_worker` の `vr2 = VR2()` 〜 `vr2.load_voice(...)` ブロックを `with worker_startup("tts"):` で包む（`with vr2.vc_roid2:` の内側の初期化群）。
- `vspeech/worker/translation.py` `translation_worker_google`: `credentials, project_id = get_credentials(gcp_config)` と `client = TranslationServiceAsyncClient(...)` を `with worker_startup("translation"):` で包む。
- `vspeech/worker/playback.py` `pyaudio_playback_worker`: `output_stream = OutputStream(config)` を `with worker_startup("playback"):` で包む（`OutputStream.__post_init__` がデバイス解決・ログを行う）。

各ファイルの import に `from vspeech.exceptions import worker_startup`。

- [ ] **Step 8: Run full suite + lint/type**

Run: `uv run pytest -q && uv run ruff check vspeech && uv run ty check`
Expected: 全 PASS（既存 + 新規、～260 件）、ruff clean、ty exit 0。

- [ ] **Step 9: Commit**

```bash
git add vspeech/exceptions.py vspeech/worker/vc.py vspeech/worker/transcription.py vspeech/worker/tts.py vspeech/worker/translation.py vspeech/worker/playback.py tests/test_worker_startup.py
git commit -m "feat(worker): attribute deep startup failures as WorkerStartupError"
```

---

## Task 6: 最終ゲート + ADR 昇格 + follow-ups

**Files:**
- Modify: `docs/adr/0038-worker-config-preflight-fail-loud.md`（Proposed → Accepted）
- Modify: `docs/adr/README.md`（索引の Status）
- Modify: `docs/follow-ups.md`（該当項目の解消・繰り延べ記録）

- [ ] **Step 1: whole-project の健全性ゲート**

Run: `uv run poe check`
Expected: 緑（既存の受理済み 2 件＝torch CVE / deadcode vr2_config 以外の新規指摘なし）。指摘が出たら潰す。

- [ ] **Step 2: 実モデル smoke（VAD 経路の起動 fail-loud を実確認）**

VAD モデルが手元にある場合のみ: `vad_gate=true` + 実在モデルで transcription を 1 発話通し、`vad_gate=true` + 不在パスでプロセスが起動時に停止することを確認。無い場合は `tests/test_preflight.py` の該当テストで代替（受入基準は満たす）。

- [ ] **Step 3: エントリポイント起動確認（LESSON: run the entry point, not just tests）**

Run: 空・不正な config で `uv run python -m vspeech --config <壊れた config.toml>` を実行し、集約された設定不備の整形ログ + `exit 1` を目視。正常 config では従来どおり worker が起動することを確認。

- [ ] **Step 4: ADR を Accepted へ昇格**

`docs/adr/0038-worker-config-preflight-fail-loud.md` の `- Status: Proposed` を `- Status: Accepted` に（1 行）。`docs/adr/README.md` の 0038 行の Status を `Proposed` → `Accepted` に。

- [ ] **Step 5: follow-ups.md を更新**

`docs/follow-ups.md` の transcription VAD 節「whisper だけ fail-loud のタイミングが遅い」項に、本ブランチで VAD セッション生成を setup 内（層B, model load 前）へ寄せて解消した旨を 1 行追記。今回スコープ外にした点（recording のストリーム open reload 経路を層B 化していない等）があれば「なぜ見送ったか」付きで追記。

- [ ] **Step 6: Commit**

```bash
git add docs/adr/0038-worker-config-preflight-fail-loud.md docs/adr/README.md docs/follow-ups.md
git commit -m "docs(adr): promote ADR-0038 to Accepted after implementation"
```

---

## Self-Review

**1. Spec coverage:**
- 受入基準①（稼働前停止）→ Task 1（preflight は TaskGroup 前）。
- ②（複数不備を集約）→ Task 1（`preflight` が全 checker を回して集約）+ test_acp_missing_fields_are_all_reported。
- ③（worker/設定値名指し・非 traceback）→ Task 1（`ConfigProblem` + cmd の整形ログ）。
- ④（ACP 4 フィールド）→ Task 1 `_check_transcription`。
- ⑤（入出力デバイス・名指し）→ Task 2 resolver + `_check_recording`/`_check_playback`。
- ⑥（VAD gate + モデル欠如で起動時停止）→ Task 1 `_check_vad_gate` + Task 3 vc。
- ⑦（feature 縮退＝recording_log）→ Task 4 `log_transcribed` DEGRADE。
- ⑧（到達不能防御コード削減）→ Task 2（device ガード）+ Task 4（Enum else / denom）。
- ⑨（HW 非依存テスト）→ Task 1-5 の `tests/test_preflight.py` 他（monkeypatch / tmp_path）。

**2. Placeholder scan:** TODO/TBD なし。全コードブロックは実内容。

**3. Type consistency:** `ConfigProblem(worker, detail)` / `ConfigError(problems)` / `WorkerStartupError(worker, detail)` / `resolve_input_device`/`resolve_output_device` / `worker_startup(worker)` はタスク間で一貫。`_CHECKERS` は Task 1/2/3 で段階的に伸長（最終 6 チェッカ）。
