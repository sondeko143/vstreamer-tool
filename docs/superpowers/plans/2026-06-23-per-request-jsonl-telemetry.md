# per-request 構造化テレメトリ（JSONL）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1発話ごとの各段所要時間と真E2Eを `trace_id` 付き JSONL 1行/段で即時永続化し、`log_file`/`jsonl_path` を UNC ネットワークパスでも耐障害に使えるようにする。

**Architecture:** `Telemetry.record/record_e2e/timer` に任意 `trace_id` 引数を足し、`Telemetry._emit_jsonl` で設定パスへ JSONL を追記。各ワーカーは処理中の `WorkerInput.trace_id` を渡す。`configure_logger` と JSONL の mkdir/open を try/except 化してネットワーク不通でも起動継続。

**Tech Stack:** Python 3.11、pydantic v2、pytest（`asyncio_mode="auto"`）、uv/ruff/ty。標準ライブラリのみ（json/os/pathlib/datetime）。

設計の出典: [docs/superpowers/specs/2026-06-23-per-request-jsonl-telemetry-design.md](../specs/2026-06-23-per-request-jsonl-telemetry-design.md)

## Global Constraints

- Python **3.11 のみ**。pydantic **v2** API のみ。imports は**一行ずつ**（ruff `force-single-line`）。
- テストは `tests/`、pytest `asyncio_mode="auto"`。
- 静的検査 `uv run ruff check .` / `uv run ty check` を **触ったファイルで** green（pruned extra 由来の unresolved-import ノイズが全体 ty に出る場合があるため、各タスクは触ったファイル指定で確認。全体は最後に確認）。
- JSONL は段ごと1行（trace_id で後結合）。protos は無改修。
- UNC パスは TOML の**シングルクォート（リテラル文字列）**で記述。`%%`→`%` 変換後に strftime（`log_file` と同規約）。
- 既存挙動（終了時サマリ、人間向け行ログ）は**非破壊**。`trace_id` は既定 `""` の任意引数＝後方互換。

---

## Task 1: Telemetry に trace_id 引数＋JSONL sink

**Files:**
- Modify: `vspeech/lib/telemetry.py`
- Test: `tests/test_telemetry_jsonl.py`（新規）

**Interfaces:**
- Produces:
  - `Telemetry.configure(enabled: bool, max_samples: int, jsonl_path: str = "") -> None`
  - `Telemetry.record(stage: str, seconds: float, trace_id: str = "") -> None`
  - `Telemetry.record_e2e(seconds: float, trace_id: str = "") -> None`
  - `Telemetry.timer(stage: str, trace_id: str = "")` … contextmanager
  - JSONL 1行 = `{"ts": float, "trace_id": str, "stage": str, "dur_s": float, "pid": int}`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_telemetry_jsonl.py`:

```python
import json

from vspeech.lib.telemetry import Telemetry


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_writes_jsonl(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    t.record("vc", 1.25, trace_id="abc")
    rows = _read(p)
    assert len(rows) == 1
    r = rows[0]
    assert r["stage"] == "vc"
    assert r["dur_s"] == 1.25
    assert r["trace_id"] == "abc"
    assert isinstance(r["pid"], int)
    assert isinstance(r["ts"], float)


def test_timer_writes_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    with t.timer("tts", trace_id="t2"):
        pass
    rows = _read(p)
    assert rows[0]["stage"] == "tts"
    assert rows[0]["trace_id"] == "t2"


def test_record_e2e_writes_stage_e2e(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    t.record_e2e(1.7, trace_id="t3")
    rows = _read(p)
    assert rows[0]["stage"] == "e2e"
    assert rows[0]["dur_s"] == 1.7
    assert rows[0]["trace_id"] == "t3"


def test_no_jsonl_when_path_empty(tmp_path):
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path="")
    t.record("vc", 1.0, trace_id="x")
    assert not (tmp_path / "tel.jsonl").exists()


def test_disabled_writes_nothing(tmp_path):
    p = tmp_path / "tel.jsonl"
    t = Telemetry()
    t.configure(enabled=False, max_samples=100, jsonl_path=str(p))
    t.record("vc", 1.0, trace_id="x")
    assert not p.exists()


def test_uncreatable_path_is_resilient(tmp_path):
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    bad = str(afile / "sub" / "tel.jsonl")  # parent is a file → mkdir fails
    t = Telemetry()
    t.configure(enabled=True, max_samples=100, jsonl_path=bad)  # must not raise
    t.record("vc", 1.0, trace_id="x")  # must not raise
    assert t.summary()["vc"]["count"] == 1  # in-memory aggregation still works
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_telemetry_jsonl.py -q`
Expected: FAIL（`configure()` が `jsonl_path` を受けない / `record()` が `trace_id` を受けない）

- [ ] **Step 3: 実装**

`vspeech/lib/telemetry.py` の import を次へ（一行ずつ・ruff順）:

```python
import json
import os
from contextlib import contextmanager
from datetime import datetime
from math import ceil
from math import floor
from pathlib import Path
from time import perf_counter
from time import time

from vspeech.logger import logger
```

`Telemetry.__init__` に handle を追加:

```python
    def __init__(self) -> None:
        self.enabled: bool = False
        self.max_samples: int = 5000
        self._durations: dict[str, list[float]] = {}
        self._e2e: list[float] = []
        self._jsonl = None
```

`configure` / `reset` を置換:

```python
    def configure(
        self, enabled: bool, max_samples: int, jsonl_path: str = ""
    ) -> None:
        self.enabled = enabled
        self.max_samples = max_samples
        self._close_jsonl()
        if enabled and jsonl_path:
            try:
                resolved = datetime.now().strftime(jsonl_path.replace("%%", "%"))
                path = Path(resolved)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._jsonl = open(path, "a", encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "telemetry jsonl disabled (cannot open %s): %s", jsonl_path, e
                )
                self._jsonl = None

    def _close_jsonl(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.close()
            except OSError:
                pass
            self._jsonl = None

    def reset(self) -> None:
        self._durations = {}
        self._e2e = []
        self._close_jsonl()
```

`record` / `record_e2e` / `timer` に trace_id を追加し、`_emit_jsonl` を追加:

```python
    def record(self, stage: str, seconds: float, trace_id: str = "") -> None:
        if not self.enabled:
            return
        self._append(self._durations.setdefault(stage, []), seconds)
        self._emit_jsonl(stage, seconds, trace_id)

    def record_e2e(self, seconds: float, trace_id: str = "") -> None:
        if not self.enabled:
            return
        self._append(self._e2e, seconds)
        self._emit_jsonl("e2e", seconds, trace_id)

    @contextmanager
    def timer(self, stage: str, trace_id: str = ""):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start, trace_id)

    def _emit_jsonl(self, stage: str, seconds: float, trace_id: str) -> None:
        if self._jsonl is None:
            return
        record = {
            "ts": time(),
            "trace_id": trace_id,
            "stage": stage,
            "dur_s": seconds,
            "pid": os.getpid(),
        }
        try:
            self._jsonl.write(json.dumps(record) + "\n")
            self._jsonl.flush()
        except OSError as e:
            logger.warning("telemetry jsonl write failed, disabling: %s", e)
            self._close_jsonl()
```

注: 既存の `_append` / `summary` / `log_summary` は不変。

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest tests/test_telemetry_jsonl.py tests/test_telemetry.py -q`
Expected: PASS（新6件＋既存 telemetry 5件 green。`record(stage, seconds)` の既存呼び出しは trace_id 既定 `""` で後方互換）

- [ ] **Step 5: 静的検査＋コミット**

```bash
uv run ruff check vspeech/lib/telemetry.py tests/test_telemetry_jsonl.py
uv run ty check vspeech/lib/telemetry.py
git add vspeech/lib/telemetry.py tests/test_telemetry_jsonl.py
git commit -m "feat(telemetry): per-request JSONL sink with trace_id"
```
Expected: ruff/ty とも `All checks passed!`。

---

## Task 2: TelemetryConfig.jsonl_path＋main 配線＋config 例

**Files:**
- Modify: `vspeech/config.py`（`TelemetryConfig` に `jsonl_path`）
- Modify: `vspeech/main.py`（`configure` に jsonl_path）
- Modify: `config.toml.example`（`[telemetry]` に jsonl_path＋UNC注記）
- Test: `tests/test_telemetry_config.py`（既存に追記）

**Interfaces:**
- Consumes: `Telemetry.configure(enabled, max_samples, jsonl_path)`（Task 1）。
- Produces: `TelemetryConfig.jsonl_path: str`（既定 `""`）。

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_telemetry_config.py` に追記:

```python
def test_jsonl_path_default_empty():
    assert TelemetryConfig().jsonl_path == ""
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_telemetry_config.py::test_jsonl_path_default_empty -q`
Expected: FAIL（`jsonl_path` 属性なし）

- [ ] **Step 3: 実装**

`vspeech/config.py` の `TelemetryConfig` に1行追加:

```python
class TelemetryConfig(BaseModel):
    enable: bool = True
    max_samples: int = 5000
    log_raw_e2e: bool = True
    skew_warn_threshold: float = 120.0
    jsonl_path: str = ""
```

`vspeech/main.py` の `telemetry.configure(...)` 呼び出しに `jsonl_path` を追加:

```python
    telemetry.configure(
        enabled=config.telemetry.enable,
        max_samples=config.telemetry.max_samples,
        jsonl_path=config.telemetry.jsonl_path,
    )
```

`config.toml.example` の `[telemetry]` 節に追記（既存 `skew_warn_threshold` の後など）:

```toml
# 1発話ごとの各段所要時間+E2Eを JSONL で記録する出力先。空=無効（既定）。
# %%Y 等の日付パターン可。Windows/UNC ネットワークパスは TOML の
# シングルクォート(リテラル文字列)で書くこと（ダブルクォートだと \d 等が壊れる）:
#   jsonl_path = '\\<NAS_HOST>\d\vs\tel_%%Y%%m%%d.jsonl'
jsonl_path = ""
```

- [ ] **Step 4: PASS 確認**

Run: `uv run pytest tests/test_telemetry_config.py -q`
Expected: PASS（3件）

- [ ] **Step 5: 静的検査＋コミット**

```bash
uv run ruff check vspeech/config.py vspeech/main.py tests/test_telemetry_config.py
uv run ty check vspeech/config.py vspeech/main.py
git add vspeech/config.py vspeech/main.py config.toml.example tests/test_telemetry_config.py
git commit -m "feat(telemetry): TelemetryConfig.jsonl_path wiring + config example (UNC note)"
```

---

## Task 3: logger.py の UNC/ネットワーク耐障害化

**Files:**
- Modify: `vspeech/logger.py`（`configure_logger` の mkdir/ファイルハンドラ生成を try/except）
- Test: `tests/test_logger_resilience.py`（新規）

**Interfaces:**
- Produces: `configure_logger(config)` は作成不能な `log_file` でも例外を投げず、stdout ハンドラのみで継続。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_logger_resilience.py`:

```python
from vspeech.config import Config
from vspeech.logger import configure_logger
from vspeech.logger import logger


def test_configure_logger_survives_bad_log_file(tmp_path):
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    bad = str(afile / "sub" / "voice.log")  # parent is a file → mkdir fails
    cfg = Config()
    cfg.log_file = bad
    before = list(logger.handlers)
    try:
        configure_logger(cfg)  # must not raise
        # stdout handler still attached (at least one handler present)
        assert len(logger.handlers) >= 1
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_logger_resilience.py -q`
Expected: FAIL（`configure_logger` が mkdir で `NotADirectoryError`/`OSError` を送出）

- [ ] **Step 3: 実装**

`vspeech/logger.py` の `configure_logger` 内、ファイルハンドラ生成ブロックを try/except 化。現状:

```python
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        file_handler = TaskFileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(log_file_format)
        file_handler.setLevel(config.log_level)
        logger.addHandler(file_handler)
```

を次へ置換（失敗は stderr へ警告して継続。logger はまだ stdout ハンドラ未追加のため `print(..., file=stderr)` を使う）:

```python
    if filename:
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            file_handler = TaskFileHandler(filename, encoding="utf-8")
            file_handler.setFormatter(log_file_format)
            file_handler.setLevel(config.log_level)
            logger.addHandler(file_handler)
        except OSError as e:
            print(
                f"log file disabled (cannot open {filename}): {e}",
                file=stderr,
            )
```

import に `from sys import stderr` を追加（一行・ruff順。既存 `from sys import stdout` の近く）。

- [ ] **Step 4: PASS 確認**

Run: `uv run pytest tests/test_logger_resilience.py -q`
Expected: PASS（1件）

- [ ] **Step 5: 静的検査＋コミット**

```bash
uv run ruff check vspeech/logger.py tests/test_logger_resilience.py
uv run ty check vspeech/logger.py
git add vspeech/logger.py tests/test_logger_resilience.py
git commit -m "fix(logger): resilient log_file open for UNC/network paths"
```

---

## Task 4: 各ワーカーで trace_id を渡す

**Files:**
- Modify: `vspeech/worker/vc.py`（`record_vc_elapsed` に trace_id）
- Modify: `vspeech/worker/playback.py`（`record_e2e` に trace_id＋timer）
- Modify: `vspeech/worker/transcription.py`（timer×3 に trace_id）
- Modify: `vspeech/worker/translation.py`（timer に trace_id）
- Modify: `vspeech/worker/tts.py`（timer×2 に trace_id）
- Test: `tests/test_worker_trace_jsonl.py`（新規・純関数の record_vc_elapsed と record_playback_e2e）

**Interfaces:**
- Consumes: `Telemetry.timer/record/record_e2e` の trace_id 引数（Task 1）、`WorkerInput.trace_id`。
- Produces: `record_vc_elapsed(seconds: float, trace_id: str = "") -> None`（vc.py）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_worker_trace_jsonl.py`:

```python
import json
from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.lib.telemetry import telemetry
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.playback import record_playback_e2e
from vspeech.worker.vc import record_vc_elapsed


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _speech(origin_ts: float, trace_id: str) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(event=EventType.playback),
        following_events=[],
        text="",
        sound=SoundInput(data=b"00", rate=16000, format=SampleFormat.INT16, channels=1),
        file_path="",
        filters=[],
        trace_id=trace_id,
        origin_ts=origin_ts,
    )


def test_vc_records_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    record_vc_elapsed(0.5, trace_id="vc-tid")
    telemetry.reset()  # close handle so file is flushed/closed
    rows = _read(p)
    assert rows[0]["stage"] == "vc"
    assert rows[0]["trace_id"] == "vc-tid"


def test_playback_e2e_records_trace_id(tmp_path):
    p = tmp_path / "tel.jsonl"
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100, jsonl_path=str(p))
    cfg = TelemetryConfig()
    record_playback_e2e(_speech(origin_ts=100.0, trace_id="pb-tid"), now=101.0, cfg=cfg)
    telemetry.reset()
    rows = _read(p)
    assert rows[0]["stage"] == "e2e"
    assert rows[0]["trace_id"] == "pb-tid"
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_worker_trace_jsonl.py -q`
Expected: FAIL（`record_vc_elapsed` が trace_id を受けない／`record_playback_e2e` が trace_id を JSONL に載せない）

- [ ] **Step 3: 実装**

**vc.py**: `record_vc_elapsed` を更新し、呼び出しに trace_id を渡す。
```python
def record_vc_elapsed(seconds: float, trace_id: str = "") -> None:
    telemetry.record("vc", seconds, trace_id=trace_id)
    logger.info("rvc elapsed time: %s", seconds)
```
呼び出し（現 `record_vc_elapsed(vc_end_time - vc_start_time)`）を:
```python
            record_vc_elapsed(vc_end_time - vc_start_time, trace_id=speech.trace_id)
```

**playback.py**: `record_playback_e2e` 内の `telemetry.record_e2e(e2e)` を:
```python
    telemetry.record_e2e(e2e, trace_id=speech.trace_id)
```
playback timer（現 `with telemetry.timer("playback"):`）を:
```python
                with telemetry.timer("playback", trace_id=speech.trace_id):
```

**transcription.py**: 3箇所の `with telemetry.timer("transcription"):` を、各々の直近でデキューした `recorded` を使って:
```python
            with telemetry.timer("transcription", trace_id=recorded.trace_id):
```
（whisper 175行・google 233行・ami 289行とも変数は `recorded`）

**translation.py**: 124行の `with telemetry.timer("translation"):` を:
```python
                with telemetry.timer("translation", trace_id=block.original.trace_id):
```
（`TranslationBlock.original` は元 `WorkerInput`、`.trace_id` を持つ）

**tts.py**: 36行・84行の `with telemetry.timer("tts"):` を:
```python
            with telemetry.timer("tts", trace_id=transcribed.trace_id):
```
（両 worker とも変数は `transcribed`）

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest tests/test_worker_trace_jsonl.py tests/test_vc_telemetry.py tests/test_playback_e2e.py -q`
Expected: PASS（新2＋既存 green）

- [ ] **Step 5: 静的検査＋コミット**

```bash
uv run ruff check vspeech/worker/vc.py vspeech/worker/playback.py vspeech/worker/transcription.py vspeech/worker/translation.py vspeech/worker/tts.py tests/test_worker_trace_jsonl.py
uv run ty check vspeech/worker/vc.py vspeech/worker/playback.py
git add vspeech/worker/ tests/test_worker_trace_jsonl.py
git commit -m "feat(telemetry): thread trace_id through all stage timers"
```
Expected: ruff `All checks passed!`。各ワーカーは base import 可能（heavy deps は遅延 import／pyaudio は audio extra で解決済み）。

---

## 最終確認（全タスク後）

```bash
uv run pytest -q
uv run ruff check .
uv run ty check
```
Expected: 全テスト PASS、ruff `All checks passed!`、ty `All checks passed!`（extra 復元済み環境のため pruned-dep ノイズなし）。

---

## Self-Review（計画作成者によるチェック結果）

**1. Spec coverage:**
- §3.1 レコード形式（ts/trace_id/stage/dur_s/pid）→ Task 1 `_emit_jsonl`。
- §3.2 trace_id 受け渡し → Task 1（引数追加）＋Task 4（各 worker）。
- §3.3 sink/既定/mkdir/耐障害 → Task 1 `configure`/`_emit_jsonl`。
- §3.5 起動時配線（`%%`→`%`）→ Task 1 `configure` 内 strftime ＋Task 2 main。
- §3.6 UNC/ネットワーク耐障害（log_file）→ Task 3。TOML 注記 → Task 2 config 例。
- §5 テスト 1-8 → Task1（1-6 のうち書込/無効/timer/e2e/valid/耐障害-jsonl）, Task3（耐障害-log_file=テスト7）, 後方互換（テスト8=Task1 Step4 の test_telemetry.py 回帰）。
- §7 影響範囲（telemetry/config/main/logger/workers/config.example）→ Task 1-4 と一致。

**2. Placeholder scan:** TODO/TBD なし。各コードステップに実コード。

**3. Type consistency:** `configure(enabled, max_samples, jsonl_path="")` / `record(stage, seconds, trace_id="")` / `record_e2e(seconds, trace_id="")` / `timer(stage, trace_id="")` / `record_vc_elapsed(seconds, trace_id="")` / `_emit_jsonl(stage, seconds, trace_id)` を定義(Task1)と利用(Task4)で整合。trace_id 出所：transcription=`recorded`、translation=`block.original`、tts=`transcribed`、vc/playback=`speech`。
