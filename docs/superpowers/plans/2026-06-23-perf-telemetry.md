# 性能テレメトリ（終了時サマリ＋真E2E）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 各プロセスが終了時に「段ごと count/p50/p95/max」を自動集計し、トレースID＋原点時刻を wire 伝搬して発話→再生の真E2Eを計測する基盤を導入する。

**Architecture:** プロセス内シングルトンのテレメトリレジストリ（メモリ集計）に、各ワーカーが `timer` で処理時間を記録。トレースID/原点時刻を vstreamer-protos の新 `Operand` フィールドで伝搬し、終点 playback が真E2Eを算出。終了時に `main` がサマリを出力。

**Tech Stack:** Python 3.11（`asyncio.TaskGroup`/`except*`）、pydantic v2、grpc.aio/protobuf（vstreamer-protos）、pytest（`asyncio_mode="auto"`）、uv/ruff/ty。percentile は純Python（依存ゼロ）。

設計の出典: [docs/superpowers/specs/2026-06-23-perf-telemetry-design.md](../specs/2026-06-23-perf-telemetry-design.md)

## Global Constraints

- Python **3.11 のみ**（`>=3.11,<3.12`）。`TaskGroup`/`except*` 使用可。floorを下げない。
- pydantic **v2** API のみ（`BaseModel`/`ConfigDict`/`model_validate`/`model_dump`/`Field`）。v1 API 禁止。
- imports は**一行ずつ**（ruff `force-single-line`）。
- テストは `tests/`、pytest `asyncio_mode="auto"`（`async def test_...` に装飾不要）。
- 静的検査: `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` を全 green に保つ。
- `vstreamer-protos` は外部 wheel。本計画では**ローカル clone**（`c:\Users\<USER>\vstreamer\vstreamer-protos`）を編集し、開発中は uv の path source で tool venv に入れる。最終タスクで正式リリース＋pin更新に戻す。
- `tests/test_event_chains.py` / `tests/test_worker_input.py` はルーティングの load-bearing テスト。trace 追加で**green を維持**すること。

---

## Phase 0 — vstreamer-protos（別repo）＋ dev ブリッジ

### Task 1: protos に trace_id / origin_ts フィールド追加＋スタブ再生成

**Files (リポジトリ `c:\Users\<USER>\vstreamer\vstreamer-protos`):**
- Modify: `protos/vstreamer_protos/commander/commander.proto:31-36`
- Regenerate: `python/vstreamer_protos/commander/commander_pb2.py` / `commander_pb2.pyi` / `commander_pb2_grpc.py`

**Interfaces:**
- Produces: `Operand` に `trace_id: str`（field 5）, `origin_ts: float`（field 6, proto `double`）。tool 側が `command.operand.trace_id` / `command.operand.origin_ts` を読み書きする。

- [ ] **Step 1: `.proto` に2フィールド追加**

`protos/vstreamer_protos/commander/commander.proto` の `Operand` を次へ:

```proto
message Operand {
    Sound sound = 1;
    string text = 2;
    string file_path = 3;
    repeated string filters = 4;
    string trace_id = 5;
    double origin_ts = 6;
}
```

- [ ] **Step 2: Python スタブを再生成**

protos リポジトリのルートで（grpc_tools を ephemeral に使う）:

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-protos
uv run --with grpcio-tools==1.* --with protobuf python -m grpc_tools.protoc \
  -Iprotos --python_out=python --pyi_out=python --grpc_python_out=python \
  protos/vstreamer_protos/commander/commander.proto
```

Expected: `python/vstreamer_protos/commander/commander_pb2.py` と `commander_pb2.pyi` が更新され、`.pyi` に `trace_id: str` と `origin_ts: float` が現れる。

- [ ] **Step 3: フィールド存在を確認**

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-protos/python
uv run --with protobuf python -c "from vstreamer_protos.commander.commander_pb2 import Operand; o=Operand(trace_id='x', origin_ts=1.5); print(o.trace_id, o.origin_ts)"
```
Expected: `x 1.5`

- [ ] **Step 4: protos リポジトリでコミット**

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-protos
git checkout -b feat/trace-fields
git add protos/ python/
git commit -m "feat: add trace_id/origin_ts to Operand for telemetry E2E"
```

（push と正式リリースは Task 10 で。ここではローカル commit のみ。）

---

### Task 2: tool venv をローカル protos へ一時ブリッジ

**Files:**
- Modify: `pyproject.toml`（`[tool.uv.sources]` の `vstreamer-protos`）

**Interfaces:**
- Consumes: Task 1 のローカル protos（`../vstreamer-protos/python`）。
- Produces: tool venv に `Operand.trace_id`/`origin_ts` が import 可能な状態。

- [ ] **Step 1: uv source を path override に切替**

`pyproject.toml` の `[tool.uv.sources]` の該当行を一時的に置換:

```toml
vstreamer-protos = { path = "../vstreamer-protos/python" }
```

（元の `{ url = "...main-04910026..." }` は Task 10 で新リリース URL に戻す。）

- [ ] **Step 2: 反映**

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-tool
uv lock
uv sync
```

> 注: 素の `uv sync` は extra（audio/whisper/vroid2/voicevox/rvc/gui）を prune する。本計画のテストは base 依存のみで通るため作業中は base で進め、Task 10 で `uv sync --extra ...` 一式により protos 正式版と extra を同時復元する。

- [ ] **Step 3: tool venv で新フィールドを確認**

```bash
uv run python -c "from vstreamer_protos.commander.commander_pb2 import Operand; o=Operand(trace_id='t', origin_ts=2.0); print(o.trace_id, o.origin_ts)"
```
Expected: `t 2.0`

- [ ] **Step 4: コミット**

```bash
git add pyproject.toml uv.lock
git commit -m "build(dev): bridge vstreamer-protos to local path for trace fields"
```

---

## Phase 1 — tool: テレメトリ基盤

### Task 3: テレメトリレジストリ `vspeech/lib/telemetry.py`

**Files:**
- Create: `vspeech/lib/telemetry.py`
- Test: `tests/test_telemetry.py`（新規）

**Interfaces:**
- Produces:
  - module 変数 `telemetry: Telemetry`（シングルトン）
  - `Telemetry.record(stage: str, seconds: float) -> None`
  - `Telemetry.record_e2e(seconds: float) -> None`
  - `Telemetry.timer(stage: str)` … コンテキストマネージャ
  - `Telemetry.summary() -> dict[str, dict[str, float]]`（キー=段名と `"e2e"`、値=`{count,p50,p95,max,mean}`）
  - `Telemetry.log_summary() -> None`
  - `Telemetry.configure(enabled: bool, max_samples: int) -> None`
  - `Telemetry.reset() -> None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_telemetry.py`:

```python
from vspeech.lib.telemetry import Telemetry


def test_summary_percentiles():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    for v in [10.0, 20.0, 30.0, 40.0]:
        t.record("vc", v)
    s = t.summary()["vc"]
    assert s["count"] == 4
    assert s["p50"] == 25.0
    assert s["p95"] == 38.5
    assert s["max"] == 40.0


def test_max_samples_keeps_recent():
    t = Telemetry()
    t.configure(enabled=True, max_samples=2)
    for v in [1.0, 2.0, 3.0]:
        t.record("vc", v)
    s = t.summary()["vc"]
    assert s["count"] == 2
    assert s["max"] == 3.0
    assert s["p50"] == 2.5  # [2.0, 3.0] linear p50


def test_disabled_is_noop():
    t = Telemetry()
    t.configure(enabled=False, max_samples=5000)
    t.record("vc", 1.0)
    t.record_e2e(1.0)
    assert t.summary() == {}


def test_timer_records_stage():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    with t.timer("tts"):
        pass
    assert t.summary()["tts"]["count"] == 1


def test_record_e2e_present():
    t = Telemetry()
    t.configure(enabled=True, max_samples=5000)
    t.record_e2e(1.0)
    t.record_e2e(3.0)
    assert t.summary()["e2e"]["count"] == 2
    assert t.summary()["e2e"]["max"] == 3.0
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_telemetry.py -q`
Expected: FAIL（`ModuleNotFoundError: vspeech.lib.telemetry`）

- [ ] **Step 3: 実装**

`vspeech/lib/telemetry.py`:

```python
from contextlib import contextmanager
from math import ceil
from math import floor
from time import perf_counter

from vspeech.logger import logger


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = floor(rank)
    hi = ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "count": float(len(s)),
        "p50": _percentile(s, 50.0),
        "p95": _percentile(s, 95.0),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


class Telemetry:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.max_samples: int = 5000
        self._durations: dict[str, list[float]] = {}
        self._e2e: list[float] = []

    def configure(self, enabled: bool, max_samples: int) -> None:
        self.enabled = enabled
        self.max_samples = max_samples

    def reset(self) -> None:
        self._durations = {}
        self._e2e = []

    def _append(self, buf: list[float], seconds: float) -> None:
        buf.append(seconds)
        if len(buf) > self.max_samples:
            del buf[0]

    def record(self, stage: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._append(self._durations.setdefault(stage, []), seconds)

    def record_e2e(self, seconds: float) -> None:
        if not self.enabled:
            return
        self._append(self._e2e, seconds)

    @contextmanager
    def timer(self, stage: str):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, samples in self._durations.items():
            if samples:
                out[stage] = _stats(samples)
        if self._e2e:
            out["e2e"] = _stats(self._e2e)
        return out

    def log_summary(self) -> None:
        s = self.summary()
        if not s:
            return
        logger.info("=== telemetry summary (seconds) ===")
        for stage, m in s.items():
            logger.info(
                "%-14s n=%-5d p50=%.3f p95=%.3f max=%.3f mean=%.3f",
                stage,
                int(m["count"]),
                m["p50"],
                m["p95"],
                m["max"],
                m["mean"],
            )


telemetry = Telemetry()
```

- [ ] **Step 4: PASS 確認**

Run: `uv run pytest tests/test_telemetry.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/lib/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): in-memory metrics registry with shutdown summary"
```

---

### Task 4: `TelemetryConfig` 追加＋`cmd` で configure

**Files:**
- Modify: `vspeech/config.py`（`TelemetryConfig` 追加、`Config` に field 追加）
- Modify: `vspeech/main.py`（`cmd` で `telemetry.configure`）
- Test: `tests/test_telemetry_config.py`（新規）

**Interfaces:**
- Consumes: `telemetry`（Task 3）。
- Produces: `TelemetryConfig(enable: bool, max_samples: int, log_raw_e2e: bool, skew_warn_threshold: float)`、`Config.telemetry`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_telemetry_config.py`:

```python
from vspeech.config import Config
from vspeech.config import TelemetryConfig


def test_defaults():
    c = TelemetryConfig()
    assert c.enable is True
    assert c.max_samples == 5000
    assert c.log_raw_e2e is True
    assert c.skew_warn_threshold == 120.0


def test_config_has_telemetry():
    assert Config().telemetry.enable is True
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_telemetry_config.py -q`
Expected: FAIL（`ImportError: cannot import name 'TelemetryConfig'`）

- [ ] **Step 3: 実装**

`vspeech/config.py` に（他のサブConfig群の近く、`class Config` の前に）追加:

```python
class TelemetryConfig(BaseModel):
    enable: bool = True
    max_samples: int = 5000
    log_raw_e2e: bool = True
    skew_warn_threshold: float = 120.0
```

`class Config(BaseSettings)` のフィールド群（rvc の行の後など）に追加:

```python
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
```

`vspeech/main.py` の `cmd` 内、`configure_logger(config)` の直後に追加（import も先頭へ一行で）:

```python
from vspeech.lib.telemetry import telemetry
```
```python
    configure_logger(config)
    telemetry.configure(
        enabled=config.telemetry.enable, max_samples=config.telemetry.max_samples
    )
```

- [ ] **Step 4: PASS 確認**

Run: `uv run pytest tests/test_telemetry_config.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/config.py vspeech/main.py tests/test_telemetry_config.py
git commit -m "feat(telemetry): TelemetryConfig and configure at startup"
```

---

## Phase 2 — tool: トレース伝搬と E2E

### Task 5: trace フィールド＋encode/decode＋wire ラウンドトリップ

**Files:**
- Modify: `vspeech/shared_context.py`（`WorkerOutput`/`WorkerInput` に trace フィールド、`encode_trace`/`decode_trace`、`to_pb`/`from_command`/`from_input`）
- Test: `tests/test_trace_propagation.py`（新規）

**Interfaces:**
- Consumes: protobuf `Operand.trace_id`/`origin_ts`（Task 1/2）。
- Produces:
  - `WorkerOutput.trace_id: str`（既定 `""`）, `WorkerOutput.origin_ts: float`（既定 `0.0`）
  - `WorkerInput.trace_id: str`, `WorkerInput.origin_ts: float`
  - `encode_trace(operand: Operand, trace_id: str, origin_ts: float) -> None`
  - `decode_trace(operand: Operand) -> tuple[str, float]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_trace_propagation.py`:

```python
from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def _output_with_trace():
    out = WorkerOutput(
        input_id=uuid4(),
        followings=[[EventAddress(event=EventType.vc, remote="//r")]],
        sound=SoundOutput(
            data=b"abc", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        text="hi",
    )
    out.trace_id = "trace-xyz"
    out.origin_ts = 1234.5
    return out


def test_to_pb_encodes_trace():
    cmd = _output_with_trace().to_pb("//r")
    assert cmd.operand.trace_id == "trace-xyz"
    assert cmd.operand.origin_ts == 1234.5


def test_from_command_decodes_trace():
    cmd = _output_with_trace().to_pb("//r")
    inputs = WorkerInput.from_command(cmd)
    assert inputs[0].trace_id == "trace-xyz"
    assert inputs[0].origin_ts == 1234.5


def test_from_input_copies_trace():
    out = _output_with_trace()
    inputs = WorkerInput.from_command(out.to_pb("//r"))
    copied = WorkerOutput.from_input(inputs[0])
    assert copied.trace_id == "trace-xyz"
    assert copied.origin_ts == 1234.5


def test_absent_trace_defaults():
    out = WorkerOutput(
        input_id=uuid4(),
        followings=[[EventAddress(event=EventType.subtitle, remote="//r")]],
        text="t",
    )
    inputs = WorkerInput.from_command(out.to_pb("//r"))
    assert inputs[0].trace_id == ""
    assert inputs[0].origin_ts == 0.0
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_trace_propagation.py -q`
Expected: FAIL（`WorkerOutput` に `trace_id` 属性が無い等）

- [ ] **Step 3: 実装**

`vspeech/shared_context.py` の import に追加（一行ずつ）:
```python
from vstreamer_protos.commander.commander_pb2 import Operand
```
（既に import 済みなら不要。`Operand` は `to_pb` で使用済みのため通常 import 済み。）

シーム関数を module レベルに追加（`WorkerOutput` 定義の前後どこでも可）:
```python
def encode_trace(operand: Operand, trace_id: str, origin_ts: float) -> None:
    operand.trace_id = trace_id
    operand.origin_ts = origin_ts


def decode_trace(operand: Operand) -> tuple[str, float]:
    return operand.trace_id, operand.origin_ts
```

`@dataclass class WorkerOutput` に2フィールド追加（既存 `sound`/`text` の後）:
```python
    trace_id: str = ""
    origin_ts: float = 0.0
```

`WorkerOutput.to_pb` を更新（Operand 構築後に encode）:
```python
    def to_pb(self, remote: str) -> Command:
        events = self.events(remote)
        operand = Operand(
            text=self.text,
            sound=self.sound.to_pb() if self.sound else None,
        )
        encode_trace(operand, self.trace_id, self.origin_ts)
        return Command(
            chains=[
                OperationChain(operations=[f.to_pb() for f in fs]) for fs in events
            ],
            operand=operand,
        )
```

`WorkerOutput.from_input` に trace コピーを追加:
```python
    @classmethod
    def from_input(cls, worker_input: "WorkerInput"):
        return cls(
            input_id=worker_input.input_id,
            followings=worker_input.following_events,
            trace_id=worker_input.trace_id,
            origin_ts=worker_input.origin_ts,
        )
```

`class WorkerInput(BaseModel)` に2フィールド追加（既存フィールド群に）:
```python
    trace_id: str = ""
    origin_ts: float = 0.0
```

`WorkerInput.from_command` で decode して各 WorkerInput に設定:
```python
    @classmethod
    def from_command(cls, command: Command) -> list["WorkerInput"]:
        input_id = uuid4()
        trace_id, origin_ts = decode_trace(command.operand)
        events = command_to_events(command)
        first_event_maps = get_first_event_map(events)
        return [
            WorkerInput(
                input_id=input_id,
                current_event=first_event,
                following_events=following_events,
                text=command.operand.text,
                sound=SoundInput.model_validate(command.operand.sound),
                file_path=command.operand.file_path,
                filters=list(command.operand.filters),
                trace_id=trace_id,
                origin_ts=origin_ts,
            )
            for first_event, following_events in first_event_maps.items()
        ]
```

注: `WorkerInput.from_output`（ローカルディスパッチ経路）も trace を運ぶ必要がある。`from_output` の `WorkerInput(...)` に `trace_id=output.trace_id, origin_ts=output.origin_ts` を追加すること。

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest tests/test_trace_propagation.py tests/test_event_chains.py tests/test_worker_input.py -q`
Expected: PASS（新4件＋既存 green。trace は Operand 直下で queries/Params/EventAddress 等価に非干渉）

- [ ] **Step 5: コミット**

```bash
git add vspeech/shared_context.py tests/test_trace_propagation.py
git commit -m "feat(telemetry): propagate trace_id/origin_ts through WorkerInput/Output and wire"
```

---

### Task 6: recording が原点を採番

**Files:**
- Modify: `vspeech/worker/recording.py:143-150`（trace 採番）
- Test: `tests/test_recording_trace.py`（新規）

**Interfaces:**
- Consumes: `WorkerOutput.trace_id`/`origin_ts`（Task 5）。
- Produces: recording の `WorkerOutput` に非空 `trace_id` と正の `origin_ts`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_recording_trace.py`:

```python
from vspeech.config import RecordingConfig
from vspeech.worker.recording import build_recording_output


def test_recording_output_has_trace():
    cfg = RecordingConfig()
    out = build_recording_output(cfg, frames=b"abc")
    assert out.trace_id != ""
    assert out.origin_ts > 0.0
    assert out.sound is not None
    assert out.sound.data == b"abc"
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_recording_trace.py -q`
Expected: FAIL（`ImportError: cannot import name 'build_recording_output'`）

- [ ] **Step 3: 実装（送出ロジックを純関数に抽出して採番）**

`vspeech/worker/recording.py` の import に追加（一行ずつ）:
```python
from time import time
from uuid import uuid4

from vspeech.shared_context import WorkerOutput
```
（`WorkerOutput` は import 済み。`time`/`uuid4` を追加。）

新しい純関数を追加:
```python
def build_recording_output(config: RecordingConfig, frames: bytes) -> WorkerOutput:
    worker_output = WorkerOutput.from_routes_list(config.routes_list)
    worker_output.trace_id = uuid4().hex
    worker_output.origin_ts = time()
    worker_output.sound = SoundOutput(
        data=frames,
        rate=config.rate,
        format=config.format,
        channels=config.channels,
    )
    return worker_output
```

`recording_worker` の本体（現 143-150）をこの関数呼び出しに置換:
```python
                worker_output = build_recording_output(rec_config, frames)
                out_queue.put_nowait(worker_output)
```

- [ ] **Step 4: PASS 確認**

Run: `uv run pytest tests/test_recording_trace.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/recording.py tests/test_recording_trace.py
git commit -m "feat(telemetry): stamp trace_id/origin_ts at recording origin"
```

---

### Task 7: playback が真E2Eを算出（＋skew警告）

**Files:**
- Modify: `vspeech/worker/playback.py`（E2E算出を純関数化して呼ぶ）
- Test: `tests/test_playback_e2e.py`（新規）

**Interfaces:**
- Consumes: `telemetry`（Task 3）、`WorkerInput.trace_id`/`origin_ts`（Task 5）、`config.telemetry`（Task 4）。
- Produces: `record_playback_e2e(speech: WorkerInput, now: float, cfg: TelemetryConfig) -> float | None`（記録した e2e 秒、skew/未設定なら None）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_playback_e2e.py`:

```python
from uuid import uuid4

from vspeech.config import EventType
from vspeech.config import SampleFormat
from vspeech.config import TelemetryConfig
from vspeech.lib.telemetry import telemetry
from vspeech.shared_context import EventAddress
from vspeech.shared_context import SoundInput
from vspeech.shared_context import WorkerInput
from vspeech.worker.playback import record_playback_e2e


def _speech(origin_ts: float) -> WorkerInput:
    return WorkerInput(
        input_id=uuid4(),
        current_event=EventAddress(event=EventType.playback),
        following_events=[],
        text="",
        sound=SoundInput(
            data=b"00", rate=16000, format=SampleFormat.INT16, channels=1
        ),
        file_path="",
        filters=[],
        trace_id="abc",
        origin_ts=origin_ts,
    )


def test_e2e_recorded():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=100.0), now=101.5, cfg=cfg)
    assert e2e == 1.5
    assert telemetry.summary()["e2e"]["count"] == 1


def test_skew_negative_not_recorded():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=200.0), now=100.0, cfg=cfg)
    assert e2e is None
    assert "e2e" not in telemetry.summary()


def test_no_origin_skipped():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    cfg = TelemetryConfig()
    e2e = record_playback_e2e(_speech(origin_ts=0.0), now=100.0, cfg=cfg)
    assert e2e is None
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_playback_e2e.py -q`
Expected: FAIL（`ImportError: cannot import name 'record_playback_e2e'`）

- [ ] **Step 3: 実装**

`vspeech/worker/playback.py` の import に追加（一行ずつ）:
```python
from vspeech.config import TelemetryConfig
from vspeech.lib.telemetry import telemetry
```

純関数を追加（module レベル）:
```python
def record_playback_e2e(
    speech: WorkerInput, now: float, cfg: TelemetryConfig
) -> float | None:
    if speech.origin_ts <= 0.0:
        return None
    e2e = now - speech.origin_ts
    if e2e < 0.0 or e2e > cfg.skew_warn_threshold:
        logger.warning(
            "clock skew suspected: e2e=%.3fs trace=%s (NTP同期を確認)",
            e2e,
            speech.trace_id,
        )
        return None
    telemetry.record_e2e(e2e)
    if cfg.log_raw_e2e:
        logger.info("e2e trace=%s %.3fs", speech.trace_id, e2e)
    return e2e
```

`pyaudio_playback_worker` の再生完了直後（現 [playback.py:137-141](../../../vspeech/worker/playback.py#L137-L141) の `logger.debug("playback end")` の後）に算出を挿入。`config: PlaybackConfig` しか渡っていないため、E2E に必要な `TelemetryConfig` は呼び出し側 `playback_worker` から `context.config.telemetry` を引数で渡すよう小調整するか、`pyaudio_playback_worker` のシグネチャに `telemetry_config: TelemetryConfig` を追加して `playback_worker` から渡す。挿入:
```python
                from time import time as _now

                record_playback_e2e(speech, now=_now(), cfg=telemetry_config)
```
（`speech` は再生対象の `WorkerInput`。`telemetry_config` は新引数。）

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest tests/test_playback_e2e.py -q && uv run pytest -q`
Expected: PASS（新3件＋全体 green）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/playback.py tests/test_playback_e2e.py
git commit -m "feat(telemetry): compute true E2E at playback with skew guard"
```

---

## Phase 3 — tool: 各段計測＋サマリ出力＋ドキュメント

### Task 8: 各ワーカーに処理時間計測を挿入

**Files:**
- Modify: `vspeech/worker/transcription.py`, `vspeech/worker/translation.py`, `vspeech/worker/tts.py`, `vspeech/worker/vc.py`, `vspeech/worker/playback.py`
- Test: `tests/test_vc_telemetry.py`（代表として vc を検証）

**Interfaces:**
- Consumes: `telemetry.timer` / `telemetry.record`（Task 3）。
- Produces: 各段の処理時間が `telemetry` に段名（`"transcription"`/`"translation"`/`"tts"`/`"vc"`/`"playback"`）で記録される。

- [ ] **Step 1: 失敗するテストを書く（vc を代表に）**

`tests/test_vc_telemetry.py`:

```python
from vspeech.lib.telemetry import telemetry
from vspeech.worker.vc import record_vc_elapsed


def test_vc_records_duration():
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    record_vc_elapsed(0.25)
    s = telemetry.summary()["vc"]
    assert s["count"] == 1
    assert s["max"] == 0.25
```

- [ ] **Step 2: 失敗確認**

Run: `uv run pytest tests/test_vc_telemetry.py -q`
Expected: FAIL（`ImportError: cannot import name 'record_vc_elapsed'`）

- [ ] **Step 3: 実装**

各ファイルの import に追加（一行ずつ）:
```python
from vspeech.lib.telemetry import telemetry
```

**vc.py**: 既存の `vc_start_time`/`vc_end_time`（`time.time()`、[vc.py:129,169-170](../../../vspeech/worker/vc.py#L129-L170)）を `perf_counter` に置換し、小ヘルパー＋記録を追加。
```python
from time import perf_counter
```
module レベルに:
```python
def record_vc_elapsed(seconds: float) -> None:
    telemetry.record("vc", seconds)
    logger.info("rvc elapsed time: %s", seconds)
```
本体: `vc_start_time = perf_counter()`（129行）、`vc_end_time = perf_counter()`（169行）にし、170行の `logger.info(...)` を:
```python
            record_vc_elapsed(vc_end_time - vc_start_time)
```

**transcription.py**: 各 `transcript_worker_*` の `model.transcribe`/認識呼び出しを `with telemetry.timer("transcription"):` で囲む（whisper の `to_thread(model.transcribe...)`、google の `recognize`、ami の `client.post` を含む処理ブロック）。

**translation.py**: 翻訳呼び出し（`logger.debug("translating...")` 以降の翻訳処理）を `with telemetry.timer("translation"):` で囲む。

**tts.py**: vr2/voicevox の生成（`logger.debug("voice generating...")` 〜 `voice generated`）を `with telemetry.timer("tts"):` で囲む。

**playback.py**: `await output_stream.playback(...)`（[playback.py:133-136](../../../vspeech/worker/playback.py#L133-L136)）を `with telemetry.timer("playback"):` で囲む。

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest tests/test_vc_telemetry.py -q && uv run pytest -q`
Expected: PASS（vc 記録＋全体 green）

- [ ] **Step 5: コミット**

```bash
git add vspeech/worker/ tests/test_vc_telemetry.py
git commit -m "feat(telemetry): per-stage timers in transcription/translation/tts/vc/playback"
```

---

### Task 9: 終了時サマリ出力＋ドキュメント

**Files:**
- Modify: `vspeech/main.py`（`vspeech_coro` の `finally` で `telemetry.log_summary()`）
- Modify: `config.toml.example`（`[telemetry]` 節＋NTP注記）
- Test: `tests/test_shutdown_summary.py`（新規）

**Interfaces:**
- Consumes: `telemetry.log_summary`（Task 3）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shutdown_summary.py`:

```python
from vspeech.lib.telemetry import telemetry


def test_log_summary_emits(caplog):
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    telemetry.record("vc", 0.5)
    import logging

    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert any("telemetry summary" in r.message for r in caplog.records)


def test_log_summary_empty_silent(caplog):
    telemetry.reset()
    telemetry.configure(enabled=True, max_samples=100)
    import logging

    with caplog.at_level(logging.INFO):
        telemetry.log_summary()
    assert not any("telemetry summary" in r.message for r in caplog.records)
```

- [ ] **Step 2: 失敗確認（log_summary は既存だが caplog 連携の確認）**

Run: `uv run pytest tests/test_shutdown_summary.py -q`
Expected: 1件目 PASS（log_summary は Task 3 で実装済み）。2件目も PASS。両方 green ならこのテストは回帰用。FAIL する場合は `log_summary` の空時 return を確認。

- [ ] **Step 3: `main.py` でサマリ出力＋example 更新**

`vspeech/main.py` の `vspeech_coro` を `try/finally` 化:
```python
async def vspeech_coro(config: Config):
    context = SharedContext(config=config)
    try:
        try:
            async with TaskGroup() as tg:
                ...  # 既存の create_*_task 群（変更なし）
        except* WorkerShutdown as eg:
            for e in eg.exceptions:
                logger.warning("workers shutdown: %s", e.args)
                logger.debug("".join(format_exception(e)))
    finally:
        telemetry.log_summary()
```
（`from vspeech.lib.telemetry import telemetry` は Task 4 で追加済み。未追加なら一行で追加。）

`config.toml.example` の末尾付近に追加:
```toml
[telemetry]
enable = true
max_samples = 5000
log_raw_e2e = true
# 発話→再生の真E2Eはマシン間で time.time() を比較するため、各マシンの
# 時計をNTPで同期しておくこと。未同期だと負値/極端値となり skew 警告が出る。
skew_warn_threshold = 120.0
```

- [ ] **Step 4: PASS 確認＋回帰**

Run: `uv run pytest -q`
Expected: PASS（全体 green）

- [ ] **Step 5: コミット**

```bash
git add vspeech/main.py config.toml.example tests/test_shutdown_summary.py
git commit -m "feat(telemetry): emit summary on shutdown; document telemetry config + NTP"
```

---

## Phase 4 — 正式リリースと pin 復帰

### Task 10: protos 正式リリース＋tool の pin 更新

**Files:**
- protos repo: push `feat/trace-fields` → main（CI リリース）
- Modify: `pyproject.toml`（`[tool.uv.sources]` を新リリース URL へ）、`uv.lock`

**Interfaces:**
- Produces: 公開 wheel に新フィールドが入り、tool が path override 無しで動く状態。

> ⚠️ **公開を伴う操作**: protos の main への push は GitHub リリース（公開）を発生させる。実行前にユーザ確認を取ること。

- [ ] **Step 1: protos を main へ反映（要ユーザ確認）**

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-protos
git checkout main && git merge --no-ff feat/trace-fields -m "feat: trace fields for telemetry E2E"
git push origin main
```
→ GitHub Actions CI が wheel をビルドし新リリース（tag `main-<sha>`）。リリースページで新しい wheel URL を控える。

- [ ] **Step 2: tool の pin を新リリース URL へ戻す**

`pyproject.toml` の `[tool.uv.sources]` を path override から新リリース URL へ:
```toml
vstreamer-protos = { url = "https://github.com/sondeko143/vstreamer-protos/releases/download/main-<new-sha>/vstreamer_protos-0.1.0-py3-none-any.whl" }
```

```bash
cd /c/Users/<USER>/vstreamer/vstreamer-tool
uv lock
# extra も同時に復元（Task 2 の素の uv sync で prune されたため）
uv sync --extra audio --extra whisper --extra vroid2 --extra voicevox --extra rvc --extra gui
```

- [ ] **Step 3: 全ゲート green を確認**

```bash
uv run pytest -q
uv run ruff format .
uv run ruff check .
uv run ty check
```
Expected: 全テスト PASS、ruff/ty とも `All checks passed!`。

- [ ] **Step 4: コミット**

```bash
git add pyproject.toml uv.lock
git commit -m "build: pin vstreamer-protos release with trace fields"
```

---

## Self-Review（計画作成者によるチェック結果）

**1. Spec coverage:**
- §4 protos 前提 → Task 1（編集/再生成）, Task 10（リリース/pin）。
- §5.1 レジストリ → Task 3。
- §5.2 各段計測 → Task 8。
- §5.3 trace 伝搬＋E2E → Task 5（フィールド/encode/decode/round-trip）, Task 6（原点）, Task 7（E2E＋skew）。
- §5.4 終了時サマリ → Task 9。
- §5.5 TelemetryConfig → Task 4（＋example は Task 9）。
- §7 テスト → 各タスクの TDD ＋既存 `test_event_chains`/`test_worker_input` 回帰（Task 5 Step4）。
- §8 リスク（順序 protos→wheel→tool）→ Phase 構成と Task 2 の dev ブリッジで担保。

**2. Placeholder scan:** TODO/TBD なし。各コードステップに実コード提示。Task 8 の各ワーカー timer 挿入のみ「処理ブロックを `with timer` で囲む」と記述（具体行は各ファイル参照を明記、代表 vc は完全コード）。

**3. Type consistency:** `telemetry.record(stage,seconds)` / `record_e2e(seconds)` / `timer(stage)` / `summary()->dict[str,dict[str,float]]` / `configure(enabled,max_samples)` を全タスクで一致使用。`TelemetryConfig(enable,max_samples,log_raw_e2e,skew_warn_threshold)`、`WorkerOutput/Input.trace_id:str`/`origin_ts:float`、`encode_trace(operand,trace_id,origin_ts)`/`decode_trace(operand)->tuple[str,float]`、`build_recording_output(config,frames)`、`record_playback_e2e(speech,now,cfg)->float|None`、`record_vc_elapsed(seconds)` を定義タスクと利用タスクで整合。
