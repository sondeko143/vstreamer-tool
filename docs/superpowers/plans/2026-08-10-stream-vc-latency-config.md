# ストリーミング VC の latency 設定化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

ADR: [0071](../../adr/0071-stream-vc-device-latency-config.md)（Proposed → Task 3 で Accepted へ昇格）

**Goal:** `[stream_vc]` に `input_latency` / `output_latency` を追加し、いま `"low"` にハードコードされているデバイス latency を入出力別に設定できるようにする。

**Architecture:** 設定値は単位変換せず sounddevice へ素通しする(float は秒 = PortAudio の `suggestedLatency` の単位)。型は `Literal["low", "high"] | float(gt=0)`、既定は両方 `"low"` = 現在のハードコード値なので既存 config の挙動は不変。出力側の open 関数は `playback.py` の 1 つを `consumer.py` が再利用しているので、変更箇所は入力 1 + 出力 1 の計 2 関数で足りる。あわせて open 時に PortAudio が実際に返した `stream.latency` をログに出す。

**Tech Stack:** Python 3.14 / pydantic v2 / sounddevice(PortAudio) / pytest(asyncio_mode=auto) / ruff / ty / uv

## Global Constraints

- Python は 3.14 のみ(`requires-python = ">=3.14,<3.15"`)。
- pydantic は v2 API のみ。v1 API(`parse_obj` / `.dict()` / `root_validator` / `json_encoders` 等)を使わない。
- コード内のコメントと docstring は**英語**(ADR-0064)。ただし `config.py` の `description=` とユーザーが読むログ/例外文言は**日本語**のまま。
- import は 1 行 1 つ(ruff `force-single-line = true`)。
- 既定値は現在のハードコード値 `"low"` と一致させる。既存 config の挙動を変えない。
- 触ったファイルに日本語コメントが残っていたら同じ変更で英語へ翻訳する(CLAUDE.md)。本 plan が触る 4 ファイルは既に英語なので、実際には該当しない見込み。
- ドキュメント(ADR / spec / `config.toml.example`)は日本語。

## File Structure

| ファイル | 役割 | 変更 |
|---|---|---|
| `vspeech/config.py` | 設定スキーマの唯一の真実 | `DeviceLatency` 型エイリアス + `StreamVcConfig` に 2 フィールド追加 |
| `vspeech/stream_vc/capture.py` | マイク入力ストリームの open | `latency=` を設定値に、granted latency をログ |
| `vspeech/stream_vc/playback.py` | 出力ストリームの open(`consumer.py` も再利用) | 同上 |
| `tests/test_stream_vc_config.py` | `[stream_vc]` の形・既定・拒否のテスト | latency の既定/受理/独立性/TOML/拒否を追加 |
| `tests/test_main.py` | 実プロセス起動時の config エラー報告 (ADR-0068) | 不正 latency の parametrize ケースを 1 件追加 |
| `tests/test_stream_vc_capture.py` | capture のテスト | 入力 open の passthrough とログを追加 |
| `tests/test_stream_vc_playback.py` | playback のテスト | 出力 open の passthrough とログを追加 |
| `config.toml.example` | 全設定の解説 | `[stream_vc]` に 2 項目を追記 |
| `docs/adr/0071-*.md` | 決定層 | Status を Accepted へ |

`consumer.py` は `playback.open_stream_vc_output_stream` を import して使っているので変更不要。この「出力 open は 1 箇所」という性質は本 plan の前提なので、Task 2 で確認ステップを置く。

---

### Task 1: 設定スキーマに input_latency / output_latency を足す

**Files:**
- Modify: `vspeech/config.py`（import 部 6-8 行目付近、`StreamVcConfig` の `output_device_index` 直後）
- Test: `tests/test_stream_vc_config.py`
- Test: `tests/test_main.py:76-82`（ADR-0068 の per-problem レポートの parametrize に 1 ケース追加）

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces:
  - `vspeech.config.DeviceLatency` — PEP 695 型エイリアス。`Literal["low", "high"] | Annotated[float, Field(gt=0)]`
  - `StreamVcConfig.input_latency: DeviceLatency`（既定 `"low"`）
  - `StreamVcConfig.output_latency: DeviceLatency`（既定 `"low"`）
  - Task 2 はこの 2 属性を読む。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_stream_vc_config.py` の末尾に追記する。ファイル先頭には既に `import io` と `from vspeech.config import Config` / `StreamVcConfig` があるので、import の追加は不要。

```python
def test_stream_vc_latency_defaults_to_low():
    """The default equals the value that used to be hardcoded, so existing configs do
    not change behaviour (ADR-0071)."""
    c = StreamVcConfig()
    assert c.input_latency == "low"
    assert c.output_latency == "low"


def test_stream_vc_latency_accepts_high_and_explicit_seconds():
    """PortAudio takes either of its two device defaults or an arbitrary
    suggestedLatency in seconds; all three have to survive validation."""
    c = StreamVcConfig.model_validate({"input_latency": "high", "output_latency": 0.02})
    assert c.input_latency == "high"
    assert c.output_latency == 0.02


def test_stream_vc_latency_sides_are_independent():
    """Input and output are different devices (different machines once the role is
    split), so raising one must not move the other -- the reason ADR-0071 rejected a
    single shared field."""
    c = StreamVcConfig.model_validate({"output_latency": "high"})
    assert c.input_latency == "low"
    assert c.output_latency == "high"


def test_stream_vc_latency_parses_from_toml():
    toml_text = b"""
[stream_vc]
input_latency = "high"
output_latency = 0.05
"""
    f = io.BytesIO(toml_text)
    f.name = "config.toml"
    c = Config.read_config_from_file(f)
    assert c.stream_vc.input_latency == "high"
    assert c.stream_vc.output_latency == 0.05


def test_stream_vc_latency_rejects_unknown_string_and_non_positive():
    """Bad values fail at config load, which ADR-0068 already routes into the same
    per-problem report preflight uses -- hence no dedicated preflight check."""
    import pytest
    from pydantic import ValidationError

    # A typo must not silently fall through to a float coercion.
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"input_latency": "lowest"})
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"output_latency": 0.0})
    with pytest.raises(ValidationError):
        StreamVcConfig.model_validate({"input_latency": -0.01})


def test_stream_vc_latency_survives_export_to_toml_round_trip():
    import toml as toml_lib

    c = Config()
    c.stream_vc.output_latency = 0.05
    reloaded = toml_lib.loads(c.export_to_toml())
    assert reloaded["stream_vc"]["input_latency"] == "low"
    assert reloaded["stream_vc"]["output_latency"] == 0.05
```

- [ ] **Step 2: テストが落ちることを確認する**

```sh
uv run pytest tests/test_stream_vc_config.py -v
```

Expected: 新規 6 件が FAIL（`AttributeError: 'StreamVcConfig' object has no attribute 'input_latency'`、および `extra="forbid"` 由来の `ValidationError` — `Config` は top-level のみ forbid なので `[stream_vc]` 側は未知キーを無視する可能性がある。どちらの落ち方でも良い）。既存のテストは PASS のまま。

- [ ] **Step 3: 型エイリアスと import を足す**

`vspeech/config.py` の `from typing import Literal`（8 行目）の直前に 1 行足す。ruff が import を並べ替えるので、アルファベット順で `Any` の後・`Literal` の前になる。

```python
from typing import Annotated
```

`type Anchor = ...`（21 行目）の直後に型エイリアスを足す。

```python
# The device latency requested of PortAudio (ADR-0071). "low"/"high" are the device's
# own defaults; a float is an explicit suggestedLatency **in seconds** -- sounddevice's
# own unit, so the value is handed over without conversion.
type DeviceLatency = Literal["low", "high"] | Annotated[float, Field(gt=0)]
```

- [ ] **Step 4: StreamVcConfig にフィールドを足す**

`vspeech/config.py` の `output_device_index: int | None = Field(default=None)` の直後、`transport_type` の直前に挿入する。

```python
    # Handed to sounddevice untouched (ADR-0071). The default matches the value that
    # used to be hardcoded, so an existing config opens exactly as before. Input and
    # output are separate fields because they are separate devices -- separate machines
    # once role is producer/consumer (ADR-0055).
    input_latency: DeviceLatency = Field(
        default="low",
        description="入力(マイク)ストリームがデバイスへ要求する latency。"
        '"low"/"high" はデバイス既定の低/高遅延、数値は秒での明示指定(例 0.02 = 20ms)。'
        "低すぎると overflow が止まらず、高すぎると block_ms を詰めても取り返せない。"
        "実際に得られた値は open 時のログに出る(要求値は保証されない)",
    )
    output_latency: DeviceLatency = Field(
        default="low",
        description="出力(再生)ストリームがデバイスへ要求する latency。"
        '"low"/"high" はデバイス既定の低/高遅延、数値は秒での明示指定(例 0.02 = 20ms)。'
        "低すぎると underflow が止まらず、高すぎると block_ms を詰めても取り返せない。"
        "実際に得られた値は open 時のログに出る(要求値は保証されない)",
    )
```

- [ ] **Step 5: テストが通ることを確認する**

```sh
uv run pytest tests/test_stream_vc_config.py -v
```

Expected: 全件 PASS。

もし PEP 695 の `type` エイリアスと `Annotated[float, Field(gt=0)]` の組み合わせを pydantic が解決できずに落ちた場合（`PydanticUserError`、または `gt=0` が効かず `-0.01` を受理してしまう場合）、エイリアスをやめて 2 フィールドへ直接書き下す。`description=` は Step 4 に書いたものをそのままコピーし、注釈だけ差し替える：

```python
    input_latency: Literal["low", "high"] | Annotated[float, Field(gt=0)] = Field(
        default="low",
        description=...,  # Step 4 の input_latency の description をそのまま
    )
    output_latency: Literal["low", "high"] | Annotated[float, Field(gt=0)] = Field(
        default="low",
        description=...,  # Step 4 の output_latency の description をそのまま
    )
```

この場合は `type DeviceLatency = ...` の行を削除する（Task 2 は属性を読むだけなので影響なし）。

- [ ] **Step 6: 不正値が ADR-0068 のレポートに乗ることを確認する**

`tests/test_main.py:76-82` の parametrize に 1 ケース足す。

```python
        pytest.param(
            '[stream_vc]\ninput_latency = "lowest"\n',
            "input_latency",
            id="stream-vc-bad-latency",
        ),
```

このケースだけ実装後に足す（他のテストと違い red から始めない）。理由: このテストは `python -m vspeech` を実サブプロセスで起動する。実装前は `StreamVcConfig` が未知キーを無視する（`extra="forbid"` は `Config` の top level だけ）ので、config は妥当と見なされてパイプラインが起動し、失敗ではなく 120 秒の `TimeoutExpired` になる。2 分待って得られる情報が無いので、実装後に緑を確認する形にする。

```sh
uv run pytest "tests/test_main.py::test_a_bad_config_file_is_reported_not_dumped_as_a_traceback" -v
```

Expected: 4 ケースとも PASS。新ケースは exit code 1、`Traceback` なし、`起動中止` と `input_latency` が出力に含まれる。

- [ ] **Step 7: 型チェックとフォーマット**

```sh
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected: 本タスクの変更に起因する新規の指摘なし（`uv audit` の torch 由来など既存の accepted 事項は対象外）。

- [ ] **Step 8: コミット**

```bash
git add vspeech/config.py tests/test_stream_vc_config.py tests/test_main.py
git commit -m "feat(stream-vc): add input_latency / output_latency to [stream_vc] (ADR-0071)"
```

---

### Task 2: 設定値をストリームの open へ渡し、実際の latency をログに出す

**Files:**
- Modify: `vspeech/stream_vc/capture.py:58-70`（`open_stream_vc_input_stream`）
- Modify: `vspeech/stream_vc/playback.py:42-55`（`open_stream_vc_output_stream`）
- Test: `tests/test_stream_vc_capture.py`
- Test: `tests/test_stream_vc_playback.py`

**Interfaces:**
- Consumes: Task 1 の `StreamVcConfig.input_latency` / `StreamVcConfig.output_latency`
- Produces: シグネチャは変えない。
  - `open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream`
  - `open_stream_vc_output_stream(config: StreamVcConfig, sample_rate: int) -> sd.RawOutputStream`

- [ ] **Step 1: 出力 open が 1 箇所しかないことを確認する**

```sh
grep -rn "RawOutputStream(" vspeech/
```

Expected: `vspeech/stream_vc/playback.py` の 1 件と `vspeech/worker/playback.py` の 1 件のみ。`vspeech/stream_vc/consumer.py` には**構築が無い**（`from vspeech.stream_vc.playback import open_stream_vc_output_stream` を使っている）。`vspeech/worker/playback.py` は発話系＝今回の非ゴールなので触らない。

- [ ] **Step 2: 失敗するテストを書く（入力側）**

`tests/test_stream_vc_capture.py` の末尾に追記する。先頭の import に以下 2 行を足す（`import logging` / `import numpy as np` / `import pytest` は既にある。ruff が並べ替える）。

```python
from vspeech.config import StreamVcConfig
from vspeech.lib.audio import DeviceInfo
```

追記する本体:

```python
class _RecordingInputStream:
    """Records the kwargs sounddevice would have been constructed with.

    `latency` is a passthrough, so the assertion has to happen at the sd boundary --
    there is nothing downstream to observe it on.
    """

    # Declared at class level so it is a known attribute (ty) rather than one that
    # springs into existence on first construction.
    kwargs: dict[str, object] = {}
    # What PortAudio granted, which is not required to equal what was requested.
    latency = 0.032

    def __init__(self, **kwargs) -> None:
        _RecordingInputStream.kwargs = kwargs

    def start(self) -> None:
        pass


def _fake_input_device() -> DeviceInfo:
    return DeviceInfo(
        host_api=0,
        max_input_channels=2,
        max_output_channels=0,
        name="Fake Mic",
        index=7,
    )


def _patch_input_open(monkeypatch) -> None:
    from vspeech.stream_vc import capture

    monkeypatch.setattr(
        capture, "resolve_stream_vc_input_device", lambda config: _fake_input_device()
    )
    monkeypatch.setattr(capture.sd, "RawInputStream", _RecordingInputStream)


def test_open_input_stream_requests_configured_latency(monkeypatch):
    """The configured value reaches sounddevice unconverted -- a float is seconds,
    PortAudio's own unit (ADR-0071)."""
    from vspeech.stream_vc import capture

    _patch_input_open(monkeypatch)
    capture.open_stream_vc_input_stream(StreamVcConfig(input_latency=0.05), hop=160)
    assert _RecordingInputStream.kwargs["latency"] == 0.05


def test_open_input_stream_defaults_to_low(monkeypatch):
    """No setting = the value that used to be hardcoded."""
    from vspeech.stream_vc import capture

    _patch_input_open(monkeypatch)
    capture.open_stream_vc_input_stream(StreamVcConfig(), hop=160)
    assert _RecordingInputStream.kwargs["latency"] == "low"


def test_open_input_stream_uses_input_latency_not_output(monkeypatch):
    """The output setting must not leak into the input stream."""
    from vspeech.stream_vc import capture

    _patch_input_open(monkeypatch)
    config = StreamVcConfig(input_latency="low", output_latency="high")
    capture.open_stream_vc_input_stream(config, hop=160)
    assert _RecordingInputStream.kwargs["latency"] == "low"


def test_open_input_stream_logs_requested_and_granted_latency(caplog, monkeypatch):
    """Reading the granted value is the point: "low" resolves to wildly different
    numbers per host API, and it cannot be read off the requested value."""
    from vspeech.stream_vc import capture

    _patch_input_open(monkeypatch)
    with caplog.at_level(logging.INFO):
        capture.open_stream_vc_input_stream(StreamVcConfig(), hop=160)
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Fake Mic" in messages  # the device line still names the device
    assert "low" in messages  # requested
    assert "0.032" in messages  # granted
```

- [ ] **Step 3: 失敗するテストを書く（出力側）**

`tests/test_stream_vc_playback.py` の末尾に追記する。ファイル先頭の import に以下を足す（既にあるものは重複させない）。

```python
import logging

from vspeech.config import StreamVcConfig
from vspeech.lib.audio import DeviceInfo
```

追記する本体:

```python
class _RecordingOutputStream:
    """Records the kwargs sounddevice would have been constructed with (mirrors the
    input-side fake in test_stream_vc_capture.py)."""

    # Declared at class level so it is a known attribute (ty) rather than one that
    # springs into existence on first construction.
    kwargs: dict[str, object] = {}
    # What PortAudio granted, which is not required to equal what was requested.
    latency = 0.048

    def __init__(self, **kwargs) -> None:
        _RecordingOutputStream.kwargs = kwargs

    def start(self) -> None:
        pass


def _fake_output_device() -> DeviceInfo:
    return DeviceInfo(
        host_api=0,
        max_input_channels=0,
        max_output_channels=2,
        name="Fake Speaker",
        index=9,
    )


def _patch_output_open(monkeypatch) -> None:
    from vspeech.stream_vc import playback

    monkeypatch.setattr(
        playback,
        "resolve_stream_vc_output_device",
        lambda config: _fake_output_device(),
    )
    monkeypatch.setattr(playback.sd, "RawOutputStream", _RecordingOutputStream)


def test_open_output_stream_requests_configured_latency(monkeypatch):
    """The configured value reaches sounddevice unconverted (ADR-0071)."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    playback.open_stream_vc_output_stream(
        StreamVcConfig(output_latency="high"), sample_rate=16000
    )
    assert _RecordingOutputStream.kwargs["latency"] == "high"


def test_open_output_stream_defaults_to_low(monkeypatch):
    """No setting = the value that used to be hardcoded."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    playback.open_stream_vc_output_stream(StreamVcConfig(), sample_rate=16000)
    assert _RecordingOutputStream.kwargs["latency"] == "low"


def test_open_output_stream_uses_output_latency_not_input(monkeypatch):
    """The input setting must not leak into the output stream."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    config = StreamVcConfig(input_latency="high", output_latency=0.02)
    playback.open_stream_vc_output_stream(config, sample_rate=16000)
    assert _RecordingOutputStream.kwargs["latency"] == 0.02


def test_open_output_stream_logs_requested_and_granted_latency(caplog, monkeypatch):
    """consumer.py reuses this function, so the consumer machine gets the same line."""
    from vspeech.stream_vc import playback

    _patch_output_open(monkeypatch)
    with caplog.at_level(logging.INFO):
        playback.open_stream_vc_output_stream(StreamVcConfig(), sample_rate=16000)
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Fake Speaker" in messages  # the device line still names the device
    assert "low" in messages  # requested
    assert "0.048" in messages  # granted
```

- [ ] **Step 4: テストが落ちることを確認する**

```sh
uv run pytest tests/test_stream_vc_capture.py tests/test_stream_vc_playback.py -v
```

Expected: 新規 8 件のうち 5 件が FAIL、3 件が PASS。内訳:

| テスト | 実装前 | 理由 |
|---|---|---|
| `test_open_input_stream_requests_configured_latency` | FAIL | `"low"` 固定なので `0.05` にならない |
| `test_open_input_stream_defaults_to_low` | PASS | 既定は元のハードコード値と同じなので実装前から通る |
| `test_open_input_stream_uses_input_latency_not_output` | PASS | 期待値が `"low"` なので同上（実装後の回帰テストとして残す） |
| `test_open_input_stream_logs_requested_and_granted_latency` | FAIL | granted latency を出していない |
| `test_open_output_stream_requests_configured_latency` | FAIL | `"high"` にならない |
| `test_open_output_stream_defaults_to_low` | PASS | 上と同じ理由 |
| `test_open_output_stream_uses_output_latency_not_input` | FAIL | `0.02` にならない |
| `test_open_output_stream_logs_requested_and_granted_latency` | FAIL | granted latency を出していない |

既存のテストは PASS のまま。

- [ ] **Step 5: capture.py を実装する**

`vspeech/stream_vc/capture.py:58-70` の `open_stream_vc_input_stream` を丸ごと差し替える。

```python
def open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream:
    device = resolve_stream_vc_input_device(config)
    # Logged before the open so a failing open still says which device was attempted.
    logger.info(
        "stream_vc input device %s: %s (latency %s requested)",
        device.index,
        device.name,
        config.input_latency,
    )
    stream = sd.RawInputStream(
        samplerate=CAPTURE_RATE,
        blocksize=hop,
        device=device.index,
        channels=1,
        dtype="int16",
        latency=config.input_latency,
    )
    # PortAudio does not have to honour the request, and "low" resolves to a different
    # number per host API. Report what was actually granted, before start() so a failing
    # start still leaves the number in the log.
    logger.info("stream_vc input stream latency: %.3fs", stream.latency)
    stream.start()
    return stream
```

- [ ] **Step 6: playback.py を実装する**

`vspeech/stream_vc/playback.py:42-55` の `open_stream_vc_output_stream` を丸ごと差し替える。

```python
def open_stream_vc_output_stream(
    config: StreamVcConfig, sample_rate: int
) -> sd.RawOutputStream:
    device = resolve_stream_vc_output_device(config)
    # Logged before the open so a failing open still says which device was attempted.
    logger.info(
        "stream_vc output device %s: %s (latency %s requested)",
        device.index,
        device.name,
        config.output_latency,
    )
    stream = sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        device=device.index,
        dtype="int16",
        latency=config.output_latency,
    )
    # PortAudio does not have to honour the request, and "low" resolves to a different
    # number per host API. Report what was actually granted, before start() so a failing
    # start still leaves the number in the log.
    logger.info("stream_vc output stream latency: %.3fs", stream.latency)
    stream.start()
    return stream
```

- [ ] **Step 7: テストが通ることを確認する**

```sh
uv run pytest tests/test_stream_vc_capture.py tests/test_stream_vc_playback.py -v
```

Expected: 全件 PASS。

- [ ] **Step 8: consumer が追随していることを確認する**

```sh
uv run pytest tests/test_stream_vc_consumer.py tests/test_stream_vc_subsystem.py -v
```

Expected: 全件 PASS。`consumer.py` は `open_stream_vc_output_stream` を再利用しているだけなので、変更なしで `output_latency` が効く。これで受入基準の「`role=producer` は入力側、`role=consumer` は出力側の設定だけを使う」も満たされる — producer は `capture_loop` しか動かさず入力 open にしか到達せず、consumer は `network_playback_loop` しか動かさず出力 open にしか到達しない、という既存の役割分担（ADR-0055、`test_stream_vc_subsystem.py` が固定している）に latency がそのまま乗るため。

- [ ] **Step 9: 型チェックとフォーマット**

```sh
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected: 本タスクの変更に起因する新規の指摘なし。

- [ ] **Step 10: コミット**

```bash
git add vspeech/stream_vc/capture.py vspeech/stream_vc/playback.py tests/test_stream_vc_capture.py tests/test_stream_vc_playback.py
git commit -m "feat(stream-vc): request the configured device latency and log what PortAudio granted"
```

---

### Task 3: config.toml.example に記載し、ADR-0071 を Accepted へ昇格する

**Files:**
- Modify: `config.toml.example:283` 付近（`output_device_index = 1` の直後）
- Modify: `docs/adr/0071-stream-vc-device-latency-config.md:3`（Status 行）
- Modify: `docs/adr/README.md`（索引の Status 列）

**Interfaces:**
- Consumes: Task 1 のフィールド名 `input_latency` / `output_latency` と既定値 `"low"`
- Produces: なし（最終タスク）

- [ ] **Step 1: config.toml.example に追記する**

`output_device_index = 1`（283 行目）の直後、`# トランスポート種別:` のコメントブロックの直前に挿入する。

```toml

# 入出力ストリームがデバイスへ要求する latency (ADR-0071)。既定は両方 "low" で、
# これは設定化する前にハードコードされていた値と同じ = 書かなければ挙動は変わらない。
#   "low"  = デバイス既定の低遅延。実際の秒数はホスト API 依存で、WASAPI と MME で桁が違う
#   "high" = デバイス既定の高遅延。overflow/underflow が止まらないときの逃げ道
#   数値   = 秒での明示指定 (例 0.02 = 20ms)。low と high の中間を狙うとき
# 入力(マイク)と出力(再生)は別デバイス、role を分ければ別マシンなので、独立に指定する。
# PortAudio は要求値を保証しない。実際に得られた値は起動時のログ
# ("stream_vc input/output stream latency: N.NNNs") に出るので、block_ms を詰める前に
# まずそれを読むこと。
input_latency = "low"
output_latency = "low"
```

- [ ] **Step 2: 設定例がそのまま読めることを確認する**

```sh
uv run python -c "import toml; d = toml.load('config.toml.example'); print(d['stream_vc']['input_latency'], d['stream_vc']['output_latency'])"
```

Expected: `low low` と出力される（TOML として壊れておらず、キー名が実装と一致している）。

- [ ] **Step 3: ADR-0071 を Accepted へ昇格する**

`docs/adr/0071-stream-vc-device-latency-config.md` の 3 行目を 1 行だけ書き換える。本文は変更しない（ADR は不変層）。

```markdown
- Status: Accepted
```

`docs/adr/README.md` の索引の 0071 の行も Status 列を `Proposed` → `Accepted` にする。

```markdown
| [0071](0071-stream-vc-device-latency-config.md) | ストリーミング VC のデバイス latency を入出力別の設定値にする | Accepted | 2026-08-10 |
```

- [ ] **Step 4: 全ゲートを回す**

```sh
uv run poe check
```

Expected: `fmt-check` / `lint` / `ty` / `pytest` が緑。`uv audit` の torch 由来など、この変更以前から accepted になっている指摘だけが残る。pytest の件数は変更前 +15 件（Task 1 で 6 件 + `test_main.py` の parametrize 1 ケース、Task 2 で 8 件）。

- [ ] **Step 5: ADR ↔ 実装の突合**

`docs/adr/0071-*.md` の Decision を読み直し、実装と食い違いがないか確認する。確認する点は 4 つ:

1. フィールド名が `input_latency` / `output_latency` である
2. 型が `Literal["low", "high"] | float(gt=0)` で、既定が両方 `"low"` である
3. 値が単位変換されず sounddevice へ渡っている（ms への変換が入っていない）
4. open 時のログに要求値と `stream.latency` の両方が出る

食い違いがあれば、実装が正しいなら新 ADR で supersede、ADR が正しいなら実装を直す。**Accepted の本文は書き換えない。**

- [ ] **Step 6: コミット**

```bash
git add config.toml.example docs/adr/0071-stream-vc-device-latency-config.md docs/adr/README.md
git commit -m "docs(stream-vc): document input_latency / output_latency and accept ADR-0071"
```

---

## 実機確認（マージ前、ユーザーの手作業）

自動テストは sounddevice の境界までしか見ない。PortAudio が実際に何を返すかは実機依存なので、マージ前に以下を 1 回だけ確認する。

- [ ] `[stream_vc]` を有効にした config で `uv run python -m vspeech --config <config>` を起動し、ログに `stream_vc input stream latency: N.NNNs` と `stream_vc output stream latency: N.NNNs` が出ることを確認する。
- [ ] `output_latency = "high"` に変えて再起動し、ログの granted 値が `"low"` のときより大きくなることを確認する（デバイスが両者を区別しない場合は同じ値になりうる。その場合は「区別しないデバイスだった」と記録して先へ進む）。
