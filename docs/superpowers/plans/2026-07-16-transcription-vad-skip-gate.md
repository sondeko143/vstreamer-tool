# transcription VAD スキップゲート Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** transcription ワーカーに opt-in の Silero VAD スキップ判定を追加し、音声の少ないチャンクを音声認識前に落とす（vc は無変更）。

**Architecture:** vc の VAD ゲート（ADR-0019）と同じ純関数ライブラリ `vspeech/lib/vad.py` を再利用する。transcription は音声を出力しないため **スキップ判定のみ**（出力ダックは無し）。入力→16kHz 変換は transcription 既存の `pcm_to_waveform`（PyAV 経由・torch 不要, ADR-0036）を再利用。3 バックエンド（ACP/GCP/WHISPER）の各ループ先頭で判定し、skip なら `continue`。新ワーカー・新 EventType・wire 変更なし（ADR-0037）。

**Tech Stack:** Python 3.14, pydantic v2, onnxruntime（Silero VAD, CPU 固定）, numpy, faster-whisper（PyAV 同梱）, pytest（`asyncio_mode="auto"`）, ruff / ty / poe。

## Global Constraints

- Python **3.14 のみ**（`>=3.14,<3.15`）。floor を下げない。
- Imports は **1 行 1 import**（ruff `force-single-line = true`, 保存時に自動整列）。
- 型チェックは **ty**（`uv run ty check`）。pyright は使わない。
- **Pydantic v2** API のみ（`Field`, `model_validate` 等）。v1 API（`.dict()`/`parse_obj` 等）禁止。
- 既定は **opt-in で OFF**（`vad_gate = false`）。ゲート無効時は本変更前と挙動同一。
- **エラー非対称**（ADR-0019 踏襲）: モデル不在/ロード失敗は**起動時 fail loudly**、推論中例外は**そのチャンクをゲートせず通す**。
- VAD セッションは **CPUExecutionProvider 固定**（`create_vad_session` が担保。ADR-0024 の意図的例外）。
- 設定は **各ワーカー独立**。`vc.vad_*` は触らない。`transcription.vad_*` を新設。
- `vspeech/` は `fairseq`/`transformers`/`torch` を transcription パスに新規 import しない（VAD 入力変換は既存 PyAV 経路を再利用）。

---

## File Structure

- **Modify** `vspeech/config.py` — `TranscriptionConfig` に VAD 4 フィールド追加。
- **Modify** `vspeech/worker/transcription.py` — VAD ヘルパ 2 個追加 + 3 バックエンドに配線。
- **Create** `tests/test_transcription_vad.py` — config 既定 + 2 ヘルパのユニットテスト。
- **Modify** `config.toml.example` — `[transcription]` に VAD フィールドを文書化。
- **Modify** `docs/adr/0037-transcription-vad-skip-gate.md` — 実装後に Status を Proposed→Accepted へ昇格。

**テスト方針（既存踏襲）:** 既存の `tests/test_vad_gate.py` は VAD の**純関数**を検証し、`rvc_worker` の無限ループ自体はユニットテストしない。本プランも同じ方針で、判定ロジックはヘルパ（Task 3）で網羅的に検証し、3 バックエンドへの配線（Task 4）は typecheck + 全スイート + 起動スモーク（Task 6）で担保する（無限ループ×外部サービス/GPU をユニット化しない）。

---

### Task 1: TranscriptionConfig に VAD フィールドを追加

**Files:**
- Modify: `vspeech/config.py:168-173`（`class TranscriptionConfig`）
- Test: `tests/test_transcription_vad.py`

**Interfaces:**
- Produces: `TranscriptionConfig.vad_gate: bool`, `.vad_model_file: Path`, `.vad_threshold: float`, `.vad_min_speech_ratio: float`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_transcription_vad.py` を新規作成:

```python
from pathlib import Path


def test_transcription_config_vad_defaults_are_off_and_sane():
    from vspeech.config import TranscriptionConfig

    config = TranscriptionConfig()
    assert config.vad_gate is False
    assert config.vad_model_file == Path()
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ratio == 0.1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_transcription_vad.py::test_transcription_config_vad_defaults_are_off_and_sane -v`
Expected: FAIL（`AttributeError: 'TranscriptionConfig' object has no attribute 'vad_gate'`）

- [ ] **Step 3: フィールドを追加**

`vspeech/config.py` の `TranscriptionConfig` を次に変更（`recording_log_dir` の直後に 4 行追加。`Field`/`Path` は既に import 済み）:

```python
class TranscriptionConfig(BaseModel):
    enable: bool = False
    worker_type: TranscriptionWorkerType = TranscriptionWorkerType.GCP
    transliterate_with_mozc: bool = False
    recording_log: bool = False
    recording_log_dir: Path = Path("./rec")
    # Silero VAD スキップゲート (opt-in)。vc.vad_* とは独立 (ADR-0037)。
    # 音声比率が閾値未満のチャンクを音声認識前に落とす。出力ダックは無し。
    vad_gate: bool = False
    vad_model_file: Path = Field(default=Path())
    vad_threshold: float = 0.5
    vad_min_speech_ratio: float = 0.1
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_transcription_vad.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py tests/test_transcription_vad.py
git commit -m "feat(transcription): add opt-in VAD skip-gate config fields"
```

---

### Task 2: VAD セッション生成ヘルパ

**Files:**
- Modify: `vspeech/worker/transcription.py`（`pcm_to_waveform` の直後, ~131 行目; TYPE_CHECKING ブロック ~46-47 行目; import 群）
- Test: `tests/test_transcription_vad.py`

**Interfaces:**
- Consumes: `TranscriptionConfig.vad_gate`, `.vad_model_file`（Task 1）; `vspeech.lib.vad.create_vad_session`
- Produces: `create_transcription_vad_session(config: TranscriptionConfig) -> InferenceSession | None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_transcription_vad.py` に追記:

```python
def test_create_session_none_when_gate_disabled():
    from vspeech.config import TranscriptionConfig
    from vspeech.worker.transcription import create_transcription_vad_session

    assert create_transcription_vad_session(TranscriptionConfig()) is None
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_transcription_vad.py::test_create_session_none_when_gate_disabled -v`
Expected: FAIL（`ImportError: cannot import name 'create_transcription_vad_session'`）

- [ ] **Step 3: import とヘルパを追加**

`vspeech/worker/transcription.py` の TYPE_CHECKING ブロックへ 1 行追加:

```python
if TYPE_CHECKING:
    import numpy as np
    from onnxruntime import InferenceSession
```

import 群（`from vspeech.lib...` の並び）へ 3 行追加（1 行 1 import）:

```python
from vspeech.lib.vad import create_vad_session
from vspeech.lib.vad import should_skip_vc
from vspeech.lib.vad import speech_probs
```

`pcm_to_waveform` の直後にヘルパを追加:

```python
def create_transcription_vad_session(
    config: TranscriptionConfig,
) -> "InferenceSession | None":
    """Build the Silero VAD session for the transcription skip gate, or None.

    Returns None when the gate is disabled so the hot loop can cheaply skip it.
    When enabled, a missing or malformed model raises inside create_vad_session
    (fail loudly at startup, ADR-0019/0037) rather than silently passing every
    chunk through. CPU-fixed session (ADR-0024 の意図的例外).
    """
    if not config.vad_gate:
        return None
    session = create_vad_session(config.vad_model_file)
    logger.info("transcription vad gate enabled: %s", config.vad_model_file)
    return session
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_transcription_vad.py -v`
Expected: PASS（全 2 テスト）

- [ ] **Step 5: ty で型を確認**

Run: `uv run ty check vspeech/worker/transcription.py`
Expected: 新規診断なし

- [ ] **Step 6: Commit**

```bash
git add vspeech/worker/transcription.py tests/test_transcription_vad.py
git commit -m "feat(transcription): add VAD session factory (None when disabled)"
```

---

### Task 3: スキップ判定ヘルパ

**Files:**
- Modify: `vspeech/worker/transcription.py`（`create_transcription_vad_session` の直後）
- Test: `tests/test_transcription_vad.py`

**Interfaces:**
- Consumes: `create_transcription_vad_session` の戻り値; `pcm_to_waveform(sound) -> np.ndarray`（16kHz float32 mono, 既存）; `speech_probs`, `should_skip_vc`（lib.vad）; `SoundInput`; `telemetry`, `to_thread`（既存 import）
- Produces: `async vad_should_skip(vad_session, sound: SoundInput, config: TranscriptionConfig, trace_id: str) -> bool`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_transcription_vad.py` に追記:

```python
import numpy as np

from vspeech.config import SampleFormat
from vspeech.config import TranscriptionConfig
from vspeech.shared_context import SoundInput


class _StubVad:
    """Duck-typed InferenceSession: returns a fixed speech prob per window and
    advances the recurrent state so state threading stays observable."""

    def __init__(self, prob: float):
        self._prob = prob

    def run(self, output_names, input_feed):
        state = input_feed["state"] + 1.0
        return np.array([[self._prob]], dtype=np.float32), state


def _sound_16k_int16(seconds: float = 1.0) -> SoundInput:
    n = int(seconds * 16000)
    data = np.zeros(n, dtype=np.int16).tobytes()
    return SoundInput(data=data, rate=16000, format=SampleFormat.INT16, channels=1)


async def test_vad_skip_disabled_returns_false():
    from vspeech.worker.transcription import vad_should_skip

    assert await vad_should_skip(None, _sound_16k_int16(), TranscriptionConfig(), "") is False


async def test_vad_skip_low_speech_returns_true():
    from vspeech.worker.transcription import vad_should_skip

    cfg = TranscriptionConfig(vad_gate=True, vad_threshold=0.5, vad_min_speech_ratio=0.1)
    assert await vad_should_skip(_StubVad(0.1), _sound_16k_int16(), cfg, "") is True


async def test_vad_pass_high_speech_returns_false():
    from vspeech.worker.transcription import vad_should_skip

    cfg = TranscriptionConfig(vad_gate=True, vad_threshold=0.5, vad_min_speech_ratio=0.1)
    assert await vad_should_skip(_StubVad(0.9), _sound_16k_int16(), cfg, "") is False


async def test_vad_exception_passes_through_ungated():
    from vspeech.worker.transcription import vad_should_skip

    class _Boom:
        def run(self, *args):
            raise RuntimeError("boom")

    cfg = TranscriptionConfig(vad_gate=True)
    assert await vad_should_skip(_Boom(), _sound_16k_int16(), cfg, "") is False
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_transcription_vad.py -k vad_ -v`
Expected: FAIL（`ImportError: cannot import name 'vad_should_skip'`）

- [ ] **Step 3: ヘルパを追加**

`vspeech/worker/transcription.py` の `create_transcription_vad_session` の直後に追加:

```python
async def vad_should_skip(
    vad_session: "InferenceSession | None",
    sound: SoundInput,
    config: TranscriptionConfig,
    trace_id: str,
) -> bool:
    """Return True if this chunk has too little speech to transcribe.

    Reuses the vc gate's pure decision (should_skip_vc) on 16kHz probs from the
    shared Silero model. Skip-only: transcription emits no audio, so there is no
    output duck (unlike the vc path). Error handling is asymmetric (ADR-0019):
    an inference failure passes the chunk through ungated (returns False) rather
    than dropping speech. Returns False immediately when the gate is disabled.
    """
    if vad_session is None:
        return False
    try:
        waveform = pcm_to_waveform(sound)
        probs = await to_thread(speech_probs, vad_session, waveform)
        skip, ratio = should_skip_vc(
            probs, config.vad_threshold, config.vad_min_speech_ratio
        )
        if skip:
            telemetry.record("transc_skip", ratio, trace_id=trace_id)
            logger.info(
                "transcription skipped: speech ratio %.3f < %.3f",
                ratio,
                config.vad_min_speech_ratio,
            )
            return True
        return False
    except Exception as e:  # noqa: BLE001 - gate failure must not drop speech
        logger.warning(
            "transcription vad gate failed; passing chunk ungated: %s", e
        )
        return False
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_transcription_vad.py -v`
Expected: PASS（全 6 テスト）

- [ ] **Step 5: ty で型を確認**

Run: `uv run ty check vspeech/worker/transcription.py`
Expected: 新規診断なし

- [ ] **Step 6: Commit**

```bash
git add vspeech/worker/transcription.py tests/test_transcription_vad.py
git commit -m "feat(transcription): add VAD skip decision (skip-only, fail-open)"
```

---

### Task 4: 3 バックエンドへ配線

**Files:**
- Modify: `vspeech/worker/transcription.py`（`transcript_worker_whisper` ~205-278, `transcript_worker_google` ~280-334, `transcript_worker_ami` ~337-390）

**Interfaces:**
- Consumes: `create_transcription_vad_session`, `vad_should_skip`（Task 2/3）; 各ジェネレータの `config: TranscriptionConfig`, `in_queue`, `recorded.sound`, `recorded.trace_id`

配線パターン（3 箇所とも同じ）: (a) `while True:` の**前**でセッションを 1 回生成、(b) `recorded = await in_queue.get()` の**直後**に skip チェックを挿入。reload 時は外側 `transcription_worker` がジェネレータを作り直すため、セッションも再構築される（`configs_depends_on` に `"transcription"` が既にあり、`vad_*` 変更で `need_reload` が立つ）。

- [ ] **Step 1: whisper バックエンドに配線**

`transcript_worker_whisper` 内、warmup（`logger.info("transcript worker [whisper] warmed up")` の `try/except` 直後）と `while True:` の間にセッション生成を追加:

```python
    vad_session = create_transcription_vad_session(config)
    while True:
        recorded = await in_queue.get()
        if await vad_should_skip(
            vad_session, recorded.sound, config, recorded.trace_id
        ):
            continue
        try:
            logger.debug("transcribing...")
            waveform = pcm_to_waveform(recorded.sound)
```

（既存の `while True: recorded = await in_queue.get(): try:` の間に 4 行を挿入する形。既存 `try` 以降は無変更。）

- [ ] **Step 2: google バックエンドに配線**

`transcript_worker_google` 内、`logger.info("transcript worker [google] started")` の直後にセッション生成を追加し、`recorded = await in_queue.get()` の直後に skip チェックを挿入:

```python
    logger.info("transcript worker [google] started")
    vad_session = create_transcription_vad_session(config)
    while True:
        recorded = await in_queue.get()
        if await vad_should_skip(
            vad_session, recorded.sound, config, recorded.trace_id
        ):
            continue
        try:
```

- [ ] **Step 3: ami バックエンドに配線**

`transcript_worker_ami` 内、`logger.info("transcript worker [ami] started")` の直後にセッション生成を追加し、`async with AsyncClient(...) as client:` 内の `recorded = await in_queue.get()` の直後に skip チェックを挿入:

```python
    logger.info("transcript worker [ami] started")
    vad_session = create_transcription_vad_session(config)
    while True:
        async with AsyncClient(timeout=ami_config.request_timeout) as client:
            recorded = await in_queue.get()
            if await vad_should_skip(
                vad_session, recorded.sound, config, recorded.trace_id
            ):
                continue
            try:
```

（`continue` は `async with` を抜けて `while` を次へ回す。意図通り。）

- [ ] **Step 4: ruff format / lint**

Run: `uv run ruff format vspeech/worker/transcription.py && uv run ruff check vspeech/worker/transcription.py`
Expected: フォーマット整形のみ、lint エラーなし

- [ ] **Step 5: ty で型を確認**

Run: `uv run ty check vspeech/worker/transcription.py`
Expected: 新規診断なし（`vad_should_skip` 呼び出しがシグネチャに一致）

- [ ] **Step 6: 全スイートが緑のまま**

Run: `uv run pytest tests/test_transcription_vad.py tests/test_event_chains.py -v`
Expected: PASS（配線で既存の routing テストが壊れていないこと）

- [ ] **Step 7: Commit**

```bash
git add vspeech/worker/transcription.py
git commit -m "feat(transcription): wire VAD skip-gate into ACP/GCP/WHISPER backends"
```

---

### Task 5: config.toml.example を文書化

**Files:**
- Modify: `config.toml.example:45-48`（`[transcription]` セクション）

- [ ] **Step 1: フィールドを追記**

`[transcription]` セクション（`transliterate_with_mozc = false` の直後）へ追加:

```toml
[transcription]
enable = true
worker_type = "ACP"
transliterate_with_mozc = false
# Silero VAD スキップゲート (opt-in)。音声比率の低いチャンクを音声認識前に落とす。
# vc.vad_* とは独立した設定 (同一感度にしたい場合は手動で揃える)。既定 OFF。
vad_gate = false
# vc と同じ silero_vad.onnx (v6.2.1) を指す。
# vad_model_file = "~/.config/vstreamer/silero_vad.onnx"
# speech 確率の閾値 (窓 32ms ごとにこれ以上で speech 扱い)。
vad_threshold = 0.5
# speech 窓の割合がこれ未満のチャンクは認識をスキップ (字幕・翻訳にも流れない)。
vad_min_speech_ratio = 0.1
```

- [ ] **Step 2: Commit**

```bash
git add config.toml.example
git commit -m "docs(config): document transcription VAD skip-gate fields"
```

---

### Task 6: 最終検証 + ADR 昇格

**Files:**
- Modify: `docs/adr/0037-transcription-vad-skip-gate.md:3`（Status 行）
- Modify: `docs/adr/README.md`（0037 行の Status 列）

- [ ] **Step 1: 健全性ゲート**

Run: `uv run poe check`
Expected: 既知の受容済み事項（memory 記載の torch CVE / vr2_config deadcode の 2 件）を除き緑。新規の ty/ruff/pytest 失敗が無いこと。

- [ ] **Step 2: 起動スモーク（transcription ホストで実施）**

`transcription.vad_gate = true` かつ `transcription.vad_model_file` を実モデルに向けた config で起動:

Run: `uv run python -m vspeech --config <transcription config>`
Expected: ログに `transcription vad gate enabled: <path>` と `transcript worker [...] started` が出て起動する。無音を流すとスキップされ、発話は従来どおり認識される（**教訓: テストだけでなくエントリポイントを実際に走らせる**）。

- [ ] **Step 3: vc 無変更を確認**

Run: `git diff main --stat -- vspeech/worker/vc.py vspeech/lib/vad.py`
Expected: 差分なし（vc パスと共有ライブラリ本体は未変更）。

- [ ] **Step 4: ADR-0037 を Accepted へ昇格**

`docs/adr/0037-transcription-vad-skip-gate.md` の Status 行を変更:

```markdown
- Status: Accepted (extends [ADR-0019](0019-vc-silero-vad-gate.md))
```

`docs/adr/README.md` の 0037 行の Status 列を `Proposed` → `Accepted (extends 0019)` に更新。

- [ ] **Step 5: ADR ↔ 実装の突合（final-review）**

ADR-0037 の Decision 各項が実装と一致することを確認（skip-only / PyAV 再利用 / 全バックエンド / 独立 config / エラー非対称 / CPU 固定）。乖離があれば実装修正または supersede。出力: 「触れた ADR ↔ 実装: 乖離なし」。

- [ ] **Step 6: Commit**

```bash
git add docs/adr/0037-transcription-vad-skip-gate.md docs/adr/README.md
git commit -m "docs(adr): promote 0037 to Accepted after implementation"
```

---

## Self-Review

**1. Spec coverage（受入基準 → タスク）:**
- opt-in・既定 OFF → Task 1 + Task 4（session None 経路）
- 閾値未満は認識も下流出力もしない → Task 3（skip True）+ Task 4（`continue`）
- 十分な音声は従来どおり出力 → Task 3（pass）+ Task 4
- ゲイン非依存 → Silero 絶対判定の再利用（lib/vad, 既存）で担保
- 全バックエンド → Task 4（3 ジェネレータ）
- モデル欠如は起動時 fail loudly → Task 2（`create_vad_session` が raise）
- 推論例外はゲートせず通す → Task 3（except → False）
- スキップをテレメトリ記録 → Task 3（`telemetry.record("transc_skip", ...)`）
- 設定は vc と独立 → Task 1（`transcription.vad_*` 新設、vc 無変更）
- vc 無変更 → Task 6 Step 3 で検証
- ゲート無効時は挙動同一 → Task 4（session None → 即 False）+ Task 6

全受入基準にタスク対応あり。ギャップなし。

**2. Placeholder scan:** "TBD"/"add appropriate ..."/"similar to Task N" 等なし。各コードステップに実コードあり。

**3. Type consistency:** `vad_should_skip(vad_session, sound, config, trace_id) -> bool` は Task 3 定義・Task 4 呼び出しで一致。`create_transcription_vad_session(config) -> InferenceSession | None` は Task 2 定義・Task 4 でその戻り値を `vad_should_skip` の第 1 引数へ渡す形で一致。`should_skip_vc(probs, threshold, min_speech_ratio) -> (bool, float)` は lib/vad の既存シグネチャに一致。
