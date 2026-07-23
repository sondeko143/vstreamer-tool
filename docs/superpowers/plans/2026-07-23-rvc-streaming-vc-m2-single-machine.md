# RVC ストリーミング VC — M2: 単一マシン内ストリーミング Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** マイク → ステートフル VC(等電力クロスフェード)→ ローカル連続再生を、Command/routing の外の in-process サブシステムとして通し、ブロック境界のクリック無し・ピッチ連続を実機で耳確認できる状態にする。

**Architecture:** 既存 `vspeech/lib/stream_vc.py` の `StreamingVc` に等電力クロスフェード overlap-add を追加(seam ブレンドは両隣ウィンドウが描画する context-overlap 帯から採るので `seq=[context|block]` は M1 のまま無改変、emit だけ差し替え)。`vspeech/stream_vc/` サブパッケージに capture(独立マイク)→ runner(`StreamingVc`)→ transport(送受信 interface + in-process 実装)→ playback(連続再生)を配線し、`config.stream_vc.enable` のとき `main.py` が 1 タスクとして起動する。設定は `[stream_vc]`(ADR-0054)、必須検査は preflight 追記で GUI 自動追従(ADR-0045)。

**Tech Stack:** Python 3.14 / uv / numpy / torch(CUDA)/ onnxruntime-gpu(rvc extra)/ sounddevice(audio extra)/ pydantic v2 / pytest(asyncio_mode=auto)/ poethepoet。

## Global Constraints

- **Python 3.14 のみ**(`requires-python = ">=3.14,<3.15"`)。3.13 以下へ下げない。
- **発話系(録音→文字起こし→翻訳→字幕)と既存 `change_voice` 経路は無改変**(ADR-0050/0052/0053)。`vspeech/lib/rvc.py` / `vspeech/worker/vc.py` / `vspeech/worker/recording.py` / `vspeech/worker/playback.py` / routing(`vspeech/lib/command.py`, `vspeech/shared_context.py`)を**編集しない**。streaming は `rvc.py` の private 部品を import 再利用するだけ。
- **routing の外**(ADR-0050): stream_vc サブシステムは `context.add_worker` / `in_queue` / `sender_queue` / `WorkerInput` / `WorkerOutput` を**使わない**。自前の queue と transport を持つ。
- **M2 スコープ外(このプランで作らない)**: 網トランスポート実装(UDP/TCP/bidi)・jitter buffer・欠落/並べ替えの穴埋め(ADR-0051、M3)。streaming 用 VAD hangover / rolling-EMA envelope は**スケジュールしない**(ADR-0053 の既定 off のまま。問題化したときに初めて追加する — M3 繰り越しではない)。
- **薄い in-process seam**(ユーザ確定): transport は `send`/`recv` interface + in-process 実装のみ。`StreamPacket` は seq/pts を持ち、M3 で網実装に差し替わる seam(ADR-0051 tier-0)。
- **重い import は遅延**: `torch` / `onnxruntime` / `vspeech.lib.rvc` は関数内 import。`sounddevice` は **audio extra** なので `capture.py` / `playback.py` の top-level に置いてよい(それらは enable 時のみ import される)。ただし `subsystem.py` は CPU import 可能に保ち、capture/runner/playback の import は `create_stream_vc_task` 内に遅延させる。純粋モジュール(`packet.py` / `transport.py`)は base 依存のみで import できること。
- **import は 1 行 1 個**(ruff `force-single-line`)。`uv run ruff format .` / `uv run ruff check .` / `uv run ty check` が緑であること。
- **GPU/モデル依存テストのゲート**は既存 golden と同じ(`tests/test_change_voice_golden.py` / `tests/test_stream_vc.py`): `torch.cuda.is_available()` かつ 環境変数 `VSPEECH_RVC_GOLDEN_CONFIG` が実在 TOML を指すとき以外は `skip`。エントリポイント smoke は実機マイク+スピーカ+モデルが要るので別環境変数 `VSPEECH_STREAM_VC_CONFIG`(stream_vc を enable した実 config)+ CUDA で gate。
- **クロスフェードは context-overlap 帯の true overlap-add**: `context_len >= crossfade_len` かつ `crossfade_len < block_len`(fail-loud で検査)。出力長はレートロック(毎 tick `round(block_len*target_sr/16000)` サンプルちょうど emit)でドリフト無し。
- **決定論**: 合成信号は seed 付き(`np.random.default_rng(seed)`)。時間計測は `time.perf_counter`。
- **コミット**: 作業ブランチ `feat/rvc-streaming-vc`(既存、main ではない)へ直接コミット。メッセージ末尾に:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- **Modify `vspeech/lib/stream_vc.py`** — `equal_power_weights` / `overlap_add` 純粋ヘルパ追加。`StreamingVc` に `crossfade_len`(既定 0=M1 挙動)+ 出力域プリコンピュート + `_output_tail` 状態 + `_emit_with_crossfade` を追加。`seq` 構築と M1 path(crossfade_len==0)は無改変。
- **Modify `vspeech/config.py`** — `TransportType` enum + `StreamVcConfig`(入れ子 `RvcConfig` 再利用)追加、`Config.stream_vc` 追加。
- **Modify `vspeech/lib/audio.py`** — `resolve_stream_vc_input_device` / `resolve_stream_vc_output_device`(`_resolve_device` 再利用、preflight と capture/playback が同一経路 = ADR-0038)。
- **Modify `vspeech/preflight.py`** — `_check_rvc_assets`(既存 `_check_vc` から抽出、DRY)+ `_check_stream_vc` 追加、`_CHECKERS` 登録。
- **Create `vspeech/stream_vc/__init__.py`** — 空(パッケージマーカ)。
- **Create `vspeech/stream_vc/packet.py`** — `StreamPacket` dataclass。
- **Create `vspeech/stream_vc/transport.py`** — `drop_oldest_put` 純粋ヘルパ + `Transport` ABC + `InProcessTransport`。
- **Create `vspeech/stream_vc/capture.py`** — `ms_to_samples` / `pcm16_to_float32` 純粋ヘルパ + 独立入力ストリーム + `capture_loop`。
- **Create `vspeech/stream_vc/runner.py`** — `build_stream_vc_runtime` + `make_streaming_vc` + `vc_loop`。
- **Create `vspeech/stream_vc/playback.py`** — `detect_gap` 純粋ヘルパ + 出力ストリーム + `playback_loop`。
- **Create `vspeech/stream_vc/subsystem.py`** — `create_stream_vc_task`(capture/vc/playback を内側 TaskGroup で束ねる)。
- **Modify `vspeech/main.py`** — `if config.stream_vc.enable: create_stream_vc_task(...)` を 1 分岐追加。
- **Create tests** — `tests/test_stream_vc.py`(既存に crossfade を追記)、`tests/test_stream_vc_config.py`、`tests/test_stream_vc_preflight.py`、`tests/test_stream_vc_transport.py`、`tests/test_stream_vc_capture.py`、`tests/test_stream_vc_playback.py`、`tests/test_stream_vc_entrypoint.py`。

---

## Task 1: 等電力クロスフェード純粋ヘルパ

**Files:**
- Modify: `vspeech/lib/stream_vc.py`(`slice_block_output` の後、`StreamingVc` の前に追記)
- Test: `tests/test_stream_vc.py`(既存に追記)

**Interfaces:**
- Consumes: (なし — numpy のみ、関数内 import)
- Produces:
  - `equal_power_weights(n: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]` — 長さ n の `(fade_in, fade_out)`。`fade_in[i]**2 + fade_out[i]**2 == 1`(sin/cos セル中心)。`n <= 0` は空配列 2 本。
  - `overlap_add(prev_tail, head, fade_in, fade_out)` — `prev_tail*fade_out + head*fade_in`(要素積、numpy/torch 両対応)。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc.py` の末尾(既存 import と純粋テストの後)に追記:
```python
def test_equal_power_weights_sum_of_squares_is_one():
    from vspeech.lib.stream_vc import equal_power_weights

    fade_in, fade_out = equal_power_weights(64)
    power = fade_in**2 + fade_out**2
    assert np.allclose(power, 1.0, atol=1e-5)


def test_equal_power_weights_direction():
    from vspeech.lib.stream_vc import equal_power_weights

    fade_in, fade_out = equal_power_weights(64)
    # fade_in rises 0->1, fade_out falls 1->0
    assert fade_in[0] < fade_in[-1]
    assert fade_out[0] > fade_out[-1]
    assert fade_in[0] < 0.1 and fade_out[0] > 0.9


def test_equal_power_weights_zero_is_empty():
    from vspeech.lib.stream_vc import equal_power_weights

    fade_in, fade_out = equal_power_weights(0)
    assert fade_in.shape == (0,) and fade_out.shape == (0,)


def test_overlap_add_boundaries():
    from vspeech.lib.stream_vc import equal_power_weights
    from vspeech.lib.stream_vc import overlap_add

    n = 100
    fade_in, fade_out = equal_power_weights(n)
    prev = np.full(n, 100.0, dtype=np.float32)
    head = np.full(n, 0.0, dtype=np.float32)
    blended = overlap_add(prev, head, fade_in, fade_out)
    # start dominated by prev (fade_out ~1), end by head (fade_out ~0)
    assert blended[0] > 99.0
    assert blended[-1] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc.py -k "equal_power or overlap_add" -v`
Expected: FAIL — `ImportError: cannot import name 'equal_power_weights'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/stream_vc.py` の `slice_block_output` 定義の直後に追記:
```python
def equal_power_weights(n: int):
    """長さ n の等電力クロスフェード重み `(fade_in, fade_out)`。

    セル中心の sin/cos なので `fade_in**2 + fade_out**2 == 1`。独立推論した
    (無相関の)隣接出力を混ぜても総電力が一定に保たれる — RVC デコーダは
    ステートレスで hop ごとに位相非整合なので、線形重みより等電力が正しい。
    `n <= 0` は空配列を返す(crossfade 無効)。
    """
    import numpy as np

    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    x = (np.arange(n, dtype=np.float32) + 0.5) / n
    fade_in = np.sin(0.5 * np.pi * x).astype(np.float32)
    fade_out = np.cos(0.5 * np.pi * x).astype(np.float32)
    return fade_in, fade_out


def overlap_add(prev_tail, head, fade_in, fade_out):
    """`prev_tail` をフェードアウト・`head` をフェードインして加算する。

    等電力 overlap-add の 1 行。要素積なので numpy 配列でも torch tensor でも
    動く(呼び出し側は同じ長さ・同じ域で渡す)。
    """
    return prev_tail * fade_out + head * fade_in
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc.py -k "equal_power or overlap_add" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/stream_vc.py tests/test_stream_vc.py
git commit -m "feat(stream-vc): add equal-power crossfade helpers (weights/overlap-add)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: StreamingVc に crossfade overlap-add を追加

**Files:**
- Modify: `vspeech/lib/stream_vc.py`(`StreamingVc.__init__` に引数追加、`_emit_with_crossfade` 追加、`process_block` の emit 分岐)
- Test: `tests/test_stream_vc.py`(GPU-gated crossfade スモーク追記)

**Interfaces:**
- Consumes: `equal_power_weights` / `overlap_add` / `next_context` / `slice_block_output`(Task 1 と既存)。
- Produces:
  - `StreamingVc.__init__(..., block_len: int, context_len: int, crossfade_len: int = 0)` — `crossfade_len` を末尾に追加(既定 0 = M1 挙動、既存呼び出し・M1 ハーネス無改変)。`crossfade_len > 0` のとき `context_len >= crossfade_len` かつ `crossfade_len < block_len` を要求(満たさなければ `ValueError`)。
  - `StreamingVc.process_block(block) -> NDArray[np.int16]` — `crossfade_len > 0` のとき毎 tick `round(block_len*target_sample_rate/16000)` サンプルちょうど(レートロック)を返す。

**重要**: `seq = [context | block]` の構築と M1 path(`crossfade_len == 0`)は**無改変**。seam ブレンドは両隣ウィンドウが描画する context-overlap 帯(出力オフセット `out_ctx-out_xf : out_ctx`)から採るので入力の二重連結は不要。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc.py` の GPU-gated セクション(`_gpu_gate` 定義済み)に追記:
```python
@_gpu_gate
def test_streaming_vc_crossfade_rate_locked_and_finite():
    from scripts import capture_change_voice_golden as cap

    from vspeech.lib.stream_vc import StreamingVc

    assert _GOLDEN_CONFIG is not None
    rt = cap.build_rvc_runtime(_GOLDEN_CONFIG)

    block_len = 1280  # 80ms @ 16k
    context_len = 1600  # 100ms @ 16k
    crossfade_len = 160  # 10ms @ 16k
    sv = StreamingVc(
        rvc_config=rt["rvc_config"],
        device=rt["device"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_session=rt["f0_session"],
        target_sample_rate=rt["target_sample_rate"],
        f0_enabled=rt["f0_enabled"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        block_len=block_len,
        context_len=context_len,
        crossfade_len=crossfade_len,
    )
    sv.warmup()

    from scripts.stream_vc_rtf import make_voiced_signal

    signal = make_voiced_signal(16000, 2.0, seed=0)
    expected = round(block_len * rt["target_sample_rate"] / 16000)
    outs = [
        sv.process_block(signal[i * block_len : (i + 1) * block_len]) for i in range(3)
    ]
    for out in outs:
        assert out.dtype == np.int16
        assert out.shape[0] == expected  # rate-locked emit, no drift
        assert np.all(np.isfinite(out))
    assert any(np.any(out != 0) for out in outs)
```

> 注: `context_len >= crossfade_len` / `crossfade_len < block_len` の guard(Step 3 で `__init__` が `ValueError`)は torch を組めない CPU では実体化して検査できない(`__init__` が `import torch`)。この guard は preflight(Task 4 の `stream_vc.crossfade_ms` 検査)で config 段の等価チェックが CPU テストされるので、`__init__` の raise は fail-loud の二重防御として残すに留める。

- [ ] **Step 2: Run test to verify it fails**

Run(GPU + config): `VSPEECH_RVC_GOLDEN_CONFIG=/path/to/config.toml uv run --all-extras pytest tests/test_stream_vc.py::test_streaming_vc_crossfade_rate_locked_and_finite -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'crossfade_len'`(GPU 無しは SKIPPED)。

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/stream_vc.py` の `StreamingVc.__init__` シグネチャに `crossfade_len` を追加(`context_len: int,` の後):
```python
        block_len: int,
        context_len: int,
        crossfade_len: int = 0,
    ) -> None:
```
`__init__` 本体末尾(`self._context = ...` の直後)に追記:
```python
        self.crossfade_len = crossfade_len
        r = target_sample_rate
        self._out_ctx = round(context_len * r / 16000)
        self._out_hop = round(block_len * r / 16000)
        self._out_xf = round(crossfade_len * r / 16000)
        self._fade_in, self._fade_out = equal_power_weights(self._out_xf)
        self._output_tail = None  # 初回 crossfade で zeros(out_xf) を遅延生成
        if crossfade_len > 0 and context_len < crossfade_len:
            raise ValueError(
                "context_len must be >= crossfade_len for context-overlap crossfade"
            )
        if crossfade_len >= block_len:
            raise ValueError("crossfade_len must be < block_len")
```
`process_block` の末尾(`out = audio_i16.detach().cpu().numpy()` の後)を差し替え:
```python
        out = audio_i16.detach().cpu().numpy()
        self._context = next_context(seq, self.context_len).detach()
        if self.crossfade_len > 0:
            return self._emit_with_crossfade(out)
        return slice_block_output(out, self.block_len, seq.shape[0])
```
`process_block` の後に `_emit_with_crossfade` を追加:
```python
    def _emit_with_crossfade(self, out: NDArray[np.int16]) -> NDArray[np.int16]:
        """context-overlap 帯で等電力 overlap-add し、hop 相当ちょうどを返す。

        `out` は `[context|block]` 全体の int16 出力。前 tick が hop 末尾 out_xf
        を `_output_tail` に保持しており、今 tick はそれと同じ入力時刻を context
        末尾として再描画した `out[out_ctx-out_xf:out_ctx]` を等電力ブレンドして
        emit 先頭にする(seam の真の overlap-add、位相非整合を隠す)。emit は毎回
        out_hop サンプルちょうど = 入力 hop と同レート(ドリフト無し)。末尾 out_xf
        を次 tick 用の tail として保持する(算法遅延は crossfade 分のみ)。
        """
        import numpy as np

        out_ctx, out_hop, out_xf = self._out_ctx, self._out_hop, self._out_xf
        need = out_ctx + out_hop
        if out.shape[0] < need:  # context_len=0 等の端で稀に丸め不足 → 左ゼロ詰め
            out = np.pad(out, (need - out.shape[0], 0))
        out_f = out.astype(np.float32)
        if self._output_tail is None:
            self._output_tail = np.zeros(out_xf, dtype=np.float32)
        ctx_tail = out_f[out_ctx - out_xf : out_ctx]
        blended = overlap_add(self._output_tail, ctx_tail, self._fade_in, self._fade_out)
        fresh_middle = out_f[out_ctx : out_ctx + out_hop - out_xf]
        emit_f = np.concatenate([blended, fresh_middle])
        self._output_tail = out_f[out_ctx + out_hop - out_xf : need].copy()
        return np.clip(np.rint(emit_f), -32768.0, 32767.0).astype(np.int16)
```

- [ ] **Step 4: Run test to verify it passes**

Run(GPU + config): `VSPEECH_RVC_GOLDEN_CONFIG=/path/to/config.toml uv run --all-extras pytest tests/test_stream_vc.py -v`
Expected: PASS(GPU 有り)/ SKIPPED(GPU 無し)。純粋ヘルパ + M1 の既存テストは常に PASS。
Run(M1 ハーネス無回帰、純粋): `uv run pytest tests/test_stream_vc_rtf.py -v`
Expected: 既存 PASS(crossfade_len 既定 0 なので M1 path 無改変)。
Run(型/lint): `uv run --all-extras ty check vspeech/lib/stream_vc.py && uv run ruff check vspeech/lib/stream_vc.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/lib/stream_vc.py tests/test_stream_vc.py
git commit -m "feat(stream-vc): equal-power context-overlap crossfade in StreamingVc

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `[stream_vc]` 設定セクション(ADR-0054)

**Files:**
- Modify: `vspeech/config.py`(`RvcConfig` の後に `TransportType` + `StreamVcConfig`、`Config` に `stream_vc` 追加)
- Test: `tests/test_stream_vc_config.py`

**Interfaces:**
- Consumes: 既存 `RvcConfig`。
- Produces:
  - `class TransportType(Enum)` — `in_process = "in_process"`(当面これのみ。ADR-0051 の切替点)。
  - `class StreamVcConfig(BaseModel)` — `enable=False` / `rvc: RvcConfig`(入れ子再利用)/ `block_ms/context_ms/crossfade_ms` / 入出力デバイス選択 / `transport_type` / `max_queued_blocks` / `output_volume`。
  - `Config.stream_vc: StreamVcConfig`。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_config.py`:
```python
import io

from vspeech.config import Config
from vspeech.config import StreamVcConfig
from vspeech.config import TransportType


def test_stream_vc_defaults():
    c = StreamVcConfig()
    assert c.enable is False
    assert c.block_ms == 80.0
    assert c.context_ms == 100.0
    assert c.crossfade_ms == 10.0
    assert c.transport_type == TransportType.in_process
    assert c.max_queued_blocks == 8
    # nested rvc is an independent RvcConfig (ADR-0054/0046)
    assert c.rvc.f0_extractor_type is not None


def test_config_has_stream_vc_section():
    c = Config()
    assert c.stream_vc.enable is False


def test_stream_vc_parses_from_toml():
    toml_text = b"""
[stream_vc]
enable = true
block_ms = 120
context_ms = 200
crossfade_ms = 12
transport_type = "in_process"

[stream_vc.rvc]
model_file = "/models/voice.onnx"
f0_extractor_type = "fcpe"
"""
    f = io.BytesIO(toml_text)
    f.name = "config.toml"
    c = Config.read_config_from_file(f)
    assert c.stream_vc.enable is True
    assert c.stream_vc.block_ms == 120.0
    assert c.stream_vc.crossfade_ms == 12.0
    assert str(c.stream_vc.rvc.model_file) == "/models/voice.onnx"
    assert c.stream_vc.rvc.f0_extractor_type.value == "fcpe"


def test_stream_vc_survives_export_to_toml_round_trip():
    import toml as toml_lib

    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.block_ms = 160.0
    dumped = c.export_to_toml()
    reloaded = toml_lib.loads(dumped)
    assert reloaded["stream_vc"]["enable"] is True
    assert reloaded["stream_vc"]["block_ms"] == 160.0
    assert reloaded["stream_vc"]["transport_type"] == "in_process"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'StreamVcConfig'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/config.py` の `RvcConfig` 定義(`fcpe_model_file` 行)の直後に追記:
```python
class TransportType(Enum):
    in_process = "in_process"


class StreamVcConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    enable: bool = False
    # 発話系 [vc]/[rvc] とは独立したモデル設定(ADR-0054)。共有素材パスは
    # 各系統へ明示 propagate する方針(ADR-0046)。
    rvc: RvcConfig = Field(default_factory=RvcConfig)
    block_ms: float = Field(default=80.0, gt=0, description="固定ブロック(hop)長 ms")
    context_ms: float = Field(default=100.0, ge=0, description="rolling 左文脈 ms")
    crossfade_ms: float = Field(
        default=10.0, ge=0, description="等電力クロスフェード帯 ms (< block, <= context)"
    )
    input_host_api_name: str | None = Field(default=None)
    input_device_name: str | None = Field(default=None)
    input_device_index: int | None = Field(default=None)
    output_host_api_name: str | None = Field(default=None)
    output_device_name: str | None = Field(default=None)
    output_device_index: int | None = Field(default=None)
    transport_type: TransportType = Field(default=TransportType.in_process)
    max_queued_blocks: int = Field(
        default=8, gt=0, description="capture/transport の上限。満杯で最古を drop"
    )
```
`Config` の `rvc: RvcConfig = ...` 行の直後に追記:
```python
    stream_vc: StreamVcConfig = Field(default_factory=StreamVcConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_config.py -v`
Expected: PASS (4 passed)
Run(型/lint): `uv run ty check vspeech/config.py && uv run ruff check vspeech/config.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py tests/test_stream_vc_config.py
git commit -m "feat(stream-vc): add [stream_vc] config section (ADR-0054)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: preflight + device 解決(ADR-0038/0045/0046)

**Files:**
- Modify: `vspeech/lib/audio.py`(`resolve_output_device` の後に stream_vc 用 2 関数追加)
- Modify: `vspeech/preflight.py`(`_check_rvc_assets` 抽出 + `_check_stream_vc` 追加 + `_CHECKERS` 登録)
- Test: `tests/test_stream_vc_preflight.py`

**Interfaces:**
- Consumes: `StreamVcConfig`(Task 3)、`_resolve_device` / `DeviceInfo`(既存 audio.py)、`ConfigProblem`(既存)。
- Produces:
  - `resolve_stream_vc_input_device(config: StreamVcConfig) -> DeviceInfo`
  - `resolve_stream_vc_output_device(config: StreamVcConfig) -> DeviceInfo`
  - `_check_rvc_assets(rvc: RvcConfig, worker: str, field_prefix: str) -> list[ConfigProblem]` — `_check_vc` と `_check_stream_vc` が共有。
  - `_check_stream_vc(config: Config) -> list[ConfigProblem]`

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_preflight.py`:
```python
from pathlib import Path

from vspeech.config import Config
from vspeech.preflight import collect_problems


def _fields(problems) -> set[str]:
    return {p.field for p in problems}


def test_stream_vc_disabled_no_problems():
    c = Config()  # stream_vc disabled by default
    fields = _fields(collect_problems(c))
    assert not any(f and f.startswith("stream_vc") for f in fields)


def test_stream_vc_enabled_missing_model_reported():
    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.rvc.model_file = Path("/nonexistent/voice.onnx")
    c.stream_vc.rvc.hubert_model_file = Path("/nonexistent/hubert")
    fields = _fields(collect_problems(c))
    assert "stream_vc.rvc.model_file" in fields
    assert "stream_vc.rvc.hubert_model_file" in fields


def test_stream_vc_enabled_crossfade_gt_block_reported():
    c = Config()
    c.stream_vc.enable = True
    c.stream_vc.block_ms = 10.0
    c.stream_vc.crossfade_ms = 20.0  # crossfade must be < block
    problems = collect_problems(c)
    assert any(p.field == "stream_vc.crossfade_ms" for p in problems)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_preflight.py -v`
Expected: FAIL — stream_vc の problem が出ない(`_check_stream_vc` 未実装)。

- [ ] **Step 3: Write minimal implementation**

`vspeech/lib/audio.py` の `resolve_output_device` の後に追記(冒頭 import に `from vspeech.config import StreamVcConfig` を 1 行足す):
```python
def resolve_stream_vc_input_device(config: StreamVcConfig) -> DeviceInfo:
    """streaming VC の独立入力デバイスを解決する (preflight と capture が同経路)。"""
    return _resolve_device(
        index=config.input_device_index,
        index_key="stream_vc.input_device_index",
        host_api_type=config.input_host_api_name,
        host_api_key="stream_vc.input_host_api_name",
        name=config.input_device_name,
        name_key="stream_vc.input_device_name",
        input=True,
    )


def resolve_stream_vc_output_device(config: StreamVcConfig) -> DeviceInfo:
    """streaming VC の出力デバイスを解決する (preflight と playback が同経路)。"""
    return _resolve_device(
        index=config.output_device_index,
        index_key="stream_vc.output_device_index",
        host_api_type=config.output_host_api_name,
        host_api_key="stream_vc.output_host_api_name",
        name=config.output_device_name,
        name_key="stream_vc.output_host_api_name".replace("host_api_name", "device_name"),
        output=True,
    )
```
> 注: `name_key` は `"stream_vc.output_device_name"` を直接書く(上の `.replace(...)` トリックは使わない — 誤記防止のため素直に文字列で書くこと):
```python
        name=config.output_device_name,
        name_key="stream_vc.output_device_name",
        output=True,
    )
```
`vspeech/preflight.py` の `_check_vc` を、RVC 資産検査を `_check_rvc_assets` に抽出してから呼ぶ形に変更。まず `from vspeech.config import RvcConfig` を import 群に追加し、`_check_vc` の前に:
```python
def _check_rvc_assets(
    rvc: RvcConfig, worker: str, field_prefix: str
) -> list[ConfigProblem]:
    """RVC モデル資産(本体/HuBERT/f0)の存在検査。[vc] と [stream_vc] が共有。"""
    problems: list[ConfigProblem] = []
    if not rvc.model_file.expanduser().is_file():
        problems.append(
            ConfigProblem(
                worker,
                f"{field_prefix}.model_file '{rvc.model_file}' が存在しません",
                field=f"{field_prefix}.model_file",
            )
        )
    hubert = rvc.hubert_model_file
    if hubert == Path() or not hubert.expanduser().is_dir():
        problems.append(
            ConfigProblem(
                worker,
                f"{field_prefix}.hubert_model_file '{hubert}' (資産ディレクトリ) が存在しません",
                field=f"{field_prefix}.hubert_model_file",
            )
        )
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        if not rvc.rmvpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    worker,
                    f"{field_prefix}.rmvpe_model_file '{rvc.rmvpe_model_file}' が存在しません",
                    field=f"{field_prefix}.rmvpe_model_file",
                )
            )
    if rvc.f0_extractor_type == F0ExtractorType.fcpe:
        if not rvc.fcpe_model_file.expanduser().is_file():
            problems.append(
                ConfigProblem(
                    worker,
                    f"{field_prefix}.fcpe_model_file '{rvc.fcpe_model_file}' が存在しません",
                    field=f"{field_prefix}.fcpe_model_file",
                )
            )
    return problems
```
`_check_vc` の本体を差し替え(RVC 資産検査部分を委譲):
```python
def _check_vc(config: Config) -> list[ConfigProblem]:
    if not config.vc.enable:
        return []
    w = "vc"
    problems = _check_rvc_assets(config.rvc, w, "rvc")
    problems.extend(_check_vad_gate(config.vc, w))
    return problems
```
`_check_subtitle` の後に `_check_stream_vc` を追加:
```python
def _check_stream_vc(config: Config) -> list[ConfigProblem]:
    if not config.stream_vc.enable:
        return []
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_stream_vc_input_device
    from vspeech.lib.audio import resolve_stream_vc_output_device

    w = "stream_vc"
    sv = config.stream_vc
    problems = _check_rvc_assets(sv.rvc, w, "stream_vc.rvc")
    if sv.crossfade_ms >= sv.block_ms:
        problems.append(
            ConfigProblem(
                w,
                f"crossfade_ms ({sv.crossfade_ms}) は block_ms ({sv.block_ms}) 未満が必須です",
                field="stream_vc.crossfade_ms",
            )
        )
    if sv.crossfade_ms > sv.context_ms:
        problems.append(
            ConfigProblem(
                w,
                f"crossfade_ms ({sv.crossfade_ms}) は context_ms ({sv.context_ms}) 以下が必須です",
                field="stream_vc.crossfade_ms",
            )
        )
    try:
        resolve_stream_vc_input_device(sv)
    except DeviceNotFoundError as e:
        problems.append(
            ConfigProblem(w, str(e), field="stream_vc.input_device_index")
        )
    try:
        resolve_stream_vc_output_device(sv)
    except DeviceNotFoundError as e:
        problems.append(
            ConfigProblem(w, str(e), field="stream_vc.output_device_index")
        )
    return problems
```
`_CHECKERS` リストに追加:
```python
    _check_subtitle,
    _check_stream_vc,
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_preflight.py -v`
Expected: PASS (3 passed)
Run(既存 preflight 無回帰): `uv run pytest tests/ -k preflight -v`
Expected: 既存 preflight テスト PASS(`_check_vc` の抽出リファクタで挙動不変)。
Run(型/lint): `uv run ty check vspeech/preflight.py vspeech/lib/audio.py && uv run ruff check vspeech/preflight.py vspeech/lib/audio.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/preflight.py vspeech/lib/audio.py tests/test_stream_vc_preflight.py
git commit -m "feat(stream-vc): preflight + device resolve for [stream_vc] (ADR-0038/0045)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: StreamPacket + transport(interface + in-process)

**Files:**
- Create: `vspeech/stream_vc/__init__.py`(空)
- Create: `vspeech/stream_vc/packet.py`
- Create: `vspeech/stream_vc/transport.py`
- Test: `tests/test_stream_vc_transport.py`

**Interfaces:**
- Consumes: (なし — base のみ)
- Produces:
  - `@dataclass class StreamPacket` — `session_id: str`, `seq: int`, `pts: float`, `pcm: bytes`, `sample_rate: int`。
  - `drop_oldest_put(q, item) -> bool` — 満杯なら最古を捨てて `item` を入れる。捨てたら `False`(遅延の単調増加を防ぐ)。
  - `class Transport(ABC)` — `async send(packet) -> bool` / `async recv() -> StreamPacket`。
  - `class InProcessTransport(Transport)` — `asyncio.Queue(maxsize=max_queued)` 実装、`send` は drop-oldest、`dropped: int` を数える。M3 で網実装に差し替わる seam(ADR-0051 tier-0)。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_transport.py`:
```python
from asyncio import Queue

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import InProcessTransport
from vspeech.stream_vc.transport import drop_oldest_put


def test_drop_oldest_put_keeps_newest():
    q: Queue[int] = Queue(maxsize=2)
    assert drop_oldest_put(q, 1) is True
    assert drop_oldest_put(q, 2) is True
    assert drop_oldest_put(q, 3) is False  # full -> drop 1, keep 3
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3


async def test_in_process_transport_send_recv_order():
    t = InProcessTransport(max_queued=4)
    for i in range(3):
        assert await t.send(StreamPacket("s", i, float(i), b"\x00\x00", 16000)) is True
    got = [(await t.recv()).seq for _ in range(3)]
    assert got == [0, 1, 2]
    assert t.dropped == 0


async def test_in_process_transport_drops_oldest_when_full():
    t = InProcessTransport(max_queued=2)
    assert await t.send(StreamPacket("s", 0, 0.0, b"", 16000)) is True
    assert await t.send(StreamPacket("s", 1, 0.0, b"", 16000)) is True
    assert await t.send(StreamPacket("s", 2, 0.0, b"", 16000)) is False
    assert t.dropped == 1
    assert (await t.recv()).seq == 1  # oldest (0) was dropped
    assert (await t.recv()).seq == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_transport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.stream_vc'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/stream_vc/__init__.py`:
```python
```
(空ファイル。)

`vspeech/stream_vc/packet.py`:
```python
"""ストリーミング VC のマシン間ユニット(ADR-0051)。

`session_id`/`seq`/`pts` を持つので consumer 側で欠落検出・整列ができる。
M2 は単一マシン in-process だが、M3 で網トランスポートへ差し替わったときに
同じ StreamPacket が流れる。
"""

from dataclasses import dataclass


@dataclass
class StreamPacket:
    session_id: str
    seq: int
    pts: float
    pcm: bytes
    sample_rate: int
```

`vspeech/stream_vc/transport.py`:
```python
"""ストリーミング VC の transport 差し替え層(ADR-0051)。

M2 は in-process(asyncio.Queue)実装のみ。producer/consumer はこの interface
の背後に置くので、M3 で網実装(UDP/TCP/bidi)へ VC・再生の他ロジックを変えずに
差し替えられる。満杯時は最古を捨てて遅延の単調増加を防ぐ(受入基準)。
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull

from vspeech.stream_vc.packet import StreamPacket


def drop_oldest_put(q: Queue, item) -> bool:
    """満杯なら最古を捨てて `item` を入れる。捨てたら False。

    capture/transport のバックプレッシャ共通処理。VC/GPU が実時間に追いつかない
    ときにキューが伸び続けるのを防ぎ、落としたことを呼び出し側が観測できるよう
    bool を返す(受入基準:遅延が単調増加せず落としたことが記録可能)。
    """
    try:
        q.put_nowait(item)
        return True
    except QueueFull:
        try:
            q.get_nowait()
        except QueueEmpty:
            pass
        try:
            q.put_nowait(item)
        except QueueFull:
            pass
        return False


class Transport(ABC):
    @abstractmethod
    async def send(self, packet: StreamPacket) -> bool:
        """packet を送る。バックプレッシャで最古を捨てたら False。"""

    @abstractmethod
    async def recv(self) -> StreamPacket:
        """次の packet を受け取る(無ければ待つ)。"""


class InProcessTransport(Transport):
    """同一プロセス内の asyncio.Queue 実装(ADR-0051 tier-0)。"""

    def __init__(self, max_queued: int) -> None:
        self._q: Queue[StreamPacket] = Queue(maxsize=max_queued)
        self.dropped = 0

    async def send(self, packet: StreamPacket) -> bool:
        ok = drop_oldest_put(self._q, packet)
        if not ok:
            self.dropped += 1
        return ok

    async def recv(self) -> StreamPacket:
        return await self._q.get()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_transport.py -v`
Expected: PASS (3 passed)
Run(型/lint): `uv run ty check vspeech/stream_vc/ && uv run ruff check vspeech/stream_vc/`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/__init__.py vspeech/stream_vc/packet.py vspeech/stream_vc/transport.py tests/test_stream_vc_transport.py
git commit -m "feat(stream-vc): StreamPacket + swappable transport (in-process, ADR-0051)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 独立マイクキャプチャ(ADR-0052)

**Files:**
- Create: `vspeech/stream_vc/capture.py`
- Test: `tests/test_stream_vc_capture.py`

**Interfaces:**
- Consumes: `StreamVcConfig`、`resolve_stream_vc_input_device`(Task 4)、`drop_oldest_put`(Task 5)、`shutdown_worker`/`worker_startup`(既存)、`telemetry`(既存)。
- Produces:
  - `ms_to_samples(ms: float, rate: int = 16000) -> int` — 純粋。
  - `pcm16_to_float32(data: bytes) -> NDArray[np.float32]` — int16 PCM → `[-1,1]`。純粋。
  - `open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream` — 16k mono int16、blocksize=hop。
  - `async def capture_loop(config: StreamVcConfig, out_queue: Queue, hop: int) -> None` — 毎 tick hop サンプル読み、float32 に直して `drop_oldest_put`、drop 時 `stream_vc_capture_drop` を記録。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_capture.py`:
```python
import numpy as np

from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.capture import pcm16_to_float32


def test_ms_to_samples():
    assert ms_to_samples(80.0) == 1280  # 80ms @ 16k
    assert ms_to_samples(10.0) == 160
    assert ms_to_samples(0.0) == 0


def test_pcm16_to_float32_range():
    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    out = pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert abs(out[1] - 1.0) < 1e-3
    assert abs(out[2] + 1.0) < 1e-3


def test_pcm16_to_float32_empty():
    out = pcm16_to_float32(b"")
    assert out.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.stream_vc.capture'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/stream_vc/capture.py`:
```python
"""streaming VC の独立マイクキャプチャ(ADR-0052)。

発話系 recording は無改変のまま、別途 16k mono でマイクを開いて固定 hop の
float32 ブロックを出す。排他デバイスで二重 open が失敗する環境向けの fan-out
フォールバックは M2 スコープ外(ADR-0052 に設計として残す)。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_input_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.transport import drop_oldest_put

CAPTURE_RATE = 16000


def ms_to_samples(ms: float, rate: int = 16000) -> int:
    """ms を rate のサンプル数へ(round)。"""
    return round(ms * rate / 1000.0)


def pcm16_to_float32(data: bytes) -> NDArray[np.float32]:
    """int16 PCM バイト列を [-1, 1] の float32 にする。"""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> sd.RawInputStream:
    device = resolve_stream_vc_input_device(config)
    logger.info("stream_vc input device %s: %s", device.index, device.name)
    stream = sd.RawInputStream(
        samplerate=CAPTURE_RATE,
        blocksize=hop,
        device=device.index,
        channels=1,
        dtype="int16",
    )
    stream.start()
    return stream


async def capture_loop(config: StreamVcConfig, out_queue: Queue, hop: int) -> None:
    """マイクから hop サンプルずつ読み、float32 ブロックを out_queue へ。"""
    with worker_startup("stream_vc"):
        stream = open_stream_vc_input_stream(config, hop)
    logger.info("stream vc capture started")
    try:
        while stream.active:
            data, overflowed = await to_thread(stream.read, hop)
            if overflowed:
                logger.warning("stream_vc capture input overflow")
            block = pcm16_to_float32(bytes(data))
            if not drop_oldest_put(out_queue, block):
                telemetry.record("stream_vc_capture_drop", 1.0)
                logger.warning("stream_vc capture queue full; dropped oldest block")
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        stream.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_capture.py -v`
Expected: PASS (3 passed)
Run(型/lint、audio extra 要): `uv run --all-extras ty check vspeech/stream_vc/capture.py && uv run ruff check vspeech/stream_vc/capture.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/capture.py tests/test_stream_vc_capture.py
git commit -m "feat(stream-vc): independent 16k mic capture producer (ADR-0052)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: VC runner(StreamingVc を回す)

**Files:**
- Create: `vspeech/stream_vc/runner.py`
- Test: `tests/test_stream_vc_capture.py`(CPU で検査できるのは無いので、GPU-gated スモークは Task 9 のエントリポイントに委ねる。ここは純粋な `make_stream_packet` の単体のみ)
- Test: `tests/test_stream_vc_runner.py`

**Interfaces:**
- Consumes: `StreamVcConfig`、`StreamingVc`(Task 2)、`StreamPacket`/`Transport`(Task 5)、`ms_to_samples`(Task 6)、`build_rvc_runtime` 相当のランタイム構築(vc.py の rvc_worker と同じ手順)、`telemetry`/`worker_startup`/`shutdown_worker`。
- Produces:
  - `make_stream_packet(session_id, seq, hop_seconds, pcm, sample_rate) -> StreamPacket` — 純粋(pts = seq * hop_seconds)。
  - `build_stream_vc_runtime(sv_config: StreamVcConfig) -> dict` — device / hubert / session / f0_session / metadata。
  - `make_streaming_vc(rt: dict, sv_config: StreamVcConfig) -> StreamingVc`。
  - `async def vc_loop(sv_config, in_queue: Queue, transport: Transport, session_id: str) -> None`。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_runner.py`:
```python
from vspeech.stream_vc.runner import make_stream_packet


def test_make_stream_packet_pts_is_seq_times_hop():
    p = make_stream_packet("sess", 5, 0.08, b"\x01\x02", 40000)
    assert p.session_id == "sess"
    assert p.seq == 5
    assert abs(p.pts - 0.4) < 1e-9  # 5 * 0.08
    assert p.pcm == b"\x01\x02"
    assert p.sample_rate == 40000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.stream_vc.runner'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/stream_vc/runner.py`:
```python
"""streaming VC の変換ループ(ADR-0053)。

capture の float32 ブロックを StreamingVc(固定ブロック+左文脈+等電力
クロスフェード)で変換し、StreamPacket にして transport へ送る。モデルの構築は
発話系 rvc_worker(vspeech/worker/vc.py)と同じ手順を [stream_vc.rvc] から行う
(発話系は無改変)。重い import は関数内。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from time import perf_counter
from typing import Any

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport


def make_stream_packet(
    session_id: str, seq: int, hop_seconds: float, pcm: bytes, sample_rate: int
) -> StreamPacket:
    """seq/pts 付きの StreamPacket(pts = seq * hop_seconds)。"""
    return StreamPacket(
        session_id=session_id,
        seq=seq,
        pts=seq * hop_seconds,
        pcm=pcm,
        sample_rate=sample_rate,
    )


def build_stream_vc_runtime(sv_config: StreamVcConfig) -> dict[str, Any]:
    """[stream_vc.rvc] から device + モデル + metadata を構築する。"""
    import json

    from vspeech.config import F0ExtractorType
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.onnx_session import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    rvc = sv_config.rvc
    device, device_name = get_device(rvc.gpu_id, rvc.gpu_name)
    logger.info("stream_vc device: %s, %s", device, device_name)
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc.hubert_model_file, device=device, is_half=half_available
    )
    session = create_session(rvc.model_file, device)
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        f0_session = create_session(rvc.rmvpe_model_file, device)
    elif rvc.f0_extractor_type == F0ExtractorType.fcpe:
        f0_session = create_session(rvc.fcpe_model_file, device)
    else:
        f0_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc,
        "device": device,
        "hubert_model": hubert_model,
        "session": session,
        "f0_session": f0_session,
        "target_sample_rate": metadata["samplingRate"],
        "f0_enabled": metadata["f0"],
        "emb_output_layer": metadata.get("embOutputLayer", 9),
        "use_final_proj": metadata.get("useFinalProj", True),
    }


def make_streaming_vc(rt: dict[str, Any], sv_config: StreamVcConfig):
    from vspeech.lib.stream_vc import StreamingVc

    return StreamingVc(
        rvc_config=rt["rvc_config"],
        device=rt["device"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_session=rt["f0_session"],
        target_sample_rate=rt["target_sample_rate"],
        f0_enabled=rt["f0_enabled"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        block_len=ms_to_samples(sv_config.block_ms),
        context_len=ms_to_samples(sv_config.context_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
    )


async def vc_loop(
    sv_config: StreamVcConfig,
    in_queue: Queue,
    transport: Transport,
    session_id: str,
) -> None:
    """capture ブロックを変換し StreamPacket として transport へ送る。"""
    with worker_startup("stream_vc"):
        from vspeech.worker.vc import check_cuda_provider

        rt = build_stream_vc_runtime(sv_config)
        check_cuda_provider(rt["session"].get_providers())
        sv = make_streaming_vc(rt, sv_config)
    await to_thread(sv.warmup)
    logger.info("stream vc worker started")
    hop_seconds = sv_config.block_ms / 1000.0
    sample_rate = rt["target_sample_rate"]
    seq = 0
    try:
        while True:
            block = await in_queue.get()
            t0 = perf_counter()
            out_i16 = await to_thread(sv.process_block, block)
            telemetry.record("stream_vc", perf_counter() - t0)
            packet = make_stream_packet(
                session_id, seq, hop_seconds, out_i16.tobytes(), sample_rate
            )
            if not await transport.send(packet):
                telemetry.record("stream_vc_send_drop", 1.0)
            seq += 1
    except CancelledError as e:
        raise shutdown_worker(e)
```

> 注: `check_cuda_provider` は `vspeech/worker/vc.py` の既存関数を import 再利用する(発話系 worker の**編集ではない** — 関数の参照利用は無改変制約に反しない)。

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_runner.py -v`
Expected: PASS (1 passed)
Run(型/lint): `uv run --all-extras ty check vspeech/stream_vc/runner.py && uv run ruff check vspeech/stream_vc/runner.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/runner.py tests/test_stream_vc_runner.py
git commit -m "feat(stream-vc): VC runner loop wiring StreamingVc to transport

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: 連続再生 + 欠落観測

**Files:**
- Create: `vspeech/stream_vc/playback.py`
- Test: `tests/test_stream_vc_playback.py`

**Interfaces:**
- Consumes: `StreamVcConfig`、`Transport`(Task 5)、`resolve_stream_vc_output_device`(Task 4)、`telemetry`/`worker_startup`/`shutdown_worker`。
- Produces:
  - `detect_gap(prev_seq: int | None, seq: int) -> int` — 欠落パケット数(`seq-prev_seq-1` が正のときその値、それ以外 0)。純粋。
  - `open_stream_vc_output_stream(config, sample_rate: int) -> sd.RawOutputStream`。
  - `async def playback_loop(config: StreamVcConfig, transport: Transport) -> None` — recv → 出力へ blocking write。seq の欠落を `stream_vc_gap` で記録(M2 は穴埋めせず観測のみ)。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_playback.py`:
```python
from vspeech.stream_vc.playback import detect_gap


def test_detect_gap_none_prev():
    assert detect_gap(None, 0) == 0


def test_detect_gap_contiguous():
    assert detect_gap(4, 5) == 0


def test_detect_gap_missing():
    assert detect_gap(4, 7) == 2  # 5, 6 missing


def test_detect_gap_reorder_or_dup_is_zero():
    assert detect_gap(7, 5) == 0  # out-of-order/dup -> not a forward gap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_playback.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.stream_vc.playback'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/stream_vc/playback.py`:
```python
"""streaming VC のローカル連続再生。

transport から受けた変換音声を出力デバイスへ連続 write する。M2 は単一マシン
in-process なので欠落は起きないが、seq の飛びは観測して記録する(受入基準:
欠落を黙って無音の穴にしない)。実際の穴埋め/整列は網が絡む M3 の担当。
"""

from asyncio import CancelledError
from asyncio import to_thread

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.audio import resolve_stream_vc_output_device
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.transport import Transport


def detect_gap(prev_seq: int | None, seq: int) -> int:
    """前 seq から見た欠落パケット数(前進の飛びのみ、並べ替え/重複は 0)。"""
    if prev_seq is None:
        return 0
    missing = seq - prev_seq - 1
    return missing if missing > 0 else 0


def open_stream_vc_output_stream(
    config: StreamVcConfig, sample_rate: int
) -> sd.RawOutputStream:
    device = resolve_stream_vc_output_device(config)
    logger.info("stream_vc output device %s: %s", device.index, device.name)
    stream = sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        device=device.index,
        dtype="int16",
    )
    stream.start()
    return stream


async def playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    """transport から受けて連続再生する。最初のパケットで出力ストリームを開く。"""
    stream: sd.RawOutputStream | None = None
    prev_seq: int | None = None
    try:
        while True:
            packet = await transport.recv()
            if stream is None:
                with worker_startup("stream_vc"):
                    stream = open_stream_vc_output_stream(config, packet.sample_rate)
                logger.info("stream vc playback started")
            gap = detect_gap(prev_seq, packet.seq)
            if gap > 0:
                telemetry.record("stream_vc_gap", float(gap))
                logger.warning("stream_vc playback gap: %d packet(s) missing", gap)
            prev_seq = packet.seq
            await to_thread(stream.write, packet.pcm)
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            stream.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_playback.py -v`
Expected: PASS (4 passed)
Run(型/lint): `uv run --all-extras ty check vspeech/stream_vc/playback.py && uv run ruff check vspeech/stream_vc/playback.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/playback.py tests/test_stream_vc_playback.py
git commit -m "feat(stream-vc): local continuous playback with gap telemetry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: サブシステム配線 + main フック + エントリポイント smoke

**Files:**
- Create: `vspeech/stream_vc/subsystem.py`
- Modify: `vspeech/main.py`(`vc` 分岐の後に 1 分岐追加)
- Test: `tests/test_stream_vc_entrypoint.py`

**Interfaces:**
- Consumes: `SharedContext`(既存)、`InProcessTransport`(Task 5)、`capture_loop`/`ms_to_samples`(Task 6)、`vc_loop`(Task 7)、`playback_loop`(Task 8)、`TransportType`(Task 3)。
- Produces:
  - `create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task` — capture/vc/playback を内側 TaskGroup で束ねた 1 タスクを登録する(routing の外 = `add_worker` 不使用)。

- [ ] **Step 1: Write the failing test**

`tests/test_stream_vc_entrypoint.py`:
```python
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_create_stream_vc_task_imports():
    # CPU import smoke: subsystem module must import without torch/rvc/audio extras.
    from vspeech.stream_vc.subsystem import create_stream_vc_task

    assert callable(create_stream_vc_task)


_STREAM_ENV = "VSPEECH_STREAM_VC_CONFIG"
_stream_config = os.environ.get(_STREAM_ENV)


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _cuda_available()
    or _stream_config is None
    or not Path(_stream_config).exists(),
    reason=f"CUDA / ${_STREAM_ENV} (real mic+speaker+model config) not available",
)
def test_entrypoint_boots_stream_vc():
    # エントリポイントを実際に起動する(「テストだけでなくエントリポイントを走らせる」)。
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "vspeech", "--config", str(_stream_config)],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    booted = False
    try:
        assert proc.stdout is not None
        import time

        deadline = time.time() + 120
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if "stream vc worker started" in line:
                booted = True
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert booted, "entrypoint did not reach 'stream vc worker started'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_entrypoint.py::test_create_stream_vc_task_imports -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.stream_vc.subsystem'`

- [ ] **Step 3: Write minimal implementation**

`vspeech/stream_vc/subsystem.py`:
```python
"""streaming VC サブシステムの配線(ADR-0050)。

Command/routing の外の自己完結サブシステム。capture(独立マイク)→ 変換 →
transport → 連続再生 を内側 TaskGroup で束ね、1 タスクとして起動する。
`context.add_worker`/`sender_queue` は使わない(発話系 routing に一切載らない)。
重い import(sounddevice/torch を引く capture/runner/playback)は起動時に遅延
させ、このモジュール自体は CPU から import できるようにする。
"""

from asyncio import CancelledError
from asyncio import Queue
from asyncio import Task
from asyncio import TaskGroup
from typing import Any
from uuid import uuid4

from vspeech.exceptions import shutdown_worker
from vspeech.shared_context import SharedContext


async def _stream_vc_subsystem(context: SharedContext) -> None:
    from vspeech.stream_vc.capture import capture_loop
    from vspeech.stream_vc.capture import ms_to_samples
    from vspeech.stream_vc.playback import playback_loop
    from vspeech.stream_vc.runner import vc_loop
    from vspeech.stream_vc.transport import InProcessTransport

    sv_config = context.config.stream_vc
    hop = ms_to_samples(sv_config.block_ms)
    session_id = uuid4().hex
    capture_queue: Queue[Any] = Queue(maxsize=sv_config.max_queued_blocks)
    transport = InProcessTransport(max_queued=sv_config.max_queued_blocks)
    try:
        async with TaskGroup() as tg:
            tg.create_task(
                capture_loop(sv_config, capture_queue, hop), name="stream_vc_capture"
            )
            tg.create_task(
                vc_loop(sv_config, capture_queue, transport, session_id),
                name="stream_vc_runner",
            )
            tg.create_task(
                playback_loop(sv_config, transport), name="stream_vc_playback"
            )
    except CancelledError as e:
        raise shutdown_worker(e)


def create_stream_vc_task(tg: TaskGroup, context: SharedContext) -> Task[None]:
    return tg.create_task(_stream_vc_subsystem(context), name="stream_vc")
```

`vspeech/main.py` の `if config.vc.enable:` ブロックの直後(`except* WorkerStartupError` の前)に追記:
```python
            if config.stream_vc.enable:
                from vspeech.stream_vc.subsystem import create_stream_vc_task

                create_stream_vc_task(tg=tg, context=context)
```

- [ ] **Step 4: Run test to verify it passes**

Run(CPU import smoke、常時): `uv run pytest tests/test_stream_vc_entrypoint.py::test_create_stream_vc_task_imports -v`
Expected: PASS。
Run(エントリポイント、実機のみ): `VSPEECH_STREAM_VC_CONFIG=/path/to/stream_vc.toml uv run --all-extras pytest tests/test_stream_vc_entrypoint.py -v`
Expected: 実機=PASS(`stream vc worker started` 到達)/ それ以外=SKIPPED。
Run(型/lint): `uv run ty check vspeech/stream_vc/subsystem.py vspeech/main.py && uv run ruff check vspeech/stream_vc/subsystem.py vspeech/main.py`
Expected: エラー無し。

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/subsystem.py vspeech/main.py tests/test_stream_vc_entrypoint.py
git commit -m "feat(stream-vc): wire capture->vc->playback subsystem + main hook (ADR-0050)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Wrap-up(実装後・別枠)

- **`poe check` 緑を確認**(`uv run --all-extras poe check`): ruff format/check・ty・pytest・uv lock・audit。既存 accepted 例外(torch CVE 等)以外の新規赤が無いこと。
- **`config.toml.example` に `[stream_vc]` を追記**(全設定を文書化する規約)。block_ms/context_ms/crossfade_ms の意味、fcpe 推奨(短入力 OK+安い、[[fcpe-onnx-availability]])、既定 disable、入出力デバイス・transport_type・max_queued_blocks を例示。実機ではマイク(録音系と同一デバイスの二重 open が排他になる環境ではフォールバックが要る、ADR-0052)とスピーカを指定。
- **実機(対象 GPU=5060Ti、block 80–160ms+fcpe)で耳確認**: `VSPEECH_STREAM_VC_CONFIG` を実 config に向け `python -m vspeech --config` で起動し、マイクへ発話 → **ブロック境界のクリック無し・ピッチ連続**を耳で確認する。詰まり/drop は telemetry(`stream_vc_capture_drop` / `stream_vc_send_drop` / `stream_vc_gap` / `stream_vc`)で観測。
- **ADR-0053 を Accepted に昇格**(耳確認で crossfade 連続性が裏づいたら)。Status 行を更新し、docs/adr/README.md の index も追随。**耳確認前は Proposed のまま**にする(数値/型だけでは昇格しない — [[dont-close-with-known-problems]])。
- **メモリ `rvc-streaming-vc.md` を M2 結果で更新**: crossfade 実装済み・耳確認結果・次は M3(2 マシン分割/jitter)。
- `superpowers:finishing-a-development-branch` で結線(`poe check` 緑・実機耳確認済みを確認してから)。ブランチは M1 から継続の `feat/rvc-streaming-vc`(未 push)。

---

## Self-Review

**Spec coverage(`2026-07-22-rvc-streaming-vc-split-machine-design.md` の M2 + 全体受入基準に対して):**
- 「マイク→ステートフル VC→ローカル連続再生が in-process で通る」→ Task 6(capture)+ Task 7(runner)+ Task 8(playback)+ Task 9(subsystem)。✓
- 「ブロック境界のクリック無し・ピッチ連続を耳で確認」→ Task 1/2(等電力クロスフェード overlap-add)+ Wrap-up(実機耳確認)。✓
- 「変換音声のみが渡り連続再生」→ Task 5(StreamPacket=pcm のみ)+ Task 8。✓
- 「追いつかない場合でも遅延が単調増加せず落としたことが観測可能」→ Task 5(drop_oldest_put)+ Task 6(`stream_vc_capture_drop`)+ Task 7(`stream_vc_send_drop`)。✓
- 「欠落・並べ替えを再生側が検出して記録(無音の穴を黙って作らない)」→ Task 8(`detect_gap` + `stream_vc_gap`)。穴埋めは M3(単機では欠落せず、観測のみ)。✓
- 「トランスポート方式を設定で切替可、切替時に他ロジック変更不要」→ Task 3(`transport_type`)+ Task 5(`Transport` ABC + in-process 実装)。網実装は M3。✓
- 「起動時に必須リソースが無ければ明確に失敗、GUI 起動前 readiness に反映」→ Task 4(`_check_stream_vc` + `_CHECKERS` 登録、GUI は `collect_problems` 再利用で自動追従=ADR-0045、GUI 編集不要)。✓
- 「streaming 設定は発話系 VC と独立」→ Task 3(`[stream_vc]` 独立セクション、入れ子 `rvc`)。✓
- 「発話系は無改変で並走」→ Global Constraints(recording/vc/routing 編集禁止、capture は独立 open=ADR-0052)。✓
- 「streaming 無効時は前後で全体挙動同一」→ 全 `[stream_vc]` は `enable=False` 既定、main.py は enable 時のみ分岐、StreamingVc の crossfade_len 既定 0 で M1 path 無改変。✓

**Placeholder scan:** TODO/TBD 無し。全ステップに実コード。Task 2 Step 1 の「削除する仮テスト」注記は明示的に除外指示済み。Task 4 の `name_key` 誤トリックは注記で素直な文字列に訂正済み。✓

**Type consistency:**
- `StreamingVc.__init__` の新引数 `crossfade_len: int = 0`(Task 2)は `make_streaming_vc`(Task 7)と GPU スモーク(Task 2)の呼び出しと一致。既定 0 で M1 ハーネス/既存スモーク無改変。
- `StreamPacket` フィールド(`session_id/seq/pts/pcm/sample_rate`, Task 5)は `make_stream_packet`(Task 7)・`detect_gap` 呼び出し側(Task 8, `packet.seq`)・transport テストと一致。
- `Transport.send -> bool` / `recv -> StreamPacket`(Task 5)は `vc_loop`(Task 7, `await transport.send`)・`playback_loop`(Task 8, `await transport.recv`)と一致。
- `resolve_stream_vc_input_device` / `resolve_stream_vc_output_device`(Task 4)は capture(Task 6)・playback(Task 8)・preflight(Task 4)で同一シグネチャ使用。
- `ms_to_samples`(Task 6)は runner(Task 7)・subsystem(Task 9)で使用、`(ms, rate=16000) -> int` 一致。
- `StreamVcConfig` フィールド(Task 3)は preflight(Task 4)・capture/runner/playback/subsystem の参照(`block_ms`/`context_ms`/`crossfade_ms`/`max_queued_blocks`/`transport_type`/device 群)と一致。
- `build_stream_vc_runtime` の戻り dict キー(Task 7)は `make_streaming_vc`(Task 7)の参照キーと一致(`rvc_config`/`device`/`hubert_model`/`session`/`f0_session`/`target_sample_rate`/`f0_enabled`/`emb_output_layer`/`use_final_proj`)。

**注意点(実装者向け):**
- `create_stream_vc_task` の CPU import 制約: `subsystem.py` は top-level で capture/runner/playback を import しないこと(sounddevice=audio extra, torch=rvc extra を引くため)。`_stream_vc_subsystem` 内での遅延 import を厳守(Task 9 のコードはそうなっている)。テスト `test_create_stream_vc_task_imports` がこれを守らせる。
