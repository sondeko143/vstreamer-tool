# streaming VC lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** streaming VC の emit 読み出し位置を設定量だけ手前へずらして未来文脈(右文脈)を増やせるようにし、その効果をバッチ経路との比較で測れるようにする。

**Architecture:** `StreamingVc._emit_with_crossfade` の読み出し位置 `nominal` から lookahead 分を引く。同時に呼び出し側が解析窓を `context_ms + lookahead_ms` に伸ばすので、左文脈は痩せない。`emit_delay_samples` は `ctx_out - nominal` なので自動で追従し、VAD ゲートと入力エンベロープのマスク整列もそのまま効く — ただし両者が持つ「前ブロック 1 個分」の履歴では足りなくなるので、直近 K ブロックへ一般化する。効果測定は、同じモデルで streaming とバッチ `change_voice` を回して log-mel 距離を出すオフラインスクリプトで行う。

**Tech Stack:** Python 3.14 / uv / numpy / torch + torchaudio + onnxruntime-gpu (rvc extra) / pydantic v2 / pytest (asyncio_mode=auto) / ruff / ty / poethepoet

## Global Constraints

- Python は **3.14 のみ** (`requires-python = ">=3.14,<3.15"`)。パッケージ管理は uv。
- コメントと docstring は **英語**。ユーザーが読む文字列 — ログ、例外メッセージ、`config.py` の `description=`、argparse の `help=`/`description=` — は **日本語**のまま (ADR-0064)。
- import は **1 行 1 つ** (ruff `force-single-line = true`)。
- `gate.py` / `envelope.py` / `lib/stream_vc.py` では **numpy をメソッド内で import** する (モジュール import を軽く保つ既存規約)。
- pydantic は **v2 API のみ**。`Field(...)`, `model_validator(mode="after")` など。v1 API 禁止。
- 既定 `lookahead_ms = 0.0` のとき **出力は導入前とビット単位で同一**。既存テストが無改変で通ることがその担保。
- 検査コマンド: `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` / `uv run pytest`。
- 終了コードはパイプ越しに見ない (`cmd | tail` の `$?` は tail のもの)。pytest は完全な node ID で指定する。
- 環境は `uv sync --all-extras` 済みであること (`--extra rvc` 単独は他の extra を外す)。

---

## File Structure

| ファイル | 責務 | 変更 |
|---|---|---|
| `vspeech/lib/stream_vc.py` | 固定ブロック変換の幾何。読み出し位置に lookahead 項を足す | 変更 |
| `vspeech/stream_vc/gate.py` | VAD マスクの生成と重畳。履歴を K ブロックへ一般化 | 変更 |
| `vspeech/stream_vc/envelope.py` | 入力包絡の整形と重畳。履歴を K ブロックへ一般化 | 変更 |
| `vspeech/stream_vc/runner.py` | 設定から `StreamingVc` を組む配線 + 起動ログ | 変更 |
| `vspeech/config.py` | `lookahead_ms` フィールド | 変更 |
| `config.toml.example` | 設定例 | 変更 |
| `scripts/stream_vc_lookahead_eval.py` | 効果測定 (純ロジック + GPU 経路) | 新規 |
| `scripts/stream_vc_rtf.py` | `_load_wav_16k` を公開名へ改名 (2 人目の利用者が出るため) | 変更 |
| `scripts/tests/test_stream_vc_lookahead_eval.py` | 整列・距離の純ロジックのテスト | 新規 |
| `poe_tasks.toml` | `stream-vc-lookahead-eval` タスク | 変更 |
| `docs/adr/0072-stream-vc-lookahead.md` | 決定記録 | 新規 |
| `docs/adr/README.md` | 索引 | 変更 |

Task 1 は幾何、Task 2/3 は下流の整列、Task 4 は配線、Task 5/6 は測定、Task 7 は記録。**Task 2 と 3 は Task 1 に依存しない**（遅延一般への対応なので単独でレビュー可能）。Task 4 は Task 1 に依存する。Task 6 は Task 1 と 4 に依存する。

---

### Task 1: `StreamingVc` の読み出し位置に lookahead を入れる

**Files:**
- Modify: `vspeech/lib/stream_vc.py:176-241` (`__init__`), `vspeech/lib/stream_vc.py:367-487` (`_emit_with_crossfade`)
- Test: `tests/test_stream_vc.py:26-49` (`_bare_streaming_vc` にパラメータ追加), 末尾に新規テスト 3 本

**Interfaces:**
- Consumes: なし
- Produces: `StreamingVc.__init__(..., crossfade_len: int = 0, sola_search_len: int = 0, lookahead_len: int = 0)`。`lookahead_len` は **16kHz 入力サンプル数**。`self.lookahead_len: int` が公開属性として読める。`emit_delay_samples` の意味は不変 (出力レートのサンプル数)。

- [ ] **Step 1: テストヘルパに `lookahead_len` を通す**

`tests/test_stream_vc.py` の `_bare_streaming_vc` に引数を足す（既存呼び出しは既定 0 でそのまま動く）。

```python
def _bare_streaming_vc(
    *,
    block_len: int = 2560,
    context_len: int = 8000,
    crossfade_len: int = 400,
    sola_search_len: int = 80,
    target_sample_rate: int = 48000,
    lookahead_len: int = 0,
):
    """A StreamingVc that drives only `_emit_with_crossfade`, with no model and no GPU.

    `__init__` requires torch and the rvc extra, so a bare instance is built with just the
    needed attributes filled in by hand (to pin the emit-length contract on CPU alone).
    """
    from vspeech.lib.stream_vc import StreamingVc

    sv = object.__new__(StreamingVc)
    sv.block_len = block_len
    sv.context_len = context_len
    sv.crossfade_len = crossfade_len
    sv.sola_search_len = sola_search_len
    sv.lookahead_len = lookahead_len
    sv.target_sample_rate = target_sample_rate
    sv._xfade_cache = None
    sv._output_tail = None
    return sv
```

- [ ] **Step 2: 失敗するテストを 3 本書く**

`tests/test_stream_vc.py` の末尾に追加。

```python
def test_lookahead_zero_reads_from_the_unchanged_nominal_position():
    """lookahead_len=0 moves the read position by not one sample (bit-identical output)."""
    sr, block_len, ctx_len = 48000, 2560, 8000
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    out = np.arange(out_total, dtype=np.int16)
    sv = _bare_streaming_vc(target_sample_rate=sr, lookahead_len=0)
    sv._emit_with_crossfade(out)
    out_hop = round(block_len * sr / 16000)
    out_xf = round(400 * sr / 16000)
    out_sola = round(80 * sr / 16000)
    expected_nominal = out_total - out_hop - out_xf - out_sola
    ctx_out = round(ctx_len * sr / 16000)
    assert sv.emit_delay_samples == ctx_out - expected_nominal


def test_lookahead_delays_the_emit_by_exactly_that_much():
    """Raising the lookahead delays the emit by exactly that much; the emit length is
    unchanged."""
    sr, block_len, ctx_len = 48000, 2560, 8000
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    out = np.arange(out_total, dtype=np.int16)
    expected_hop = round(block_len * sr / 16000)
    delays: dict[float, int] = {}
    for look_ms in (0.0, 40.0, 80.0, 160.0):
        sv = _bare_streaming_vc(
            target_sample_rate=sr, lookahead_len=round(look_ms * 16)
        )
        emitted = [sv._emit_with_crossfade(out).shape[0] for _ in range(3)]
        # the rate lock is not affected by the lookahead
        assert set(emitted) == {expected_hop}
        delays[look_ms] = sv.emit_delay_samples
    for look_ms in (40.0, 80.0, 160.0):
        out_look = round(round(look_ms * 16) * sr / 16000)
        assert delays[look_ms] - delays[0.0] == out_look


def test_lookahead_buys_right_context_one_for_one():
    """The window left beyond the emit end (= right context) grows by exactly the
    lookahead."""
    sr, block_len, ctx_len = 48000, 2560, 8000
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    out = np.arange(out_total, dtype=np.int16)
    ctx_out = round(ctx_len * sr / 16000)
    out_hop = round(block_len * sr / 16000)
    rights: dict[float, int] = {}
    for look_ms in (0.0, 160.0):
        sv = _bare_streaming_vc(
            target_sample_rate=sr, lookahead_len=round(look_ms * 16)
        )
        sv._emit_with_crossfade(out)
        # usable end of the render and the emit end, both relative to the block start
        usable_end = out_total - ctx_out
        emit_end = out_hop - 1 - sv.emit_delay_samples
        rights[look_ms] = usable_end - emit_end
    # the default (lookahead 0) leaves about 30ms of right context
    assert 0 < rights[0.0] < round(0.035 * sr)
    assert rights[160.0] - rights[0.0] == round(round(160.0 * 16) * sr / 16000)


def test_a_large_lookahead_with_the_extended_window_never_trips_the_guard():
    """With the window extended by the lookahead, no lookahead is too large.

    The lookahead cancels out of the read-position condition, so the effective ceiling is
    latency and RTF alone (ADR-0072). If this broke, preflight would need a new check --
    it is the load-bearing property of the design.
    """
    sr, block_len, ctx_ms = 48000, 2560, 500.0
    for look_ms in (0.0, 160.0, 500.0, 2000.0):
        ctx_len = round((ctx_ms + look_ms) * 16)
        out_total = round((ctx_len + block_len - 320) * sr / 16000)
        sv = _bare_streaming_vc(
            block_len=block_len,
            context_len=ctx_len,
            target_sample_rate=sr,
            lookahead_len=round(look_ms * 16),
        )
        emitted = sv._emit_with_crossfade(np.arange(out_total, dtype=np.int16))
        assert emitted.shape[0] == round(block_len * sr / 16000)
```

- [ ] **Step 3: テストが落ちることを確認**

```
uv run pytest tests/test_stream_vc.py::test_lookahead_zero_reads_from_the_unchanged_nominal_position tests/test_stream_vc.py::test_lookahead_delays_the_emit_by_exactly_that_much tests/test_stream_vc.py::test_lookahead_buys_right_context_one_for_one tests/test_stream_vc.py::test_a_large_lookahead_with_the_extended_window_never_trips_the_guard -v
```
期待: 4 本とも FAIL（`_bare_streaming_vc` に `lookahead_len` が無い、または `self.lookahead_len` 未定義で AttributeError）。

- [ ] **Step 4: `__init__` に `lookahead_len` を足す**

`vspeech/lib/stream_vc.py` の `__init__` シグネチャ末尾に追加し、属性を保存し、`_xfade_cache` の型注釈を 4 int に広げる。

```python
        crossfade_len: int = 0,
        sola_search_len: int = 0,
        lookahead_len: int = 0,
    ) -> None:
```

`self.sola_search_len = sola_search_len` の直後に:

```python
        # How many input samples earlier than the tail anchor to read the emit from.
        # Buying right context this way costs exactly this much extra latency; the caller
        # is expected to extend context_len by the same amount so the left context at the
        # emit start does not shrink (ADR-0072).
        self.lookahead_len = lookahead_len
```

`_xfade_cache` の注釈:

```python
        self._xfade_cache: (
            tuple[int, int, int, int, NDArray[np.float32], NDArray[np.float32]] | None
        ) = None
```

- [ ] **Step 5: `_emit_with_crossfade` の読み出し位置を変える**

キャッシュ構築ブロック内、`out_sola = round(...)` の直後に `out_look` を足す。

```python
            out_hop = round(self.block_len * r / 16000)
            out_xf = round(self.crossfade_len * r / 16000)
            out_sola = round(self.sola_search_len * r / 16000)
            out_look = round(self.lookahead_len * r / 16000)
```

`out_sola` のクランプ行の直後に、lookahead の成立条件を **fail-loud** で検査する（黙ってクランプすると、測定者が設定した量と実際の量がずれて A/B が無意味になる）。

```python
            # nominal - out_sola >= 0 must still hold with the lookahead subtracted. The
            # caller extends context_len by the lookahead, so out_total grows by the same
            # amount and this can only trip on a hand-built geometry -- fail loud rather
            # than clamp, or the measured lookahead would silently differ from the
            # configured one.
            if out_total - out_hop - out_xf - 2 * out_sola - out_look < 0:
                raise ValueError(
                    f"lookahead ({out_look}) が描画長に対して大きすぎる "
                    f"(out_total={out_total} hop={out_hop} xf={out_xf} "
                    f"sola={out_sola}): lookahead_ms を減らすか context_ms を"
                    "増やすこと。"
                )
```

キャッシュの保存と展開に `out_look` を通す。

```python
            self._xfade_cache = (out_hop, out_xf, out_sola, out_look, fade_in, fade_out)
        out_hop, out_xf, out_sola, out_look, fade_in, fade_out = self._xfade_cache
```

読み出し位置:

```python
        nominal = out_total - out_hop - out_xf - out_sola - out_look
```

- [ ] **Step 6: docstring を更新**

`_emit_with_crossfade` の docstring の "Index invariants" に 1 項追加する。

```
        - `lookahead_len` shifts `nominal` earlier by exactly `out_look`, so the emitted
          content is that much older and every emitted sample gains that much right
          context. It never changes the emit length, so the rate lock is untouched; the
          cost is exactly `out_look` of extra latency. `lookahead_len == 0` reproduces the
          pre-lookahead read position sample for sample.
```

- [ ] **Step 7: テストが通ることを確認（新規 + 既存の回帰）**

```
uv run pytest tests/test_stream_vc.py -v
```
期待: 新規 4 本 PASS、既存の `test_emit_with_crossfade_hop_is_realtime_clock_not_render_ratio` / `test_emit_delay_is_the_offset_from_the_block_start` / `test_emit_delay_does_not_move_with_the_sola_lag` / `test_emit_with_crossfade_raises_when_output_shorter_than_hop` が**無改変で** PASS（これがビット同一の担保）。

- [ ] **Step 8: 静的検査**

```
uv run ruff format .
uv run ruff check .
uv run ty check
```
それぞれ exit 0（`ty` は既存の受容済み 2 件以外に新規診断が無いこと）。

- [ ] **Step 9: Commit**

```bash
git add vspeech/lib/stream_vc.py tests/test_stream_vc.py
git commit -m "feat(stream-vc): read the emit lookahead_len earlier to buy right context"
```

---

### Task 2: VAD ゲートのマスク履歴を K ブロックへ一般化

**Files:**
- Modify: `vspeech/stream_vc/gate.py:77-113` (`__init__` / `reset`), `vspeech/stream_vc/gate.py:150-230` (`apply`)
- Test: `tests/test_stream_vc_gate.py` 末尾に新規テスト 3 本

**Interfaces:**
- Consumes: なし（Task 1 に依存しない）
- Produces: `StreamingVadGate.apply(out_i16, gains, delay_samples, sample_rate) -> NDArray[np.int16]` — シグネチャ不変。内部状態が `_prev_gains: NDArray | None` から `_history: list[tuple[NDArray[np.float64], int]]`（古い順、要素は `(window_gains, emit_len)`）に変わる。

**なぜ要るか:** マスクは絶対サンプル格子に置かれる。履歴が 1 ブロックだと最古の窓中心は `0.5*step - n` なので、`delay_samples > n - 0.5*step`（= 既定 40kHz で 144ms）を超えると emit の頭が中心より左に落ち、`prev[0]` にクランプされてマスクが追従しなくなる。既定 delay は 50ms なので lookahead 94ms で頭打ちになる。

- [ ] **Step 1: 失敗するテストを 3 本書く**

`tests/test_stream_vc_gate.py` の末尾（`# --- the capture-reopen sentinel ---` の前、純ロジックテスト群の直後）に追加。

```python
def test_apply_reaches_two_blocks_back_when_the_delay_exceeds_one_hop():
    """With a delay past one hop, the emit's head carries the mask from two blocks back.

    With only one block of history the head falls left of the oldest window centre and
    clamps to the previous block's first value, i.e. the mask stops tracking. Lookahead
    puts the geometry in exactly that region.
    """
    gate = _gate(threshold=0.5, hangover_ms=0.0, min_gain=0.0)
    rate, n = 40000, 6400  # 160ms @40k
    delay = 9000  # past one hop (6400) = about 65ms of lookahead
    ones = np.full(n, 10000, dtype=np.int16)
    # block 0: all speech / block 1: all silence / block 2: all silence
    gate.apply(ones.copy(), gate.window_gains(np.full(5, 0.9)), delay, rate)
    gate.apply(ones.copy(), gate.window_gains(np.zeros(5)), delay, rate)
    out = gate.apply(ones.copy(), gate.window_gains(np.zeros(5)), delay, rate)
    g = out.astype(np.float64) / 10000.0
    # the head carries audio from two blocks back (speech), so the gate is open
    assert g[0] > 0.9
    # the tail has come all the way down to the silence side
    assert g[-1] < 0.05


def test_apply_is_unchanged_by_the_history_generalisation_at_the_default_delay():
    """At the default delay the history generalisation changes nothing: adding knots to
    the left of the evaluated range cannot move an np.interp value."""
    rate, n, delay = 40000, 6400, 2000
    ones = np.full(n, 10000, dtype=np.int16)
    seq = [np.full(5, 0.9), np.zeros(5), np.full(5, 0.9), np.zeros(5)]
    got = []
    gate = _gate(threshold=0.5, hangover_ms=300.0, min_gain=0.0)
    for probs in seq:
        got.append(gate.apply(ones.copy(), gate.window_gains(probs), delay, rate).copy())
    # expected: the one-block-history algorithm, computed by hand
    ref_gate = _gate(threshold=0.5, hangover_ms=300.0, min_gain=0.0)
    step = 512 * rate / 16000
    prev = None
    for probs, expected in zip(seq, got, strict=True):
        gains = ref_gate.window_gains(probs)
        base = np.full(max(1, ceil(n / step)), 0.0) if prev is None else prev
        prev = gains
        centers = np.concatenate(
            [
                (np.arange(base.shape[0], dtype=np.float64) + 0.5) * step - n,
                (np.arange(gains.shape[0], dtype=np.float64) + 0.5) * step,
            ]
        )
        gain = np.interp(
            np.arange(n, dtype=np.float64) - delay,
            centers,
            np.concatenate([base, gains]),
        )
        ref = np.clip(
            np.rint(np.full(n, 10000, dtype=np.int16).astype(np.float32) * gain),
            -32768.0,
            32767.0,
        ).astype(np.int16)
        assert np.array_equal(expected, ref)
```

```python
def test_apply_has_no_gain_step_at_the_seam_with_a_long_delay():
    """With a delay past one hop, the gain still does not step at a block boundary
    (= no click).

    Pins that adding history does not break seam continuity. The gain ramps across a 32ms
    window, so even a full swing moves at about 1/1280 per sample.
    """
    gate = _gate(threshold=0.5, hangover_ms=0.0, min_gain=0.0)
    rate, n, delay = 40000, 6400, 9000
    ones = np.full(n, 10000, dtype=np.int16)
    pattern = [np.zeros(5), np.full(5, 0.9), np.full(5, 0.9), np.zeros(5), np.zeros(5)]
    curve = [
        gate.apply(ones.copy(), gate.window_gains(probs), delay, rate).astype(np.float64)
        / 10000.0
        for probs in pattern
    ]
    full = np.concatenate(curve)
    assert float(np.abs(np.diff(full)).max()) < 0.01
```

`tests/test_stream_vc_gate.py` の先頭 import に `from math import ceil` を足す。

- [ ] **Step 2: テストが落ちることを確認**

```
uv run pytest tests/test_stream_vc_gate.py::test_apply_reaches_two_blocks_back_when_the_delay_exceeds_one_hop -v
```
期待: FAIL（`g[0]` が 2 ブロック前ではなく前ブロックの先頭値にクランプされ 0 付近になる）。

- [ ] **Step 3: 状態を履歴に置き換える**

`__init__` の `self._prev_gains: NDArray[np.float64] | None = None` を差し替え。

```python
        # The masks of the most recent blocks, oldest first, as (window_gains, emit_len)
        # pairs. One block is enough while the emit delay stays below a hop, but lookahead
        # (ADR-0072) pushes the delay past it, and then the head of the emit carries audio
        # decided two or more blocks ago. The length is derived per call from the delay,
        # so it self-sizes and stays at one block for the pre-lookahead geometry.
        self._history: list[tuple[NDArray[np.float64], int]] = []
```

`reset()` の `self._prev_gains = None` を `self._history = []` に差し替え、docstring 中の「前ブロックのマスク」を「マスク履歴」に直す。

- [ ] **Step 4: `apply` を履歴対応にする**

`apply` の docstring 内、`_prev_gains` を参照している段落を次の文面に差し替える。

```
        The first `delay_samples` of the emit correspond to **earlier** input blocks, so
        the masks of the most recent blocks (`_history`) are concatenated on the left
        before interpolating -- which simultaneously guarantees gain continuity across the
        block boundary (no step = no click). How many blocks are needed follows from the
        delay: `ceil((delay_samples + step/2) / emit_len)`, which is one block for the
        pre-lookahead geometry and grows as `lookahead_ms` pushes the delay past a hop
        (ADR-0072). That continuity only holds while `delay_samples` is constant across
        ticks, which is why `StreamingVc` publishes the **nominal** delay, excluding
        SOLA's lag (ADR-0059).
```

`prev = self._prev_gains` から `all_gains = np.concatenate([prev, gains])` までの本体を次で置き換える（`step` の算出行はそのまま残す）。

```python
        # How many blocks of history the delay reaches back into. Constant across ticks in
        # practice (the delay is nominal, hence fixed), and exactly 1 for the geometry
        # that existed before lookahead -- which is why this is a no-op there.
        need = max(1, ceil((delay_samples + step / 2.0) / n))
        history = self._history[-need:]
        if len(history) < need:
            # No previous information (right after startup or a reset). The head of the
            # emit is audio rendered from before the real-time jump, or from a zeros
            # context, so start from the closed state (min_gain) -- matching
            # `_since_speech`'s initial value. Seed **a hop's worth of windows, not one**:
            # with a single element its centre lands a whole hop earlier, so the ramp is
            # handed over across a hop instead of one window and the head never fully
            # closes. Count the windows the same way the real mask does (ceil): round
            # would give one window fewer when the block length is not a multiple of the
            # window length (block_ms=80), shifting the last seed centre earlier.
            seed = np.full(max(1, ceil(n / step)), self.min_gain, dtype=np.float64)
            history = [(seed, n)] * (need - len(history)) + history
        self._history.append((gains, n))
        del self._history[:-need]
        if float(gains.min()) == 1.0 and all(
            float(g.min()) == 1.0 for g, _ in history
        ):
            return out_i16
        # Each history block's origin is its own emit length earlier, accumulated. Using
        # the stored emit length rather than assuming it equals `n` keeps a length change
        # from silently shifting an origin (the same discipline as envelope.py). Note the
        # window count times the window length is NOT the block length: speech_probs
        # zero-pads to ceil(block_len/512) windows, so at block_ms=80 the windows total
        # more than the block.
        centers_parts: list[NDArray[np.float64]] = []
        gains_parts: list[NDArray[np.float64]] = []
        offset = 0.0
        for past_gains, past_len in reversed(history):
            offset -= past_len
            centers_parts.append(
                (np.arange(past_gains.shape[0], dtype=np.float64) + 0.5) * step + offset
            )
            gains_parts.append(past_gains)
        centers_parts.reverse()
        gains_parts.reverse()
        centers_parts.append((np.arange(gains.shape[0], dtype=np.float64) + 0.5) * step)
        gains_parts.append(gains)
        centers = np.concatenate(centers_parts)
        all_gains = np.concatenate(gains_parts)
```

`apply` 冒頭の早期 return（`n == 0 or gains.shape[0] == 0`）と末尾の `np.interp` 以降は変更しない。

- [ ] **Step 5: テストが通ることを確認**

```
uv run pytest tests/test_stream_vc_gate.py -v
```
期待: 新規 3 本 PASS、既存 20 本超が**無改変で** PASS（特に `test_apply_is_bit_identical_when_every_window_is_open` / `test_apply_keeps_the_head_closed_on_the_first_block_after_reset` / `test_apply_places_the_previous_block_one_hop_back_not_one_window_grid_back` / `test_apply_ramps_across_a_window_without_a_step` / `test_reset_closes_the_gate_and_drops_the_previous_mask`）。

- [ ] **Step 6: Commit**

```bash
git add vspeech/stream_vc/gate.py tests/test_stream_vc_gate.py
git commit -m "fix(stream-vc): carry the vad mask over K blocks so a long emit delay stays aligned"
```

---

### Task 3: 入力エンベロープの shape 履歴を K ブロックへ一般化

**Files:**
- Modify: `vspeech/stream_vc/envelope.py:46-81` (`__init__` / `reset`), `vspeech/stream_vc/envelope.py:83-187` (`apply`)
- Test: `tests/test_stream_vc_envelope.py` 末尾に新規テスト 2 本

**Interfaces:**
- Consumes: なし（Task 1 に依存しない）
- Produces: `StreamingEnvelope.apply(out_i16, in_block, delay_samples) -> NDArray[np.int16]` — シグネチャ不変。内部状態が `_prev_shape` / `_prev_len` から `_history: list[tuple[NDArray[np.float64], int]]`（古い順、`(shape, emit_len)`）に変わる。

- [ ] **Step 1: 失敗するテストを 2 本書く**

`tests/test_stream_vc_envelope.py` の末尾に追加。ファイル先頭のヘルパ `_env(**kw)`（既定を上書きして `StreamingEnvelope` を作る）と `_block(level, n)` をそのまま使う。

```python
def test_shape_reaches_two_blocks_back_when_the_delay_exceeds_one_emit():
    """With a delay past one emit length, the head carries the shape from two blocks back.

    With only one block of history the head falls left of the oldest frame centre and
    clamps to the previous block's first value. Lookahead puts the geometry in exactly
    that region.
    """
    env = _env(strength=1.0, min_gain=0.0, max_gain=1.0)
    out_len = 6400
    delay = 9000  # past one emit length (6400)
    loud = _block(0.2, n=2560)
    quiet = _block(0.002, n=2560)
    ones = np.full(out_len, 10000, dtype=np.int16)
    env.apply(ones.copy(), loud, delay)
    env.apply(ones.copy(), quiet, delay)
    got = env.apply(ones.copy(), quiet, delay)
    g = got.astype(np.float64) / 10000.0
    # the head is audio from two blocks back (loud), so it is not ducked
    assert g[0] > 0.5
    # the tail has come down to the quiet level
    assert g[-1] < 0.2


def test_gain_is_continuous_across_the_seam_with_a_long_delay():
    """With a delay past one emit length, the gain still does not step at a block boundary
    (extends ADR-0065's guarantee across the history generalisation)."""
    env = _env(strength=1.0, min_gain=0.0, max_gain=1.0)
    out_len, delay = 6400, 9000
    ones = np.full(out_len, 10000, dtype=np.int16)
    curve = [
        env.apply(ones.copy(), _block(level, n=2560), delay).astype(np.float64) / 10000.0
        for level in (0.02, 0.02, 0.3, 0.3, 0.05, 0.02)
    ]
    full = np.concatenate(curve)
    assert float(np.abs(np.diff(full)).max()) < 0.02
```

- [ ] **Step 2: テストが落ちることを確認**

```
uv run pytest tests/test_stream_vc_envelope.py::test_shape_reaches_two_blocks_back_when_the_delay_exceeds_one_emit -v
```
期待: FAIL（`g[0]` が 2 ブロック前ではなく前ブロック（quiet）の先頭値にクランプされ 0.5 以下）。

- [ ] **Step 3: 状態を履歴に置き換える**

`__init__` の `_prev_shape` / `_prev_len` の 2 行を差し替え。

```python
        # The shapes of the most recent blocks, oldest first, as (shape, emit_len) pairs.
        # One block suffices while the emit delay stays below one emit length; lookahead
        # (ADR-0072) pushes it past that, and then the head of the emit carries audio
        # shaped two or more blocks ago. The length is derived per call from the delay.
        self._history: list[tuple[NDArray[np.float64], int]] = []
```

`reset()` の該当 2 行を `self._history = []` に差し替え、docstring の「前ブロックの shape」を「shape 履歴」に直す。

- [ ] **Step 4: `apply` を履歴対応にする**

`prev_shape, prev_len = self._prev_shape, self._prev_len` の行を削除し、`ref < 1e-8` の分岐と shape 確定部を次で置き換える。

```python
        # Output samples per input frame, and half a frame -- the margin the seam
        # continuity needs (see the bounds note below).
        half_frame = out_len / n_frames / 2.0
        need = max(1, math.ceil((delay_samples + half_frame) / out_len))
        history = self._history[-need:]
        if len(history) < need:
            # Startup, or right after a reset. The head of the emit is rendered from a
            # zeros context or from before a real-time jump, so hand over from **unity** --
            # the same "the first block is not ducked" cold start as `_ema_level`. Seed a
            # whole emit's worth of frames (not one): with a single element its centre
            # would land a whole emit earlier and stretch the ramp over two blocks.
            seed = np.ones(n_frames, dtype=np.float64)
            history = [(seed, out_len)] * (need - len(history)) + history
        # effectively digital silence (e.g. pure silence right after init) -> pass through
        if ref < 1e-8:
            # This block went out at unity, so hand unity over: leaving the older shape in
            # place would make the next block step off a value that was never applied.
            self._history.append((np.ones(n_frames, dtype=np.float64), out_len))
            del self._history[:-need]
            return out_i16
        # The relative shape (relative to the reference, not mean~1), linearly
        # interpolated onto the emit's sample grid.
        shape_now = frame_rms / ref
        self._history.append((shape_now, out_len))
        del self._history[:-need]
```

続いて `centers` の構築を置き換える（`n_prev` を使う 2 行と `centers = np.concatenate([...])` を削除）。

```python
        # Frame centres on the emit's absolute sample grid. Each history block sits its
        # own emit length earlier, accumulated, which is what makes the gain continuous
        # across the seam: with the delay correction the seam falls in the interior of the
        # shape, where both blocks interpolate the *same* two frame centres with the same
        # values (ADR-0065). The emit length is carried per block rather than assumed
        # equal to `out_len` so a length change cannot silently shift an origin.
        centers_parts: list[NDArray[np.float64]] = []
        shape_parts: list[NDArray[np.float64]] = []
        offset = 0.0
        for past_shape, past_len in reversed(history):
            offset -= past_len
            k = past_shape.shape[0]
            centers_parts.append(
                (np.arange(k, dtype=np.float64) + 0.5) / k * past_len + offset
            )
            shape_parts.append(past_shape)
        centers_parts.reverse()
        shape_parts.reverse()
        centers_parts.append(
            (np.arange(n_frames, dtype=np.float64) + 0.5) / n_frames * out_len
        )
        shape_parts.append(shape_now)
        centers = np.concatenate(centers_parts)
```

`np.interp` 呼び出しの第 3 引数を `np.concatenate([prev_shape, shape_now])` から `np.concatenate(shape_parts)` に変える。

- [ ] **Step 5: 「大きすぎる delay」の注記を書き換える**

`np.interp` 直前のコメントブロックのうち、"Too large" の項を次に差し替える（"Too small" 側はそのまま残す）。

```
        # - Too large: the history now grows with the delay, so the head no longer clamps
        #   to the oldest frame. What remains bounded is the reference EMA: shaping audio
        #   from `need` blocks ago against a reference that has since moved on gets less
        #   apt as the lookahead grows. envelope_ema_ms (2000ms by default) is far longer
        #   than any usable lookahead, so this stays a second-order effect.
```

- [ ] **Step 6: テストが通ることを確認**

```
uv run pytest tests/test_stream_vc_envelope.py -v
```
期待: 新規 2 本 PASS、既存 9 本が**無改変で** PASS（特に `test_gain_is_continuous_across_the_block_seam` / `test_a_dropped_carry_does_not_shift_the_shape_of_the_next_block` / `test_empty_and_silent_passthrough`）。

- [ ] **Step 7: Commit**

```bash
git add vspeech/stream_vc/envelope.py tests/test_stream_vc_envelope.py
git commit -m "fix(stream-vc): carry the envelope shape over K blocks for long emit delays"
```

---

### Task 4: `lookahead_ms` 設定と配線、起動ログ

**Files:**
- Modify: `vspeech/config.py:465-470` (`sola_search_ms` の直後), `vspeech/stream_vc/runner.py:220-242` (`make_streaming_vc`), `vspeech/stream_vc/runner.py:275-282` (`vc_loop` の起動ログ), `config.toml.example:219-221` (`sola_search_ms` の直後)
- Test: `tests/test_stream_vc_config.py`, `tests/test_stream_vc_runner.py`

**Interfaces:**
- Consumes: Task 1 の `StreamingVc.__init__(..., lookahead_len: int)`
- Produces: `StreamVcConfig.lookahead_ms: float`（既定 `0.0`, `ge=0`）。`make_streaming_vc(rt, sv_config)` が `context_len = ms_to_samples(context_ms + lookahead_ms)` と `lookahead_len = ms_to_samples(lookahead_ms)` を渡す。`geometry_summary(sv_config: StreamVcConfig, emit_delay_samples: int, target_sample_rate: int) -> str`（純粋、日本語 1 行）。

- [ ] **Step 1: 失敗するテストを 3 本書く**

`tests/test_stream_vc_config.py` の末尾:

```python
def test_lookahead_defaults_to_zero_and_rejects_negative():
    import pytest
    from pydantic import ValidationError

    assert StreamVcConfig().lookahead_ms == 0.0
    with pytest.raises(ValidationError):
        StreamVcConfig(lookahead_ms=-1.0)
```

`tests/test_stream_vc_runner.py` の末尾:

```python
def test_make_streaming_vc_extends_the_context_by_the_lookahead(monkeypatch):
    """The analysis window is passed extended by the lookahead (so the left context
    does not shrink)."""
    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc import runner as runner_mod

    captured: dict[str, object] = {}

    class _Spy:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("vspeech.lib.stream_vc.StreamingVc", _Spy)
    sv = StreamVcConfig(context_ms=500.0, lookahead_ms=160.0, block_ms=160.0)
    rt = {
        "rvc_config": sv.rvc,
        "device": None,
        "hubert_model": None,
        "session": None,
        "f0_session": None,
        "target_sample_rate": 40000,
        "f0_enabled": True,
        "emb_output_layer": 9,
        "use_final_proj": True,
    }
    runner_mod.make_streaming_vc(rt, sv)
    assert captured["context_len"] == round((500.0 + 160.0) * 16)
    assert captured["lookahead_len"] == round(160.0 * 16)
    # at the default (0) the window length is unchanged
    captured.clear()
    runner_mod.make_streaming_vc(rt, StreamVcConfig(context_ms=500.0))
    assert captured["context_len"] == round(500.0 * 16)
    assert captured["lookahead_len"] == 0


def test_geometry_summary_reports_the_window_and_both_delays():
    """The startup log carries the window, the emit delay, and the added latency."""
    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc.runner import geometry_summary

    sv = StreamVcConfig(context_ms=500.0, block_ms=160.0, lookahead_ms=160.0)
    line = geometry_summary(sv, emit_delay_samples=8400, target_sample_rate=40000)
    assert "解析窓 820ms" in line  # 500 + 160 + 160
    assert "emit 遅延 210.0ms" in line  # 8400 / 40000
    assert "付加遅延 160ms" in line
```

- [ ] **Step 2: テストが落ちることを確認**

```
uv run pytest tests/test_stream_vc_config.py::test_lookahead_defaults_to_zero_and_rejects_negative tests/test_stream_vc_runner.py::test_make_streaming_vc_extends_the_context_by_the_lookahead tests/test_stream_vc_runner.py::test_geometry_summary_reports_the_window_and_both_delays -v
```
期待: 3 本とも FAIL（`lookahead_ms` が未定義で `ValidationError`／`KeyError`、`geometry_summary` が `ImportError`）。

- [ ] **Step 3: config フィールドを足す**

`vspeech/config.py` の `sola_search_ms` フィールド定義の直後に追加。

```python
    lookahead_ms: float = Field(
        default=0.0,
        ge=0,
        description="emit の読み出し位置を手前へずらして未来文脈(右文脈)を増やす ms。"
        "既定 0 は導入前と完全に同一の出力。既定構成の右文脈は 30ms しかなく、"
        "HuBERT は窓全体に attention を張る双方向モデルなのでここが品質の天井に"
        "なっている。増やした分だけ片道遅延が増え、解析窓も同じだけ伸びる"
        "(context_ms の左文脈は保たれる)ので推論コストも上がる。"
        "`uv run poe stream-vc-lookahead-eval` で測ってから決めること",
    )
```

- [ ] **Step 4: `make_streaming_vc` を配線する**

`vspeech/stream_vc/runner.py` の `StreamingVc(...)` 呼び出しで `context_len` を差し替え、`lookahead_len` を追加。

```python
        block_len=ms_to_samples(sv_config.block_ms),
        context_len=ms_to_samples(sv_config.context_ms + sv_config.lookahead_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
        sola_search_len=ms_to_samples(sv_config.sola_search_ms),
        lookahead_len=ms_to_samples(sv_config.lookahead_ms),
    )
```

同じ関数の docstring 冒頭コメント（`rvc.quality` の段落）の後ろに 1 文足す。

```python
    # The analysis window is extended by lookahead_ms on top of context_ms, so that
    # buying right context does not eat into the left context the emit start sees
    # (ADR-0072). context_ms therefore keeps exactly the meaning it had before.
```

- [ ] **Step 5: 起動ログを足す（純粋ヘルパ経由）**

`vspeech/stream_vc/runner.py` に、`make_streaming_vc` の直後へヘルパを追加する。`runner.py` は
`make_stream_packet` / `apply_input_boost` / `make_stream_envelope` と同じく純粋ヘルパを切り出して
単体テストする流儀なので、それに合わせる（`vc_loop` を起動しないと文言を検証できない状態にしない）。

```python
def geometry_summary(
    sv_config: StreamVcConfig, emit_delay_samples: int, target_sample_rate: int
) -> str:
    """A one-line startup summary of the analysis window and the delays it implies.

    Pure, so the wording can be pinned without standing up a worker. Japanese because the
    reader is an operator reading the log (ADR-0064).
    """
    window_ms = sv_config.context_ms + sv_config.lookahead_ms + sv_config.block_ms
    delay_ms = emit_delay_samples * 1000.0 / target_sample_rate
    return (
        f"stream_vc geometry: 解析窓 {window_ms:.0f}ms "
        f"(context {sv_config.context_ms:.0f} + lookahead {sv_config.lookahead_ms:.0f}"
        f" + block {sv_config.block_ms:.0f}), emit 遅延 {delay_ms:.1f}ms, "
        f"lookahead による付加遅延 {sv_config.lookahead_ms:.0f}ms"
    )
```

`vc_loop` の `logger.info("stream vc worker started")` の**直前**に 1 行挿入する（warmup 済みなので
`emit_delay_samples` は確定している。`sample_rate` はこの時点でまだ束縛されていないので
`rt["target_sample_rate"]` を直接使う）。

```python
    logger.info(
        "%s", geometry_summary(sv_config, sv.emit_delay_samples, rt["target_sample_rate"])
    )
```

- [ ] **Step 6: `config.toml.example` に追記**

`sola_search_ms = 5.0` の直後に挿入。

```toml
# 先読み(右文脈)ms。0 で無効 = 導入前と完全に同一の出力。
# 既定構成では emit は入力ブロックの [-50,+110]ms を覆い、emit 開始点の左文脈 450ms に
# 対し右文脈は 30ms しか残らない。HuBERT は窓全体に attention を張る双方向モデルなので、
# この非対称が streaming とバッチ変換の品質差の主因になっている (ADR-0072)。
# 増やすと読み出し位置がその分手前へ動き、右文脈が 1:1 で増える。代償は
#   - 片道遅延がちょうど同じだけ増える
#   - 解析窓が context_ms + lookahead_ms + block_ms へ伸びるので推論コストが上がる
#     (左文脈は痩せない = context_ms の意味は変わらない)
# 既定値は実機で決めること:
#   uv run poe stream-vc-lookahead-eval --config ./config.toml --input voice.wav
# これはバッチ経路との log-mel 距離の表と、各設定の WAV を出す。RTF ハーネスと違って
# **[stream_vc.rvc] をそのまま読む**ので、[rvc] へ写す必要は無い。
lookahead_ms = 0.0
```

- [ ] **Step 7: テストが通ることを確認**

```
uv run pytest tests/test_stream_vc_config.py tests/test_stream_vc_runner.py tests/test_config_stream_vc.py tests/test_preflight.py tests/test_stream_vc_preflight.py -v
```
期待: 全 PASS。

- [ ] **Step 8: 静的検査**

```
uv run ruff format .
uv run ruff check .
uv run ty check
```

- [ ] **Step 9: Commit**

```bash
git add vspeech/config.py vspeech/stream_vc/runner.py config.toml.example tests/test_stream_vc_config.py tests/test_stream_vc_runner.py
git commit -m "feat(stream-vc): add lookahead_ms and extend the analysis window to match"
```

---

### Task 5: 効果測定の純ロジック（整列と距離）

**Files:**
- Create: `scripts/stream_vc_lookahead_eval.py`（この Task では純ロジックのみ）
- Create: `scripts/tests/test_stream_vc_lookahead_eval.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `frame_energy(x: NDArray[np.float32], hop: int) -> NDArray[np.float64]`
  - `best_offset(ref: NDArray, test: NDArray, hint: int, search: int) -> int`
  - `align_offset(ref: NDArray[np.float32], test: NDArray[np.float32], hint: int, coarse_hop: int = 256, coarse_search: int = 64, fine_search: int = 240, excerpt: int = 80000) -> int`
  - `spectral_distance(ref_logmel: NDArray[np.float64], test_logmel: NDArray[np.float64], floor_db: float = -40.0) -> tuple[float, float]`（mean, p95）

- [ ] **Step 1: 失敗するテストを書く**

`scripts/tests/test_stream_vc_lookahead_eval.py` を新規作成。

```python
import numpy as np

from scripts.stream_vc_lookahead_eval import align_offset
from scripts.stream_vc_lookahead_eval import frame_energy
from scripts.stream_vc_lookahead_eval import spectral_distance


def _speechlike(n: int, seed: int = 0) -> np.ndarray:
    """A signal with a speech-like envelope, so the coarse search has something to lock
    onto."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n) * 0.2
    env = np.abs(np.sin(np.linspace(0.0, 12.0 * np.pi, n)))
    return (base * env).astype(np.float32)


def test_frame_energy_is_per_hop_rms():
    x = np.concatenate([np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)])
    got = frame_energy(x, hop=4)
    assert got.shape == (2,)
    assert got[0] == 0.0
    assert abs(got[1] - 1.0) < 1e-12


def test_frame_energy_of_a_short_signal_is_empty():
    assert frame_energy(np.ones(3, dtype=np.float32), hop=4).shape == (0,)


def test_align_offset_recovers_a_known_shift():
    ref = _speechlike(60000)
    shift = 3111
    test = np.concatenate([np.zeros(shift, dtype=np.float32), ref])
    assert align_offset(ref, test, hint=3000) == shift


def test_align_offset_recovers_a_shift_far_from_the_hint():
    ref = _speechlike(60000, seed=2)
    shift = 9000
    test = np.concatenate([np.zeros(shift, dtype=np.float32), ref])
    # the coarse stage finds it even when the hint is a whole block off
    assert align_offset(ref, test, hint=2400) == shift


def test_spectral_distance_is_zero_for_identical_input():
    rng = np.random.default_rng(1)
    lm = rng.standard_normal((80, 300)) * 2.0 - 3.0
    assert spectral_distance(lm, lm) == (0.0, 0.0)


def test_spectral_distance_equals_a_uniform_offset():
    lm = np.zeros((80, 100))
    mean, p95 = spectral_distance(lm, lm + 3.0)
    assert abs(mean - 3.0) < 1e-9
    assert abs(p95 - 3.0) < 1e-9


def test_spectral_distance_ignores_frames_below_the_floor():
    lm = np.full((80, 10), -100.0)
    lm[:, 0] = 0.0  # the only frame with energy
    test = lm.copy()
    test[:, 1:] = 50.0  # wreck the silent frames only
    mean, _ = spectral_distance(lm, test, floor_db=-40.0)
    assert mean == 0.0
```

- [ ] **Step 2: テストが落ちることを確認**

```
uv run pytest scripts/tests/test_stream_vc_lookahead_eval.py -v
```
期待: FAIL（`ModuleNotFoundError: No module named 'scripts.stream_vc_lookahead_eval'`）。

- [ ] **Step 3: 純ロジックを実装する**

`scripts/stream_vc_lookahead_eval.py` を新規作成。

```python
"""Measure what lookahead buys, against the batch path as the reference (ADR-0072).

Converts one wav both ways with **the same model** -- streaming at several
`lookahead_ms` settings, and the batch `change_voice` that has full two-sided context --
then reports a log-mel distance per setting plus the wavs for an ear A/B. The batch output
is a ceiling to approach, not perceptual ground truth, so the numbers rank the settings
and the wavs decide.

Unlike `stream_vc_rtf.py`, this reads **[stream_vc.rvc]** directly, so there is no need to
mirror it into [rvc].

    uv sync --all-extras
    uv run poe stream-vc-lookahead-eval --config ./config.toml --input voice.wav \
        --lookahead 0,40,80,160 --out-dir ./lookahead_eval

The helpers above `main` are pure numpy so they can be unit tested on CPU with no model
(scripts/tests/test_stream_vc_lookahead_eval.py); everything touching torch lives below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def frame_energy(x: NDArray[np.float32], hop: int) -> NDArray[np.float64]:
    """Per-hop RMS, used for the coarse alignment search."""
    n = x.shape[0] // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = x[: n * hop].astype(np.float64).reshape(n, hop)
    return np.sqrt(np.mean(frames**2, axis=1))


def best_offset(
    ref: NDArray[np.floating[Any]],
    test: NDArray[np.floating[Any]],
    hint: int,
    search: int,
) -> int:
    """The offset o in [hint-search, hint+search] maximising the normalized correlation
    between ref[:m] and test[o:o+m]. Brute force -- the caller keeps `search` small."""
    best, best_score = hint, -np.inf
    for o in range(hint - search, hint + search + 1):
        if o < 0 or o >= test.shape[0]:
            continue
        m = min(ref.shape[0], test.shape[0] - o)
        if m <= 0:
            continue
        a = ref[:m].astype(np.float64)
        b = test[o : o + m].astype(np.float64)
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 0.0:
            continue
        score = float(a @ b) / den
        if score > best_score:
            best_score, best = score, o
    return best


def align_offset(
    ref: NDArray[np.float32],
    test: NDArray[np.float32],
    hint: int,
    coarse_hop: int = 256,
    coarse_search: int = 64,
    fine_search: int = 240,
    excerpt: int = 80000,
) -> int:
    """How many samples `test` lags `ref` by, searched coarse (energy envelope) then fine
    (waveform).

    A single brute-force pass over the whole signal would be O(search * n); the coarse
    stage runs on one value per `coarse_hop` samples and the fine stage only on the first
    `excerpt` samples, which keeps this a couple of seconds offline.
    """
    coarse = (
        best_offset(
            frame_energy(ref, coarse_hop),
            frame_energy(test, coarse_hop),
            hint // coarse_hop,
            coarse_search,
        )
        * coarse_hop
    )
    m = min(excerpt, ref.shape[0])
    return best_offset(ref[:m], test, coarse, fine_search)


def spectral_distance(
    ref_logmel: NDArray[np.float64],
    test_logmel: NDArray[np.float64],
    floor_db: float = -40.0,
) -> tuple[float, float]:
    """(mean, p95) of the per-frame log-mel L2 distance, over the frames the reference
    actually has energy in.

    Both arrays are (n_mels, frames) on a 10*log10 scale. Frames more than `floor_db`
    below the reference's loudest frame are excluded, so silence does not dilute the
    number into meaninglessness.
    """
    m = min(ref_logmel.shape[1], test_logmel.shape[1])
    if m == 0:
        return 0.0, 0.0
    a = ref_logmel[:, :m]
    b = test_logmel[:, :m]
    energy = a.mean(axis=0)
    mask = energy >= (energy.max() + floor_db)
    if not mask.any():
        mask = np.ones_like(energy, dtype=bool)
    d = np.sqrt(np.mean((a - b) ** 2, axis=0))[mask]
    return float(d.mean()), float(np.percentile(d, 95))


def write_wav(path: Path, samples: NDArray[np.int16], rate: int) -> None:
    """Write mono int16 PCM. stdlib `wave` -- no torchaudio backend dependency."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
```

- [ ] **Step 4: テストが通ることを確認**

```
uv run pytest scripts/tests/test_stream_vc_lookahead_eval.py -v
```
期待: 7 本すべて PASS。

- [ ] **Step 5: 静的検査**

```
uv run ruff format .
uv run ruff check .
uv run ty check
```

- [ ] **Step 6: Commit**

```bash
git add scripts/stream_vc_lookahead_eval.py scripts/tests/test_stream_vc_lookahead_eval.py
git commit -m "feat(scripts): pure alignment + log-mel distance helpers for lookahead eval"
```

---

### Task 6: 効果測定スクリプトの GPU 経路と poe タスク

**Files:**
- Modify: `scripts/stream_vc_lookahead_eval.py`（`main` と GPU 経路を追記）
- Modify: `scripts/stream_vc_rtf.py:403` (`_load_wav_16k` → `load_wav_16k`), `scripts/stream_vc_rtf.py:353`（呼び出し側）
- Modify: `poe_tasks.toml`（`stream-vc-rtf` の定義の直後）

**Interfaces:**
- Consumes: Task 1 の `StreamingVc(..., lookahead_len)`、Task 4 の `StreamVcConfig.lookahead_ms`、Task 5 の `align_offset` / `spectral_distance` / `write_wav`
- Produces: CLI のみ（他タスクが import しない）

- [ ] **Step 1: `_load_wav_16k` を公開名にする**

`scripts/stream_vc_rtf.py` で定義名を `load_wav_16k` に改め、同ファイル内の唯一の呼び出し（`signal = _load_wav_16k(args.wav)`）を `load_wav_16k(args.wav)` に直す。docstring はそのまま。

- [ ] **Step 2: 改名の回帰を確認**

```
uv run pytest scripts/tests/ -v
uv run ruff check scripts/stream_vc_rtf.py
```
期待: PASS / exit 0（`_load_wav_16k` への参照が残っていないこと）。

- [ ] **Step 3: GPU 経路と `main` を追記**

`scripts/stream_vc_lookahead_eval.py` の `write_wav` の後ろに追記。

```python
def load_config_and_runtime(config_path: Path) -> tuple[Any, dict[str, Any]]:
    """Read the config and build the streaming runtime from **[stream_vc.rvc]**.

    Reusing `build_stream_vc_runtime` means the reference and every streaming run share
    one model load, and that the comparison is against the same weights the streaming path
    would really use.
    """
    from vspeech.config import Config
    from vspeech.stream_vc.runner import build_stream_vc_runtime

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    return config.stream_vc, build_stream_vc_runtime(config.stream_vc)


def run_batch_reference(
    rt: dict[str, Any], signal_16k: NDArray[np.float32], seed: int
) -> NDArray[np.int16]:
    """The batch `change_voice` over the whole signal = the two-sided-context ceiling.

    `run_change_voice` takes the same runtime dict shape `build_stream_vc_runtime`
    produces, so no second model load is needed. Seeded, because the RVC synthesizer is
    stochastic by design but reproducible under a seed.
    """
    from scripts.capture_change_voice_golden import run_change_voice
    from scripts.capture_change_voice_golden import seed_all

    frames = (np.clip(signal_16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    seed_all(seed)
    return run_change_voice(rt, frames, 16000)


def run_streaming(
    rt: dict[str, Any],
    sv_config: Any,
    signal_16k: NDArray[np.float32],
    lookahead_ms: float,
    seed: int,
) -> tuple[NDArray[np.int16], NDArray[np.float64], int]:
    """Convert the signal block by block at one lookahead setting.

    Returns (emitted int16, per-block seconds, emit_delay_samples).
    """
    import time

    from scripts.capture_change_voice_golden import seed_all
    from vspeech.lib.stream_vc import StreamingVc
    from vspeech.stream_vc.capture import ms_to_samples

    block_len = ms_to_samples(sv_config.block_ms)
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
        context_len=ms_to_samples(sv_config.context_ms + lookahead_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
        sola_search_len=ms_to_samples(sv_config.sola_search_ms),
        lookahead_len=ms_to_samples(lookahead_ms),
    )
    sv.warmup()
    seed_all(seed)
    emits: list[NDArray[np.int16]] = []
    durations: list[float] = []
    for i in range(signal_16k.shape[0] // block_len):
        block = signal_16k[i * block_len : (i + 1) * block_len]
        t0 = time.perf_counter()
        emits.append(sv.process_block(block))
        durations.append(time.perf_counter() - t0)
    return (
        np.concatenate(emits),
        np.array(durations, dtype=np.float64),
        sv.emit_delay_samples,
    )


def log_mel(x: NDArray[np.int16], rate: int) -> NDArray[np.float64]:
    """(n_mels, frames) log-mel on a 10*log10 scale."""
    import torch
    import torchaudio.transforms as T

    mel = T.MelSpectrogram(sample_rate=rate, n_fft=1024, hop_length=256, n_mels=80)
    spec = mel(torch.from_numpy(x.astype(np.float32) / 32768.0))
    return (10.0 * torch.log10(spec + 1e-10)).numpy().astype(np.float64)


def format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "  L(ms) right(ms)  added(ms)  window(ms)  rtf_p50  rtf_p95  "
        "align_err  logmel_mean  logmel_p95"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['lookahead_ms']:>7.0f} {r['right_ms']:>9.1f} "
            f"{r['added_ms']:>10.0f} {r['window_ms']:>11.0f} "
            f"{r['rtf_p50']:>8.2f} {r['rtf_p95']:>8.2f} "
            f"{r['align_err']:>10d} {r['logmel_mean']:>12.3f} "
            f"{r['logmel_p95']:>11.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    from scripts.stream_vc_rtf import load_wav_16k
    from scripts.stream_vc_rtf import parse_grid

    parser = argparse.ArgumentParser(
        description="lookahead ごとに streaming VC の出力をバッチ変換と比較する"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="入力 wav")
    parser.add_argument("--lookahead", default="0,40,80,160", help="lookahead_ms のリスト")
    parser.add_argument("--out-dir", type=Path, default=Path("./lookahead_eval"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    sv_config, rt = load_config_and_runtime(args.config)
    rate = rt["target_sample_rate"]
    signal = load_wav_16k(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = run_batch_reference(rt, signal, args.seed)
    write_wav(args.out_dir / "batch_reference.wav", ref, rate)
    ref_f = ref.astype(np.float32)

    # The first blocks are rendered against a zeros context, and the batch reference has
    # no such warm-up, so drop that span from both before comparing.
    skip = round((sv_config.context_ms + sv_config.block_ms) / 1000.0 * rate)
    rows: list[dict[str, Any]] = []
    for lookahead_ms in parse_grid(args.lookahead):
        stream, durations, delay = run_streaming(
            rt, sv_config, signal, lookahead_ms, args.seed
        )
        write_wav(args.out_dir / f"lookahead_{lookahead_ms:.0f}ms.wav", stream, rate)
        ref_t = ref_f[skip:]
        stream_t = stream.astype(np.float32)[skip:]
        offset = align_offset(ref_t, stream_t, hint=delay)
        aligned = stream_t[offset : offset + ref_t.shape[0]]
        m = min(aligned.shape[0], ref_t.shape[0])
        mean, p95 = spectral_distance(
            log_mel(ref_t[:m].astype(np.int16), rate),
            log_mel(aligned[:m].astype(np.int16), rate),
        )
        block_seconds = sv_config.block_ms / 1000.0
        rows.append(
            {
                "lookahead_ms": lookahead_ms,
                "right_ms": 30.0 + lookahead_ms,
                "added_ms": lookahead_ms,
                "window_ms": sv_config.context_ms + lookahead_ms + sv_config.block_ms,
                "rtf_p50": float(np.percentile(durations, 50)) / block_seconds,
                "rtf_p95": float(np.percentile(durations, 95)) / block_seconds,
                # Should land near 0: the analytic emit delay is what we aligned by.
                "align_err": int(offset - delay),
                "logmel_mean": mean,
                "logmel_p95": p95,
            }
        )
        print(f"lookahead={lookahead_ms:.0f}ms done", flush=True)

    print()
    print(format_table(rows))
    print()
    print(f"wav: {args.out_dir}  (batch_reference.wav と聞き比べること)")
    print("logmel 距離が小さいほどバッチ変換に近い。ただしこれは代理指標なので、")
    print("順位付けに使い、最終判断は wav の耳 A/B で行うこと。")
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: poe タスクを足す**

`poe_tasks.toml` の `stream-vc-rtf = { ... }` の行の直後に追加。

```toml
# lookahead(右文脈)ごとの変換品質をバッチ経路と比較する (ADR-0072)。--all-extras で
# sync 済みのプロジェクト環境で走らせる。RTF ハーネスと違い **[stream_vc.rvc] をそのまま
# 読む**ので、[rvc] へ写す必要は無い。
#   uv sync --all-extras
#   uv run poe stream-vc-lookahead-eval --config ./config.toml --input voice.wav
# log-mel 距離の表と、各設定の wav (バッチ基準を含む) を出す。
stream-vc-lookahead-eval = { cmd = "python scripts/stream_vc_lookahead_eval.py", help = "Compare streaming VC output against the batch path across lookahead settings" }
```

- [ ] **Step 5: CLI が起動することを確認（モデル不要の範囲）**

```
uv run poe stream-vc-lookahead-eval --help
```
期待: exit 0、日本語の description が化けずに出る。

- [ ] **Step 6: 静的検査と全テスト**

```
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
期待: すべて exit 0。

- [ ] **Step 7: Commit**

```bash
git add scripts/stream_vc_lookahead_eval.py scripts/stream_vc_rtf.py poe_tasks.toml
git commit -m "feat(scripts): compare streaming VC against the batch path across lookahead"
```

---

### Task 7: ADR-0072 と索引

**Files:**
- Create: `docs/adr/0072-stream-vc-lookahead.md`
- Modify: `docs/adr/README.md`（表の末尾に 1 行追加）

**Interfaces:**
- Consumes: Task 1〜6 の実装結果
- Produces: なし

- [ ] **Step 1: ADR を書く**

`docs/adr/0072-stream-vc-lookahead.md` を新規作成。実測が入るまで **Status: Proposed**。

```markdown
# 0072. streaming VC の emit 読み出し位置を lookahead ぶん手前へずらして右文脈を買う

- Status: Proposed
- Date: 2026-08-10
- Related: [spec](../superpowers/specs/2026-08-10-stream-vc-lookahead-design.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0057](0057-streaming-input-envelope-rolling-ema.md), [0059](0059-stream-vc-window-resolution-vad-gate.md), [0065](0065-streaming-envelope-seam-handover.md)

## Context

streaming VC の品質はバッチ経路に届いていない。幾何を実測すると、既定構成
(block 160ms / context 500ms / crossfade 25ms / SOLA 5ms) で emit は入力ブロックの相対時刻
**[-50, +110]ms** を覆う。emit 開始点の左文脈が 450ms あるのに対し、emit 終端より後ろに残る
右文脈は **30ms しかない**。HuBERT/ContentVec は解析窓全体に attention を張る双方向モデル
なので、この非対称はそのまま表現の質に効く。

右文脈は遅延との交換でしか増やせないが、その比率を選ぶ手段が無かった。`crossfade_ms` を
伸ばせば副作用として右文脈も増える (実測: 25ms → 70ms で右文脈 30ms → 75ms) が、同時に
「二つの別描画の和」である帯が全出力の 15.6% から 43% へ比例して広がる。SOLA で位相を
合わせてなお相関 0.82 = 無相関成分 18% を含む帯なので、品質のためのノブとしては筋が悪い。

## Decision

`lookahead_ms` を追加し、emit の読み出し位置 `nominal` をその分だけ手前へずらす。同時に
**解析窓を `context_ms + lookahead_ms` へ自動で伸ばす**ので、`context_ms` は従来どおり
「emit 開始点の左文脈」を意味したままになり、lookahead は純粋な追加になる。

この選び方には幾何上の副産物がある。読み出しの成立条件
`out_total - out_hop - out_xf - 2*out_sola - out_look >= 0` は、窓を lookahead ぶん伸ばすと
`out_total` も同じだけ増えるので **lookahead が両辺で相殺**し、既存の
`context_ms >= 20 + crossfade_ms + 2*sola_search_ms` に還元される。つまり lookahead には
幾何的な上限が無く、実効上限は遅延と RTF だけになる。preflight に新しい検査は要らない。

emit 長は実時間クロック由来のまま変えないので、ADR-0053 の index invariant とレートロックは
すべて保たれる。付加される片道遅延はちょうど `lookahead_ms`。`emit_delay_samples` は
`ctx_out - nominal` なので自動で追従し、VAD ゲート (0059) と入力エンベロープ (0057/0065) の
マスク整列はそのまま効く。

ただし両者が持っていた「前ブロック 1 個分」の履歴では足りなくなる。マスクは絶対サンプル格子に
置かれるので、履歴 1 ブロックだと最古の中心は `0.5*step - n`、すなわち delay が
`n - step/2` (既定 40kHz で 144ms) を超えた時点で emit の頭が中心より左に落ち、先頭値へ
クランプされてマスクが追従しなくなる。既定 delay 50ms に対する余裕は lookahead 94ms しかない。
そこで両者の履歴を **直近 K ブロック** (`K = ceil((delay + 半フレーム) / emit長)`) に一般化する。
K は既定構成では 1 になるので、これは lookahead を使わない限り恒等な変更である。左側に knot を
足しても `np.interp` の値は変わらないため、**既定 `lookahead_ms = 0` の出力はビット単位で不変**。

既定値は据え置き 0 とし、実機測定で決める。そのために `poe stream-vc-lookahead-eval` を用意し、
同一モデルで streaming (複数の lookahead) とバッチ `change_voice` を回して log-mel 距離の表と
各設定の wav を出す。バッチ出力は「近づきたい上限」であって知覚的な正解ではないので、数値は
順位付けに使い、最終判断は wav の耳 A/B で行う。

## Alternatives rejected

- **`crossfade_ms` を先読みの代用にする** — コード変更ゼロで右文脈は増える (25→70ms で
  30→75ms、実測) が、二重描画の和である帯が全出力の 15.6%→43% へ比例して広がり、先読みの利得を
  自分で相殺しかねない。`crossfade < block` の制約で lookahead が block_ms 未満に縛られ、
  ADR-0053 が決めたフェード則の前提 (SOLA 整列点の相関 0.82) にも触る。測定の比較対象としては
  有用なので、eval スクリプトで並べられる状態にはしてある。
- **`block_ms` を縮めて浮いた遅延を先読みに回す (合計遅延据え置き)** — block 80ms + lookahead
  80ms なら合計遅延は今と同じで右文脈 110ms になる。ただし推論レートが 2 倍 (RTF ~0.5) になり、
  ADR-0053 は block=80 を「seam のプチプチが可聴」で一度却下している。本決定と直交するので、
  必要になった時点で独立に評価する (先読みが seam のクリックを軽減する可能性もある)。
- **解析窓を固定したまま左文脈を削って右に回す** — 推論コストは完全に据え置けるが、
  ADR-0053 が耳で却下した左文脈不足の領域 (500ms 未満で「ガタゴト」) に入る。左と右のどちらが
  効くのかを同時に動かして測ることになり、切り分けができない。
- **`lookahead_ms` に恣意的な上限を設ける** — 幾何制約が相殺で消えるため、根拠のある値が無い。
  実効上限は RTF であり、それは eval スクリプトが測る。タイプミスは窓が伸びて warmup で OOM に
  なるか RTF が 1 を超えて drop が立つので、黙って壊れることはない。
- **VAD ゲート / エンベロープの履歴を 1 ブロックのまま、lookahead に上限を課す** — 94ms で
  頭打ちになり、1 ブロック分 (160ms) の先読みすら測れない。履歴の一般化は既定構成では恒等
  なので、上限を課す理由がない。

## Consequences

右文脈を遅延と引き換えに買えるようになり、その交換比率を運用者が選べる。既定は 0 なので、
この決定だけでは音は一切変わらない — 既定値の決定は実機測定に委ねた別の判断になる。

代償は推論コスト。解析窓が `context_ms + lookahead_ms + block_ms` へ伸びるので RTF が上がる
(lookahead 160ms なら窓 660→820ms)。GPU ごとに違うので、別マシンへ持っていくときは
`poe stream-vc-lookahead-eval` で測り直すこと。

VAD ゲートと入力エンベロープが K ブロックのマスク履歴を持つようになる (メモリは 1 ブロック
あたり窓数個の float64 なので無視できる)。既定構成では K=1 で従来と恒等。

品質を測る土台ができた副産物として、streaming とバッチの差を数値で追えるようになった。RTF
ハーネスと違い `[stream_vc.rvc]` をそのまま読むので、測定用に `[rvc]` へ写す手間も要らない。

**残る天井**: 右文脈を増やしても、ブロック単位で独立に f0 を抽出していること、クロスフェード帯が
二つの別描画の和であること、reset 直後にゼロ文脈から立ち上がることは変わらない。いずれも
本決定とは独立に評価できる。
```

- [ ] **Step 2: README 索引に 1 行足す**

`docs/adr/README.md` の表の末尾（0069 の行の直後）に追加。

```markdown
| [0072](0072-stream-vc-lookahead.md) | streaming VC の emit 読み出し位置を lookahead ぶん手前へずらして右文脈を買う | Proposed | 2026-08-10 |
```

- [ ] **Step 3: リンク切れが無いことを確認**

```
uv run python -c "
import re, sys
from pathlib import Path
adr = Path('docs/adr/0072-stream-vc-lookahead.md')
text = adr.read_text(encoding='utf-8')
missing = [m for m in re.findall(r'\]\(([^)]+\.md)\)', text)
           if not (adr.parent / m).exists()]
print('missing:', missing)
sys.exit(1 if missing else 0)
"
```
期待: `missing: []` と exit 0（0053 / 0057 / 0059 / 0065 と spec への相対リンクがすべて実在する）。

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0072-stream-vc-lookahead.md docs/adr/README.md
git commit -m "docs(adr): 0072 — lookahead で streaming VC の右文脈を買う"
```

---

## 実装後（この計画の外、実機が要る）

1. `uv sync --all-extras` した GPU ホストで
   `uv run poe stream-vc-lookahead-eval --config <実 config> --input <実声 wav> --json out.json`
2. 表と wav を持ち帰って耳 A/B → `lookahead_ms` の既定値を決める
3. ADR-0072 に実測を書き足し、Status を Accepted に上げる。既定値を動かすなら
   `config.py` と `config.toml.example` も同時に更新する
