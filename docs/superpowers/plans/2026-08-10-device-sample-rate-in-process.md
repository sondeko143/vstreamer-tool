# デバイス境界のサンプルレート変換をプロセス内で行う — 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

ADR: [0070](../../adr/0070-device-boundary-inhouse-polyphase-resampler.md)（デバイス境界＝自前ポリフェーズ）, [0071](../../adr/0071-device-native-rate-resolution.md)（真のデバイスレート解決）。どちらも `Proposed`、Task 10 で昇格させる。
Spec: [2026-08-10-device-sample-rate-in-process-design.md](../specs/2026-08-10-device-sample-rate-in-process-design.md)

**Goal:** オーディオデバイスをネイティブレートで開き、サンプルレート変換を OS ではなく numpy 製ポリフェーズ FIR でプロセス内に持ち込む（stream_vc の入口/出口 + 発話系の録音/再生の 4 箇所）。

**Architecture:** 新モジュール `vspeech/lib/resample.py`（有理比ポリフェーズ FIR、numpy のみ）と `vspeech/lib/pcm.py`（format 対応のデコード/飽和エンコード）を足し、4 つのデバイス境界がそれを使う。開くレートは `vspeech/lib/audio.py` の `resolve_device_rate` が決める（設定 → WASAPI の `default_samplerate` → WASAPI 同名デバイスからの逆引き）。デバイスレートと境界を流れる音声のレートが等しいときはリサンプラを作らず素通しする。

**Tech Stack:** Python 3.14 / numpy 2 / sounddevice(PortAudio) / pydantic v2 / pytest(asyncio_mode=auto) / ruff / ty

## Global Constraints

- コメントと docstring は**英語**。ユーザーが読む文字列（ログ・例外メッセージ・`config.py` の `description=`）は**日本語**（ADR-0064）。編集したファイルに日本語コメントが残っていたら同じ変更で英訳する。
- import は 1 行 1 つ（ruff `force-single-line = true`）。`from x import y` 形式を既存に合わせる。
- 型検査は `ty`。`uv run ty check`（プロジェクト全体。ファイル指定はテストの型エラーを見逃す）。
- 依存パッケージを追加しない。numpy は `ctranslate2`（base 依存）経由で必ず入っている。`scipy` / `torch` / `torchaudio` / `av` を新しく import しない。
- `vspeech/stream_vc/consumer.py` から到達するコードは **torch を import してはならない**（ADR-0055）。`tests/test_forbidden_imports.py` を壊さない。
- 検証コマンドは必ず終了コードで判定する。`cmd | tail` の `$?` は `tail` の結果なので、パイプ越しに成否を判断しない。
- pytest は完全な node ID で指定する（`-k` を使わない）。
- 実行は worktree `C:\Users\<USER>\vstreamer\vstreamer-tool\.claude\worktrees\feat-device-rate-resample` から（`git rev-parse --show-toplevel` で確認できる）。Windows コンソールで日本語が出るコマンドは `PYTHONIOENCODING=utf-8` を付ける。

---

## Implementer Authority

この plan が拘束するのは 3 つだけ: **公開契約**（各 task の「契約」欄の名前・型・方向）、
**Global Constraints の逐語値**、**各 task の受入基準**。

それ以外 — 内部設計、関数・ファイルの分割、命名、アルゴリズム、テストの設計と粒度、
エラー処理の形、依存の使い方 — はすべて実装者が決める。plan に書かれていない実装を
選んだことは逸脱ではない。

plan の記述より良い方法を見つけたら、良い方を採る。plan は使い捨てなので書き換えない。
その選択が adr-writing の基準に当たるならトリガ2 で ADR を起票する。

停止して人間に確認するのは、上の 3 つのいずれかを**変える必要がある**と判断したときだけ。

> **Task 1-4 は旧形式（実装コード込み）で実行済み。** その 4 task すべてで brief の
> コードに誤りが見つかり、実装者が訂正した。Task 5 以降はこの節の規律で実行する。

---


### Task 1: ポリフェーズリサンプラ (`vspeech/lib/resample.py`)

**Files:**
- Create: `vspeech/lib/resample.py`
- Test: `tests/test_resample.py`

**Interfaces:**
- Consumes: なし（numpy のみ）
- Produces:
  - `class PolyphaseResampler(src_rate: int, dst_rate: int, *, transition_width: float = 0.10, stopband_db: float = 80.0)`
  - `PolyphaseResampler.delay_samples: int`（出力サンプル単位の群遅延。必ず整数）
  - `PolyphaseResampler.out_len(n_in: int) -> int`
  - `PolyphaseResampler.process(x: NDArray[np.float32]) -> NDArray[np.float32]`（ストリーミング。`(n,)` と `(n, channels)` の両方）
  - `PolyphaseResampler.resample_full(x) -> NDArray[np.float32]`（ワンショット。群遅延を除去し `round(n*dst/src)` 本返す。呼び出し後の状態はリセット済み）
  - `PolyphaseResampler.reset() -> None`
  - `make_resampler(src_rate: int, dst_rate: int) -> PolyphaseResampler | None`（`src == dst` なら `None` = 素通し）

**背景（試作で実測済み。実装の前提なので信じてよい）:**
- 阻止域は 48k→16k で -90.0dB、44.1k→16k で -85.4dB、40k→48k のイメージ抑圧 -90.7dBc。
- 群遅延は 48k→16k / 44.1k→16k で 3.1ms、40k→48k で 1.3ms。
- 固定 hop（160ms）でブロック分割したときの出力は一括変換と **bit 完全一致**。不規則長・多チャンネルでは float32 の BLAS 加算順序差で最大 -122dBFS 相対の差が出る（float64 では差 0）。
- **事前充填は不要**。因果 FIR なので出力本数は `ceil(L*n/M)` で欠けず、1 device tick あたり 1 ブロックがそのまま出る（実測で配信遅れ min=max=0）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_resample.py` を新規作成:

```python
"""Numeric contract of the polyphase resampler (ADR-0070)."""

import numpy as np
import pytest

from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler

DOWN = [(48000, 16000), (44100, 16000)]
UP = [(40000, 48000), (24000, 48000)]
ALL = DOWN + UP


def _db(v: float) -> float:
    return 20.0 * np.log10(max(float(v), 1e-30))


def _tone(freq: float, rate: int, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize(("src", "dst"), DOWN)
def test_downsample_rejects_above_output_nyquist(src: int, dst: int) -> None:
    """Input above the output Nyquist must not fold back into the output."""
    worst = -999.0
    for freq in np.linspace(dst / 2, src / 2 * 0.999, 24):
        r = PolyphaseResampler(src, dst)
        y = r.process(_tone(float(freq), src))[dst // 2 :]
        worst = max(worst, _db(np.abs(y).max()))
    assert worst < -80.0, f"aliasing only {worst:.1f} dB down"


@pytest.mark.parametrize(("src", "dst"), UP)
def test_upsample_suppresses_images(src: int, dst: int) -> None:
    """No image energy above the source Nyquist in the upsampled output."""
    worst = -999.0
    for freq in np.linspace(200.0, src / 2 * 0.9, 12):
        r = PolyphaseResampler(src, dst)
        y = r.process(_tone(float(freq), src))[dst // 2 :]
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1 / dst)
        worst = max(worst, _db(spec[freqs > src / 2].max() / spec.max()))
    assert worst < -80.0, f"images only {worst:.1f} dBc down"


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_passband_is_flat(src: int, dst: int) -> None:
    """Flat to 0.9x the narrower Nyquist, within 0.5 dB."""
    nyquist = min(src, dst) / 2
    levels = []
    for freq in np.linspace(50.0, nyquist * 0.9, 20):
        r = PolyphaseResampler(src, dst)
        y = r.process(_tone(float(freq), src))[dst // 4 : -dst // 4]
        levels.append(_db(np.abs(y).max()))
    assert max(levels) - min(levels) < 0.5
    assert max(levels) < 0.2


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_fixed_hop_blocks_match_one_shot_exactly(src: int, dst: int) -> None:
    """The pipeline's own access pattern (fixed 160 ms hops) must be bit-exact.

    This is the core continuity guarantee: no discontinuity at block seams.
    """
    x = np.random.default_rng(0).standard_normal(src).astype(np.float32)
    whole = PolyphaseResampler(src, dst).process(x)
    r = PolyphaseResampler(src, dst)
    hop = int(src * 0.160)
    chunked = np.concatenate([r.process(x[i : i + hop]) for i in range(0, len(x), hop)])
    assert len(chunked) == len(whole)
    assert np.array_equal(chunked, whole)


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_irregular_blocks_match_one_shot(src: int, dst: int) -> None:
    """Arbitrary block sizes agree to float32 rounding (BLAS sums in a different
    order for a different row count; the maths is identical -- verified at -122
    dBFS relative)."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(src).astype(np.float32)
    whole = PolyphaseResampler(src, dst).process(x)
    r = PolyphaseResampler(src, dst)
    parts, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 5000))
        parts.append(r.process(x[i : i + n]))
        i += n
    chunked = np.concatenate(parts)
    assert len(chunked) == len(whole)
    assert np.allclose(chunked, whole, atol=1e-5, rtol=0)


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_same_block_sequence_is_bit_reproducible(src: int, dst: int) -> None:
    x = np.random.default_rng(11).standard_normal(src).astype(np.float32)
    hop = int(src * 0.160)
    runs = []
    for _ in range(2):
        r = PolyphaseResampler(src, dst)
        runs.append(
            np.concatenate([r.process(x[i : i + hop]) for i in range(0, len(x), hop)])
        )
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_out_len_predicts_process(src: int, dst: int) -> None:
    r = PolyphaseResampler(src, dst)
    for n in (1, 13, 100, int(src * 0.160), 3, int(src * 0.160)):
        predicted = r.out_len(n)
        assert predicted == len(r.process(np.zeros(n, dtype=np.float32)))


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_group_delay_is_an_integer_and_matches_the_impulse(src: int, dst: int) -> None:
    r = PolyphaseResampler(src, dst)
    assert isinstance(r.delay_samples, int)
    x = np.zeros(src, dtype=np.float32)
    x[src // 2] = 1.0
    peak = int(np.argmax(np.abs(r.process(x))))
    assert peak - (src // 2) * dst / src == pytest.approx(r.delay_samples, abs=1.0)


def test_fixed_hop_cadence_needs_no_priming() -> None:
    """One device tick in -> exactly one pipeline block out, from the first tick.

    A resampler that held audio back would make delivery lag by a whole block
    (measured +160 ms with soxr). The causal polyphase does not (ADR-0070).
    """
    for src in (48000, 44100):
        r = PolyphaseResampler(src, 16000)
        hop_out = 2560
        hop_in = round(hop_out * src / 16000)
        produced = 0
        for tick in range(200):
            produced += len(r.process(np.zeros(hop_in, dtype=np.float32)))
            assert produced // hop_out == tick + 1, f"{src}: lag at tick {tick}"


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_resample_full_keeps_length_and_alignment(src: int, dst: int) -> None:
    """One-shot mode returns the full duration with the group delay removed."""
    r = PolyphaseResampler(src, dst)
    n = src // 2
    t = np.arange(n) / src
    x = (np.sin(2 * np.pi * 440 * t) * np.hanning(n)).astype(np.float32)
    y = r.resample_full(x)
    assert len(y) == round(n * dst / src)
    tt = np.arange(len(y)) / dst
    ref = (np.sin(2 * np.pi * 440 * tt) * np.hanning(len(y))).astype(np.float32)
    assert _db(np.abs(y - ref).max() / np.abs(ref).max()) < -60.0


@pytest.mark.parametrize(("src", "dst"), [(48000, 16000), (40000, 48000)])
def test_multichannel_matches_per_channel(src: int, dst: int) -> None:
    x = np.random.default_rng(3).standard_normal((src, 2)).astype(np.float32)
    got = PolyphaseResampler(src, dst).process(x)
    want = np.stack(
        [
            PolyphaseResampler(src, dst).process(np.ascontiguousarray(x[:, c]))
            for c in range(2)
        ],
        axis=1,
    )
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5, rtol=0)


def test_empty_input_returns_empty() -> None:
    r = PolyphaseResampler(48000, 16000)
    assert r.process(np.zeros(0, dtype=np.float32)).shape == (0,)
    assert r.process(np.zeros((0, 2), dtype=np.float32)).shape == (0, 2)


def test_reset_restores_the_initial_state() -> None:
    r = PolyphaseResampler(48000, 16000)
    x = np.random.default_rng(5).standard_normal(7680).astype(np.float32)
    first = r.process(x)
    r.reset()
    assert np.array_equal(r.process(x), first)


def test_make_resampler_is_none_when_rates_match() -> None:
    assert make_resampler(48000, 48000) is None
    assert make_resampler(48000, 16000) is not None
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_resample.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.resample'`

- [ ] **Step 3: 実装を書く**

`vspeech/lib/resample.py` を新規作成:

```python
"""Rational-ratio polyphase FIR resampler for the device boundaries (ADR-0070).

numpy only -- no torch, no scipy, no sounddevice. This module is reachable from the
streaming consumer role, which must not pull torch in (ADR-0055), and numpy is
guaranteed present because the base dependency ctranslate2 requires it.

The filter is a Kaiser-windowed sinc whose length is *derived* from the requested
transition width and stopband attenuation rather than fixed at some tap count. A
fixed tap count silently produces a useless filter as the ratio changes: at 16
taps per phase the 48k->16k transition band spans 6.6-9.4 kHz, leaving 8.4 kHz
only 14 dB down (measured).
"""

from math import ceil
from math import gcd
from math import pi

import numpy as np
from numpy.typing import NDArray


def _kaiser_beta(stopband_db: float) -> float:
    """Kaiser's empirical beta for a target stopband attenuation."""
    if stopband_db > 50.0:
        return 0.1102 * (stopband_db - 8.7)
    if stopband_db >= 21.0:
        excess = stopband_db - 21.0
        return 0.5842 * excess**0.4 + 0.07886 * excess
    return 0.0


class PolyphaseResampler:
    """Stateful L/M polyphase resampler over float32.

    `process` is the streaming entry point: it keeps the filter tail across calls, so
    feeding a signal in blocks gives the same result as feeding it whole. Use it for
    a continuous stream (mic capture, continuous playback). `resample_full` is the
    one-shot entry point for a self-contained buffer (one utterance): it flushes the
    tail and removes the group delay, then resets.
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        transition_width: float = 0.10,
        stopband_db: float = 80.0,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"rates must be positive: {src_rate} -> {dst_rate}")
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        divisor = gcd(self.src_rate, self.dst_rate)
        self.up = self.dst_rate // divisor
        self.down = self.src_rate // divisor
        # The prototype runs at the interpolated rate up*src. Its stopband must start
        # at the narrower of the two Nyquist limits: below that is signal we keep,
        # above it is what would alias (downsampling) or show up as an image
        # (upsampling).
        interpolated = self.up * self.src_rate
        nyquist = min(self.src_rate, self.dst_rate) / 2.0
        passband = nyquist * (1.0 - transition_width)
        cutoff = 0.5 * (passband + nyquist)
        width = 2 * pi * (nyquist - passband) / interpolated
        n_taps = int(ceil((stopband_db - 8.0) / (2.285 * width))) + 1
        # Round the half length up to a multiple of `down` so the group delay is a
        # whole number of OUTPUT samples. A fractional delay cannot be trimmed exactly
        # in resample_full and shows up as a phase error (measured -25 dB against the
        # reference, versus -74 dB once the delay is integral).
        self._half_len = ceil(((n_taps - 1) // 2) / self.down) * self.down
        self.delay_samples = self._half_len // self.down
        n_taps = 2 * self._half_len + 1
        index = np.arange(-self._half_len, self._half_len + 1, dtype=np.float64)
        normalised_cutoff = 2.0 * cutoff / interpolated
        taps = (
            normalised_cutoff
            * np.sinc(normalised_cutoff * index)
            * np.kaiser(n_taps, _kaiser_beta(stopband_db))
        )
        # Unity gain at DC after the up-fold zero stuffing.
        taps *= self.up / taps.sum()
        self.taps_per_phase = ceil(n_taps / self.up)
        padded = np.concatenate(
            [taps, np.zeros(self.taps_per_phase * self.up - n_taps)]
        )
        # phase p holds taps[p::up], reversed so each output is a forward dot product
        # against a forward window of the input.
        self._phases = np.ascontiguousarray(
            padded.reshape(self.taps_per_phase, self.up).T[:, ::-1].astype(np.float32)
        )
        self._tail: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._fed = 0
        self._emitted = 0
        self.reset()

    def reset(self) -> None:
        """Drop the filter state. Call this whenever the stream is discontinuous
        (device reopen, pause/resume, a new sender session)."""
        self._tail = np.zeros(self.taps_per_phase - 1, dtype=np.float32)
        self._fed = 0
        self._emitted = 0

    def out_len(self, n_in: int) -> int:
        """How many output samples `process` will return for `n_in` more inputs."""
        total = self._fed + n_in
        return -((-self.up * total) // self.down) - self._emitted

    def process(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample a block, carrying the filter state across calls.

        Accepts `(n,)` or `(n, channels)`. Output sample k is the dot product of
        phase `k*down % up` with the input window ending at `k*down // up`, so the
        output is delayed by `delay_samples` and no samples are held back.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[0] == 0:
            return np.zeros_like(x, shape=(0, *x.shape[1:]))
        if self._tail.ndim != x.ndim:
            # First block decides the channel layout.
            self._tail = np.zeros(
                (self.taps_per_phase - 1, *x.shape[1:]), dtype=np.float32
            )
        full = np.concatenate([self._tail, x])
        total = self._fed + x.shape[0]
        end = -((-self.up * total) // self.down)
        n_out = end - self._emitted
        out = np.zeros((n_out, *x.shape[1:]), dtype=np.float32)
        if n_out:
            window = np.lib.stride_tricks.sliding_window_view(
                full, self.taps_per_phase, axis=0
            )
            if self.up == 1:
                # Integer decimation: one phase, and the window start advances by a
                # constant `down`, so this is a single strided matvec.
                start = self._emitted * self.down - self._fed
                out[:] = window[start :: self.down][:n_out] @ self._phases[0]
            else:
                # Within one phase the window start also advances by exactly `down`
                # (k -> k+up maps to m -> m+down), so each phase is a strided view and
                # no gather copy is needed.
                for offset in range(min(self.up, n_out)):
                    k = self._emitted + offset
                    phase = (k * self.down) % self.up
                    start = (k * self.down) // self.up - self._fed
                    count = (n_out - offset + self.up - 1) // self.up
                    rows = window[start :: self.down][:count]
                    out[offset :: self.up][: len(rows)] = rows @ self._phases[phase]
        self._emitted = end
        self._fed = total
        keep = self.taps_per_phase - 1
        self._tail = full[-keep:] if keep else full[:0]
        return out

    def resample_full(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample one self-contained buffer: flush the tail, remove the group delay.

        Streaming `process` would leave the last `delay_samples` worth of audio inside
        the filter, so an utterance played through it would lose its tail.
        """
        self.reset()
        flush = np.zeros(
            (self._half_len // self.up + self.down, *x.shape[1:]), dtype=np.float32
        )
        out = np.concatenate([self.process(x), self.process(flush)])
        self.reset()
        want = -((-self.up * x.shape[0]) // self.down)
        return out[self.delay_samples : self.delay_samples + want]


def make_resampler(src_rate: int, dst_rate: int) -> PolyphaseResampler | None:
    """A resampler, or None when the rates already match.

    None means "pass the bytes through untouched" -- the callers rely on that to stay
    bit-identical to the pre-ADR-0070 behaviour when the device already runs at the
    pipeline's rate.
    """
    if src_rate == dst_rate:
        return None
    return PolyphaseResampler(src_rate, dst_rate)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_resample.py -q`
Expected: PASS（全件）

- [ ] **Step 5: lint / 型検査**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: いずれも終了コード 0。`echo $?` で確認する（パイプに通さない）。

- [ ] **Step 6: コミット**

```bash
git add vspeech/lib/resample.py tests/test_resample.py
git commit -m "feat(resample): デバイス境界用のポリフェーズリサンプラを追加 (ADR-0070)"
```

---

### Task 2: 共有 PCM デコード/エンコード (`vspeech/lib/pcm.py`)

**Files:**
- Create: `vspeech/lib/pcm.py`
- Modify: `vspeech/worker/transcription.py:64-95`（`_pcm_to_float32_mono` を `decode_pcm` に置き換える）
- Test: `tests/test_pcm_codec.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `decode_pcm(data: bytes, format: SampleFormat, channels: int) -> NDArray[np.float32]` — `channels == 1` なら `(n,)`、それ以外は `(n, channels)`
  - `encode_pcm(x: NDArray[np.float32], format: SampleFormat) -> bytes` — `[-1, 1]` へ**飽和クリップ**してから量子化

`transcription.py` の既存デコードは private で mono 固定なので、チャンネルを保つ形へ昇格させて 3 箇所（transcription / recording / playback）で共有する。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_pcm_codec.py` を新規作成:

```python
"""Shared PCM decode/encode used at every device boundary (ADR-0070)."""

import numpy as np
import pytest

from vspeech.config import SampleFormat
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm

FORMATS = [
    SampleFormat.UINT8,
    SampleFormat.INT8,
    SampleFormat.INT16,
    SampleFormat.INT24,
    SampleFormat.FLOAT32,
]


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_preserves_the_signal(fmt: SampleFormat) -> None:
    x = (np.sin(np.linspace(0, 20, 500)) * 0.8).astype(np.float32)
    got = decode_pcm(encode_pcm(x, fmt), fmt, channels=1)
    # 8-bit formats quantise coarsely; everything else is far finer.
    tol = 1 / 100.0 if fmt in (SampleFormat.UINT8, SampleFormat.INT8) else 1 / 30000.0
    assert got.shape == x.shape
    assert np.max(np.abs(got - x)) < tol


def test_uint8_silence_is_the_128_bias_not_full_scale_dc() -> None:
    """unsigned 8-bit PCM centres on 128. Decoding it as signed turns silence into
    full-scale DC."""
    silence = bytes([128] * 64)
    assert np.max(np.abs(decode_pcm(silence, SampleFormat.UINT8, channels=1))) == 0.0


def test_int24_is_sign_extended() -> None:
    # -1 in 3-byte little-endian two's complement.
    data = b"\xff\xff\xff" * 8
    got = decode_pcm(data, SampleFormat.INT24, channels=1)
    assert np.all(got < 0.0)
    assert np.allclose(got, -1.0 / (1 << 23), atol=1e-9)


def test_multichannel_is_deinterleaved_not_downmixed() -> None:
    interleaved = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    encoded = encode_pcm(interleaved, SampleFormat.FLOAT32)
    got = decode_pcm(encoded, SampleFormat.FLOAT32, channels=2)
    assert got.shape == (2, 2)
    assert np.allclose(got[:, 0], [0.5, 0.25])
    assert np.allclose(got[:, 1], [-0.5, -0.25])


@pytest.mark.parametrize("fmt", FORMATS)
def test_encode_saturates_instead_of_wrapping(fmt: SampleFormat) -> None:
    """Resampling overshoots past the original peak (Gibbs). A wrapping cast turns
    that overshoot into a full-scale sign flip, i.e. a click."""
    over = np.array([1.9, -1.9, 3.0, -3.0], dtype=np.float32)
    got = decode_pcm(encode_pcm(over, fmt), fmt, channels=1)
    assert np.all(got[[0, 2]] > 0.9), f"{fmt}: positive overshoot wrapped"
    assert np.all(got[[1, 3]] < -0.9), f"{fmt}: negative overshoot wrapped"


def test_unsupported_format_raises() -> None:
    with pytest.raises(ValueError):
        decode_pcm(b"\x00\x00", SampleFormat.INVALID, channels=1)
    with pytest.raises(ValueError):
        encode_pcm(np.zeros(2, dtype=np.float32), SampleFormat.INVALID)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_pcm_codec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vspeech.lib.pcm'`

- [ ] **Step 3: 実装を書く**

`vspeech/lib/pcm.py` を新規作成:

```python
"""Format-aware PCM decode/encode shared by every device boundary (ADR-0070).

Dispatch is keyed on SampleFormat, NOT byte width: UINT8 and INT8 share a width but
differ in sign and bias, so a width-keyed table would decode unsigned-8 as signed and
skip its 128 offset (silence -> full-scale DC).

This started as transcription.py's private `_pcm_to_float32_mono`. It is shared now
because recording and playback need the same decode, and they must keep the channel
layout rather than downmix.
"""

import numpy as np
from numpy.typing import NDArray

from vspeech.config import SampleFormat

_INT24_SIGN_BIT = 0x800000
_INT24_SCALE = float(1 << 23)


def decode_pcm(
    data: bytes, format: SampleFormat, channels: int
) -> NDArray[np.float32]:
    """Decode interleaved PCM bytes into float32 in [-1, 1].

    Returns `(frames,)` for mono and `(frames, channels)` otherwise -- the shape
    PolyphaseResampler.process expects.
    """
    if format == SampleFormat.FLOAT32:
        samples = np.frombuffer(data, dtype=np.float32).astype(np.float32)
    elif format == SampleFormat.UINT8:
        # unsigned 8-bit PCM is biased by 128 (128 == silence).
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif format == SampleFormat.INT8:
        samples = np.frombuffer(data, dtype=np.int8).astype(np.float32) / 128.0
    elif format == SampleFormat.INT16:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif format == SampleFormat.INT24:
        # 3-byte little-endian signed PCM -> sign-extended int32 -> [-1, 1).
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        as32 = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        as32 = (as32 ^ _INT24_SIGN_BIT) - _INT24_SIGN_BIT
        samples = as32.astype(np.float32) / _INT24_SCALE
    else:
        raise ValueError(f"unsupported PCM format: {format!r}")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return np.ascontiguousarray(samples, dtype=np.float32)


def encode_pcm(x: NDArray[np.float32], format: SampleFormat) -> bytes:
    """Encode float32 (mono `(n,)` or interleavable `(n, channels)`) back to PCM.

    Always saturates at full scale. Resampling overshoots past the original peak
    (Gibbs), and a wrapping cast turns that overshoot into a sign flip -- an audible
    click. Never replace this with a bare `.astype(np.int16)`.
    """
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if format == SampleFormat.FLOAT32:
        # float32 output still gets clipped: PortAudio would clip it anyway, and
        # leaving it unbounded makes the boundary behave differently per format.
        return np.clip(flat, -1.0, 1.0).astype(np.float32).tobytes()
    clipped = np.clip(flat, -1.0, 1.0)
    if format == SampleFormat.UINT8:
        return (np.rint(clipped * 127.0) + 128.0).astype(np.uint8).tobytes()
    if format == SampleFormat.INT8:
        return np.rint(clipped * 127.0).astype(np.int8).tobytes()
    if format == SampleFormat.INT16:
        return np.rint(clipped * 32767.0).astype(np.int16).tobytes()
    if format == SampleFormat.INT24:
        as32 = np.rint(clipped * (_INT24_SCALE - 1.0)).astype(np.int32)
        packed = np.empty((as32.size, 3), dtype=np.uint8)
        packed[:, 0] = as32 & 0xFF
        packed[:, 1] = (as32 >> 8) & 0xFF
        packed[:, 2] = (as32 >> 16) & 0xFF
        return packed.tobytes()
    raise ValueError(f"unsupported PCM format: {format!r}")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_pcm_codec.py -q`
Expected: PASS

- [ ] **Step 5: `transcription.py` を共有デコーダに寄せる**

`vspeech/worker/transcription.py` の `_pcm_to_float32_mono`（64-95 行）を削除し、`pcm_to_waveform`（129-138 行）を次に差し替える:

```python
def pcm_to_waveform(sound: SoundInput) -> np.ndarray:
    """Decode PCM to a mono float32 waveform at 16 kHz for faster-whisper.

    Decodes per sound.format, downmixes to mono, and resamples to 16 kHz when
    sound.rate differs (see _resample_to_16k for why the model needs 16 kHz).
    """
    from vspeech.lib.pcm import decode_pcm

    samples = decode_pcm(sound.data, sound.format, sound.channels)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    if sound.rate != WHISPER_SAMPLE_RATE:
        samples = _resample_to_16k(samples, sound.rate)
    return samples
```

`_resample_to_16k` は **触らない**（whisper 入力は PyAV のまま = ADR-0036）。不要になった `SampleFormat` の import が残る場合は ruff が拾う。

- [ ] **Step 6: 既存の transcription テストが緑のままか確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_transcription_vad.py tests/test_pcm_codec.py -q`
（`tests/` に他の transcription 系テストがあれば併せて指定する。ファイル名は `ls tests/` で確認する。）
Expected: PASS

- [ ] **Step 7: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/lib/pcm.py tests/test_pcm_codec.py vspeech/worker/transcription.py
git commit -m "feat(pcm): format 対応の PCM デコード/飽和エンコードを共有モジュールへ昇格 (ADR-0070)"
```

---

### Task 3: デバイスレートの解決 (`vspeech/lib/audio.py`)

**Files:**
- Modify: `vspeech/lib/audio.py`（末尾に追加。既存の `_resolve_device` 群には触らない）
- Test: `tests/test_device_rate.py`

**Interfaces:**
- Consumes: `DeviceInfo`（`vspeech/lib/audio.py:19`）
- Produces:
  - `class DeviceRateUnresolvedError(DeviceNotFoundError)`（`vspeech/exceptions.py` に追加）
  - `resolve_device_rate(device: DeviceInfo, override: int | None, *, input: bool, config_key: str) -> tuple[int, str]` — `(rate, どう解決したかの説明)`

**実測の根拠（ADR-0071）:** PortAudio は MME/DirectSound/WDM-KS に `default_samplerate = 44100` を返す（48kHz で動いているエンドポイントも同じ）。MME は `check_input_settings` にどのレートを渡しても OK を返すのでプローブも効かない。WASAPI だけが真のミックスレートを返し、MME 名は WASAPI 名を 31 文字で切り詰めたものになっている。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_device_rate.py` を新規作成:

```python
"""Resolving the true device rate (ADR-0071).

sounddevice is stubbed: these are pure decisions over the device table, and the real
table differs per machine.
"""

import pytest

from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.audio import resolve_device_rate

WASAPI = 2
MME = 0

# (hostapi index -> name) and the device rows resolve_device_rate reads.
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
_DEVICES = [
    # MME truncates the name to 31 chars and lies about the rate.
    {"index": 0, "name": "Speakers (Realtek(R) Audio)", "hostapi": MME,
     "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
    {"index": 1, "name": "Microphone Array (Realtek(R) Au", "hostapi": MME,
     "max_input_channels": 4, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"index": 2, "name": "Microsoft サウンド マッパー - Input", "hostapi": MME,
     "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"index": 3, "name": "Ambiguous Device", "hostapi": MME,
     "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": WASAPI,
     "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
    {"index": 11, "name": "Microphone Array (Realtek(R) Audio)", "hostapi": WASAPI,
     "max_input_channels": 4, "max_output_channels": 0, "default_samplerate": 48000.0},
    {"index": 12, "name": "Ambiguous Device A", "hostapi": WASAPI,
     "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
    {"index": 13, "name": "Ambiguous Device B", "hostapi": WASAPI,
     "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
]


@pytest.fixture(autouse=True)
def _stub_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    import vspeech.lib.audio as audio

    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _HOSTAPIS)
    monkeypatch.setattr(audio.sd, "query_devices", lambda: _DEVICES)


def _device(index: int) -> DeviceInfo:
    for raw in _DEVICES:
        if raw["index"] == index:
            return DeviceInfo.model_validate(raw)
    raise AssertionError(index)


def test_explicit_override_wins() -> None:
    rate, how = resolve_device_rate(
        _device(0), 96000, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 96000
    assert "playback.output_device_rate" in how


def test_wasapi_device_uses_its_own_default_samplerate() -> None:
    rate, how = resolve_device_rate(
        _device(10), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    assert "WASAPI" in how


def test_mme_device_takes_the_rate_from_its_wasapi_counterpart() -> None:
    """PortAudio reports 44100 for this MME device; the endpoint really runs at 48000."""
    rate, _ = resolve_device_rate(
        _device(0), None, input=False, config_key="playback.output_device_rate"
    )
    assert rate == 48000
    rate, _ = resolve_device_rate(
        _device(1), None, input=True, config_key="recording.input_device_rate"
    )
    assert rate == 48000


def test_counterpart_match_respects_direction() -> None:
    """An output-only WASAPI row must not answer for an input device."""
    with pytest.raises(DeviceRateUnresolvedError):
        resolve_device_rate(
            _device(0), None, input=True, config_key="recording.input_device_rate"
        )


def test_pseudo_device_without_a_counterpart_fails_loud() -> None:
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(2), None, input=True, config_key="recording.input_device_rate"
        )
    assert "recording.input_device_rate" in str(excinfo.value)


def test_conflicting_counterparts_fail_loud_rather_than_guess() -> None:
    """Two WASAPI rows match the prefix and disagree: never pick one silently."""
    with pytest.raises(DeviceRateUnresolvedError) as excinfo:
        resolve_device_rate(
            _device(3), None, input=True, config_key="stream_vc.input_device_rate"
        )
    assert "stream_vc.input_device_rate" in str(excinfo.value)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_device_rate.py -q`
Expected: FAIL — `ImportError: cannot import name 'DeviceRateUnresolvedError'`

- [ ] **Step 3: 例外を足す**

`vspeech/exceptions.py` の `DeviceNotFoundError` の定義直後に追加:

```python
class DeviceRateUnresolvedError(DeviceNotFoundError):
    """The device's true sample rate could not be determined (ADR-0071).

    A subclass of DeviceNotFoundError so the existing preflight handlers keep
    catching it, while callers that care can tell the two apart.
    """
```

- [ ] **Step 4: `resolve_device_rate` を実装**

`vspeech/lib/audio.py` の末尾（`get_sd_dtype` の前）に追加し、先頭の import に `from vspeech.exceptions import DeviceRateUnresolvedError` を足す:

```python
_WASAPI_HOST_API = "Windows WASAPI"


def _wasapi_counterpart_rates(name: str, *, input: bool) -> set[int]:
    """Mix rates of the WASAPI devices whose name starts with `name`.

    PortAudio's WMME/DirectSound backends report a hardcoded 44100 for every device,
    so their `default_samplerate` cannot be trusted. Their device names, however, are
    the WASAPI names truncated to 31 characters, which makes the WASAPI row for the
    same endpoint findable by prefix (ADR-0071).
    """
    host_apis = sd.query_hostapis()
    rates: set[int] = set()
    for raw in sd.query_devices():
        device = DeviceInfo.model_validate(dict(raw))
        if host_apis[device.host_api]["name"] != _WASAPI_HOST_API:
            continue
        if input and device.max_input_channels <= 0:
            continue
        if not input and device.max_output_channels <= 0:
            continue
        if device.name.startswith(name):
            rates.add(int(round(float(dict(raw)["default_samplerate"]))))
    return rates


def resolve_device_rate(
    device: DeviceInfo, override: int | None, *, input: bool, config_key: str
) -> tuple[int, str]:
    """The rate to open `device` at, plus a human-readable note on how it was decided.

    Order: explicit config -> the device's own default_samplerate when it is a WASAPI
    device -> the mix rate of its WASAPI counterpart (ADR-0071). Anything ambiguous
    raises rather than guessing: opening at the wrong rate silently reinstates the OS
    resampler that ADR-0070 exists to remove.
    """
    if override is not None:
        return override, f"{config_key} で明示"
    host_apis = sd.query_hostapis()
    host_api_name = host_apis[device.host_api]["name"]
    if host_api_name == _WASAPI_HOST_API:
        raw = dict(sd.query_devices(device.index))
        return int(round(float(raw["default_samplerate"]))), "WASAPI のミックス形式"
    rates = _wasapi_counterpart_rates(device.name, input=input)
    if len(rates) == 1:
        return rates.pop(), f"WASAPI の同名デバイス ({host_api_name} 経由)"
    kind = "入力" if input else "出力"
    if not rates:
        detail = "対応する WASAPI デバイスが見つかりません"
    else:
        detail = f"対応する WASAPI デバイスのレートが一致しません ({sorted(rates)})"
    raise DeviceRateUnresolvedError(
        f"{kind}デバイス '{device.name}' ({host_api_name}) の実レートを判定できません: "
        f"{detail}。Windows のサウンド設定で「既定の形式」を確認し "
        f"{config_key} に明示してください"
    )
```

- [ ] **Step 5: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_device_rate.py -q`
Expected: PASS

- [ ] **Step 6: 実機で自動解決が効くことを確認（この機体でのみ意味がある確認）**

Run:
```bash
PYTHONIOENCODING=utf-8 uv run python -c "
import sounddevice as sd
from vspeech.lib.audio import DeviceInfo, resolve_device_rate
from vspeech.exceptions import DeviceRateUnresolvedError
ok=bad=0
for raw in sd.query_devices():
    d=DeviceInfo.model_validate(dict(raw))
    for inp in (True,False):
        if (d.max_input_channels if inp else d.max_output_channels)<=0: continue
        try:
            r,how=resolve_device_rate(d,None,input=inp,config_key='x'); ok+=1
        except DeviceRateUnresolvedError: bad+=1
print('resolved',ok,'unresolved',bad)"
```
Expected: `unresolved` が 4 以下（疑似デバイス「サウンド マッパー」「プライマリ サウンド ドライバー」の入出力ぶんのみ）。

- [ ] **Step 7: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/lib/audio.py vspeech/exceptions.py tests/test_device_rate.py
git commit -m "feat(audio): デバイスの実サンプルレートを WASAPI 同名デバイスから解決する (ADR-0071)"
```

---

### Task 4: 設定フィールド 4 つ

**Files:**
- Modify: `vspeech/config.py`（`RecordingConfig` 142-168 / `PlaybackConfig` 191-205 / `StreamVcConfig` 542-547）
- Modify: `config.toml.example`（`[recording]` 13 行目〜 / `[playback]` 70 行目〜 / `[stream_vc]` 196 行目〜）
- Test: `tests/test_config_device_rate.py`

**Interfaces:**
- Produces: `RecordingConfig.input_device_rate`, `PlaybackConfig.output_device_rate`, `StreamVcConfig.input_device_rate`, `StreamVcConfig.output_device_rate` — すべて `int | None = None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_config_device_rate.py` を新規作成:

```python
"""The four device-rate overrides (ADR-0071)."""

import pytest
from pydantic import ValidationError

from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import StreamVcConfig


def test_defaults_are_none_so_existing_configs_still_load() -> None:
    assert RecordingConfig().input_device_rate is None
    assert PlaybackConfig().output_device_rate is None
    assert StreamVcConfig().input_device_rate is None
    assert StreamVcConfig().output_device_rate is None


def test_explicit_rates_are_accepted() -> None:
    assert RecordingConfig(input_device_rate=48000).input_device_rate == 48000
    assert PlaybackConfig(output_device_rate=44100).output_device_rate == 44100
    sv = StreamVcConfig(input_device_rate=48000, output_device_rate=48000)
    assert (sv.input_device_rate, sv.output_device_rate) == (48000, 48000)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_rates_are_rejected(bad: int) -> None:
    with pytest.raises(ValidationError):
        RecordingConfig(input_device_rate=bad)
    with pytest.raises(ValidationError):
        PlaybackConfig(output_device_rate=bad)
    with pytest.raises(ValidationError):
        StreamVcConfig(input_device_rate=bad)
    with pytest.raises(ValidationError):
        StreamVcConfig(output_device_rate=bad)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_config_device_rate.py -q`
Expected: FAIL — `AttributeError: 'RecordingConfig' object has no attribute 'input_device_rate'`

- [ ] **Step 3: フィールドを足す**

`RecordingConfig`（`input_device_index` の直後、168 行目付近）:

```python
    input_device_rate: int | None = Field(
        default=None,
        gt=0,
        description="録音デバイスを開くレート。未指定なら自動判定 (WASAPI の"
        "ミックス形式から逆引き)。ここと rate が違うときだけプロセス内で"
        "リサンプルする。MME/DirectSound は PortAudio が 44100 固定を返すので、"
        "自動判定できない場合のみ Windows のサウンド設定の「既定の形式」を指定する",
    )
```

`PlaybackConfig`（`output_device_index` の直後、205 行目付近）:

```python
    output_device_rate: int | None = Field(
        default=None,
        gt=0,
        description="再生デバイスを開くレート。未指定なら自動判定。固定レートで"
        "開くので、音声ソースのレートが変わってもデバイスは開き直さない",
    )
```

`StreamVcConfig`（`input_device_index` / `output_device_index` の直後、544/547 行目付近）:

```python
    input_device_rate: int | None = Field(
        default=None,
        gt=0,
        description="ストリーミング入力デバイスを開くレート。未指定なら自動判定",
    )
    output_device_rate: int | None = Field(
        default=None,
        gt=0,
        description="ストリーミング出力デバイスを開くレート。未指定なら自動判定",
    )
```

- [ ] **Step 4: `config.toml.example` に追記**

`[recording]` の `input_device_name = ""`（39 行目）の直後:

```toml
# デバイスを開くサンプルレート。未指定 (コメントアウト) なら自動判定する。
# 自動判定に失敗すると起動時にエラーになるので、そのときだけ Windows の
# サウンド設定 → デバイスのプロパティ → 詳細 →「既定の形式」の値を書く。
# input_device_rate = 48000
```

`[playback]` の `output_device_name = ""`（74 行目）の直後:

```toml
# 再生デバイスを開くサンプルレート。未指定なら自動判定。
# output_device_rate = 48000
```

`[stream_vc]` の `input_device_name = ""`（277 行目）と `output_device_name = ""`（282 行目）のそれぞれ直後:

```toml
# input_device_rate = 48000
```
```toml
# output_device_rate = 48000
```

- [ ] **Step 5: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_config_device_rate.py tests/test_config_input.py tests/test_config_stream_vc.py -q`
Expected: PASS

- [ ] **Step 6: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/config.py config.toml.example tests/test_config_device_rate.py
git commit -m "feat(config): デバイスを開くレートの上書き設定を 4 つ追加 (ADR-0071)"
```

---

### Task 5: stream_vc の入口 (`vspeech/stream_vc/capture.py`)

**目的:** ストリーミング VC の入力デバイスをネイティブレートで開き、16kHz への変換をプロセス内で行う。

**範囲:** `vspeech/stream_vc/capture.py`。テストは `tests/`。

**契約**

- Consumes:
  - `vspeech.lib.resample.make_resampler(src_rate: int, dst_rate: int) -> PolyphaseResampler | None`（`src == dst` なら `None` = 素通し）
  - `PolyphaseResampler.process(x) -> NDArray[np.float32]`（状態保持。`(n,)` と `(n, channels)`）、`.reset() -> None`、`.out_len(n_in) -> int`、`.delay_samples: int`
  - `vspeech.lib.audio.resolve_device_rate(device, override, *, input: bool, config_key: str) -> tuple[int, str]`。解決できないときは `vspeech.exceptions.DeviceRateUnresolvedError`
  - `StreamVcConfig.input_device_rate: int | None`
  - 既存: `resolve_stream_vc_input_device`, `run_with_device_retry`, `drop_oldest_put`, `CaptureSignal`, `CAPTURE_RATE`
- Produces:
  - `capture_loop(config, out_queue, hop, ready, running)` のシグネチャは**不変**（`subsystem.py` が呼ぶ）。
  - `out_queue` に載る要素の型と意味は**不変**（長さ `hop` の `NDArray[np.float32]`、または `CaptureSignal.REOPEN`）。`vspeech/stream_vc/runner.py` は無改造で動くこと。

**この task 固有の制約**

- `run_with_device_retry` は `[T: _Closable]` に束縛され、`open_stream` の戻り値をそのまま `close_quietly()` に渡す。したがって `open_stream` はストリーム自体を返さなければならず、タプルは返せない。`vspeech/stream_vc/retry.py` は変更しない。
- **事前充填を入れないこと。** 因果ポリフェーズは出力本数が欠けないので、1 device tick あたり 1 ブロックがそのまま出る（Task 1 の受入基準として検証済み）。滞留を抱える実装向けの対策をここに持ち込むと、逆に丸ごと 1 hop ぶんの遅延が乗る。

**受入基準**

- [ ] 入力デバイスは `resolve_device_rate` が返したレートで開かれる。`stream_vc.input_device_rate` を明示した場合はその値で開かれる。
- [ ] 解決したレートが `CAPTURE_RATE` と等しいときはリサンプラを構築せず、変更前と出力がビット一致する。
- [ ] 解決したレートが `CAPTURE_RATE` と異なるとき、`out_queue` に載るブロックは常にちょうど `hop` サンプルである。
- [ ] デバイスの 1 回の読み取りにつき `out_queue` へ載るブロック数が定常的に 1 である。48000Hz と 44100Hz の両方で満たすこと。
- [ ] 比が割り切れず 1 回の読み取りが `hop` に満たない／超える場合でも、端数サンプルは失われず次へ繰り越される。
- [ ] デバイス再オープン後、再オープン前のフィルタ状態と作りかけのブロックが新しいストリームへ持ち越されない。
- [ ] 開いたレート、その決定根拠、変換の有無がログに残る。
- [ ] 一時停止中のドロップ、キュー満杯のドロップ、入力オーバーフローについて、テレメトリの記録とログ間引きの挙動が変更前と同じである。

**検証**

```
PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc.py tests/test_stream_vc_capture.py tests/test_stream_vc_config.py tests/test_stream_vc_consumer.py tests/test_stream_vc_entrypoint.py tests/test_stream_vc_envelope.py tests/test_stream_vc_gate.py -q
uv run ruff format . && uv run ruff check . && uv run ty check
```

既存テストが緑のままであること、いずれも終了コード 0 であること。上の受入基準を検証する新規テストを足すこと。

**コミット単位:** capture の変更と新規テストで 1 コミット。

---

### Task 6: stream_vc の出口 (`playback.py` / `consumer.py`)

**目的:** ストリーミング再生の出力デバイスを固定のデバイスレートで開き、パケットのレートからの変換をプロセス内で行う。

**範囲:** `vspeech/stream_vc/playback.py`, `vspeech/stream_vc/consumer.py`。テストは `tests/`。

**契約**

- Consumes:
  - `vspeech.lib.resample.make_resampler`, `PolyphaseResampler.process`, `.reset`
  - `vspeech.lib.pcm.decode_pcm(data, format, channels)`, `vspeech.lib.pcm.encode_pcm(x, format)`（`encode_pcm` は飽和クリップ済み）
  - `vspeech.lib.audio.resolve_device_rate`, `StreamVcConfig.output_device_rate`
- Produces: 後続 task はこの 2 ファイルに依存しない。ただし出力ストリームを開く関数は `playback.py` と `consumer.py` の両方から使われるので、両者で整合していること。

**この task 固有の制約**

- `consumer.py` から到達するコードは **torch を import してはならない**（ADR-0055）。`tests/test_forbidden_imports.py` を壊さないこと。
- パケットの PCM は int16 モノラル。jitter buffer の concealment が返す PCM のブロック長は `packet.sample_rate` ではなく `JitterBuffer` 内部の値に由来する。両者が整合しているかを実装時に確認し、**整合していなければ実装で辻褄を合わせず、その事実を報告すること。**

**受入基準**

- [ ] 出力デバイスは `resolve_device_rate` が返したレートで開かれ、`packet.sample_rate` では開かれない。
- [ ] `packet.sample_rate` がデバイスのレートと等しいとき、PCM は変換されずそのまま書き込まれる。
- [ ] 異なるとき、書き込まれる PCM はデバイスのレートに変換されている。
- [ ] 連続するパケットをまたいでフィルタ状態が保たれ、パケット境界に不連続が生じない。
- [ ] 送信側のセッションが変わっても出力ストリームは開き直されない。リサンプラの状態だけが破棄される。
- [ ] 送信側のモデルレートが変わったとき、リサンプラが作り直される。
- [ ] 変換で振幅が int16 の範囲を超えた場合、ラップアラウンドせず飽和する。
- [ ] デバイス障害からの遅延再オープンと、underflow / seq gap / stale drop のテレメトリとログ間引きの挙動が変更前と同じである。
- [ ] `role=consumer` の経路が torch を import せずに動く。

**検証**

```
PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc.py tests/test_stream_vc_consumer.py tests/test_stream_vc_capture.py tests/test_forbidden_imports.py -q
uv run ruff format . && uv run ruff check . && uv run ty check
```

既存テストが緑のままであること、いずれも終了コード 0 であること。上の受入基準を検証する新規テストを足すこと。

**コミット単位:** playback.py と consumer.py の変更と新規テストで 1 コミット。

---

### Task 7: 発話系の録音 (`vspeech/worker/recording.py`)

**目的:** 録音デバイスをネイティブレートで開き、`recording.rate` への変換をプロセス内で行う。

**範囲:** `vspeech/worker/recording.py`。テストは `tests/`。

**契約**

- Consumes: `make_resampler`, `PolyphaseResampler.process`, `decode_pcm`/`encode_pcm`, `resolve_device_rate`, `RecordingConfig.input_device_rate`
- Produces: **下流の契約は不変。** この worker が出す `SoundOutput` の `rate` は今までどおり `config.rate`、`format` は `config.format`、`channels` は `config.channels`。`worker/transcription.py` と `worker/vc.py` は無改造で動くこと。

**この task 固有の制約**

- 既存コードは 1 回の読み取りで得たフレーム数を、実測ではなく設定値の定数で数えている。デバイスレートで読むようになるとその定数は実際のフレーム数と一致しなくなる。これは推測ではなく現在のコードの事実であり、見落とすと下の時間換算の受入基準が壊れる。

**受入基準**

- [ ] 録音デバイスは `resolve_device_rate` が返したレートで開かれる。
- [ ] 下流が受け取る `SoundOutput` の `rate` / `format` / `channels` が変更前と同じである。
- [ ] デバイスのレートと `recording.rate` が等しいときは変換されず、変更前と出力がビット一致する。
- [ ] **無音判定・`interval_sec`・`max_recording_sec` の時間換算が実時間の秒数と一致する。** 48000Hz のデバイスを `rate = 16000` で使う設定で、1 秒ぶんの音声が 1 秒として扱われること。
- [ ] `channels > 1` の設定でチャンネル数が保たれる（モノラルに畳まれない）。
- [ ] 開いたレートと決定根拠がログに残る。
- [ ] 入力オーバーフローのログと、デバイス障害時の再試行の挙動が変更前と同じである。

**検証**

```
PYTHONIOENCODING=utf-8 uv run pytest tests/test_recording_metrics.py tests/test_recording_trace.py -q
uv run ruff format . && uv run ruff check . && uv run ty check
```

既存テストが緑のままであること、いずれも終了コード 0 であること。上の受入基準を検証する新規テストを足すこと。

**コミット単位:** recording.py の変更と新規テストで 1 コミット。

---

### Task 8: 発話系の再生 (`vspeech/worker/playback.py`)

**目的:** 再生デバイスを固定のデバイスレートで開き、音声ソースのレートからの変換をプロセス内で行う。

**範囲:** `vspeech/worker/playback.py`。テストは `tests/`。

**契約**

- Consumes: `make_resampler`, `PolyphaseResampler.resample_full`（ワンショット。末尾までフラッシュし群遅延を除去して、呼び出し後は状態を残さない）, `decode_pcm`/`encode_pcm`, `resolve_device_rate`, `PlaybackConfig.output_device_rate`
- Produces: 後続 task はこのファイルに依存しない。

**この task 固有の制約**

- 発話は 1 件ずつ独立した buffer であって連続ストリームではない。`PolyphaseResampler` はストリーミング用とワンショット用で入口が分かれており、ストリーミング用を使うと毎回末尾が欠ける。
- この worker は TTS(VOICEROID2 / VOICEVOX)・VC・録音の各ソースから、異なるレート・フォーマット・チャンネル数の音声を受け取る。

**受入基準**

- [ ] 再生デバイスは `resolve_device_rate` が返したレートで開かれ、音声ソースの `rate` では開かれない。
- [ ] **音声ソースのサンプルレートが変わってもデバイスは開き直されない。**（24000Hz の TTS と 40000Hz の VC を交互に再生してもリオープンが起きないこと。）
- [ ] 1 発話ぶんの音声を変換したとき、出力の長さが公称の長さと一致し、末尾が欠けない。
- [ ] 発話をまたいでリサンプラの状態が持ち越されない。同じ入力を続けて 2 回渡すと同じ出力が返ること。
- [ ] ソースのレートとデバイスのレートが等しいときは変換されない。
- [ ] 変換で振幅が範囲を超えた場合、ラップアラウンドせず飽和する。
- [ ] 音量調整、e2e テレメトリ、デバイス障害時の挙動が変更前と同じである。

**検証**

```
PYTHONIOENCODING=utf-8 uv run pytest tests/test_playback_e2e.py -q
uv run ruff format . && uv run ruff check . && uv run ty check
```

既存テストが緑のままであること、いずれも終了コード 0 であること。上の受入基準を検証する新規テストを足すこと。

**コミット単位:** playback.py の変更と新規テストで 1 コミット。

---

### Task 9: preflight でレート解決とデバイス開通を検証

**目的:** 解決したデバイスレートと、そのレートでデバイスを実際に開けるかどうかを、起動時に fail-loud で検証する。

**範囲:** `vspeech/preflight.py`。テストは `tests/`。

**契約**

- Consumes: `resolve_device_rate`, `DeviceRateUnresolvedError`, 4 つの設定フィールド（`recording.input_device_rate` / `playback.output_device_rate` / `stream_vc.input_device_rate` / `stream_vc.output_device_rate`）, 既存の `ConfigProblem` と `Checker`
- Produces: 後続 task はこのファイルに依存しない。

**この task 固有の制約**

- `DeviceRateUnresolvedError` は `DeviceNotFoundError` の派生なので、既存のハンドラが先に捕まえてしまう位置関係に注意すること。
- 出力側のレートが静的になったのは Task 6/8 の結果であり、それ以前は実行時にしか決まらなかった。出力デバイスをここで検証できるようになったのは今回が初めてである。

**受入基準**

- [ ] recording / playback / stream_vc の入力 / stream_vc の出力について、そのワーカーが有効なときにレート解決が検証される。
- [ ] レートを解決できない場合に `ConfigProblem` が上がり、その `field` が操作者の指定すべき設定キーを指す。
- [ ] 解決したレートでデバイスを開けない場合に `ConfigProblem` が上がり、メッセージにそのレートが含まれる。
- [ ] `role=producer` では入力側のみ、`role=consumer` では出力側のみが検証される（既存の役割分岐に従う）。
- [ ] 無効化されているワーカーについては何も検証されない。
- [ ] 既存の preflight 検査の結果が変わらない。

**検証**

```
PYTHONIOENCODING=utf-8 uv run pytest tests/test_preflight.py tests/test_device_rate.py tests/test_device_resolver.py -q
uv run ruff format . && uv run ruff check . && uv run ty check
```

既存テストが緑のままであること、いずれも終了コード 0 であること。上の受入基準を検証する新規テストを足すこと。

**コミット単位:** preflight.py の変更と新規テストで 1 コミット。

---

### Task 10: 全体検証・ADR 昇格・実機確認

**Files:**
- Modify: `docs/adr/0070-device-boundary-inhouse-polyphase-resampler.md`（Status と事前充填の記述）
- Modify: `docs/adr/0071-device-native-rate-resolution.md`（Status）
- Modify: `docs/adr/README.md`（索引の Status 列）

- [ ] **Step 1: スイート全体とヘルスゲート**

```bash
PYTHONIOENCODING=utf-8 uv run pytest
echo "pytest=$?"
uv run poe check
echo "check=$?"
```
Expected: pytest は Task 0 のベースライン 714 passed に新規ぶんが上乗せされ、失敗 0。`poe check` は既知の受容済み 2 件（audit の torch / vulture の vr2_config）以外に新規指摘なし。**終了コードで判定すること**（`| tail` の `$?` は `tail` のもの）。

- [ ] **Step 2: エントリポイントを実際に起動する**

テストだけでは掴めない不具合（3.14 の `get_event_loop`、cp1252 ロガー）が過去 2 回この段で出ている。必ず走らせる。

```bash
PYTHONIOENCODING=utf-8 uv run python -m vspeech --config <実機の config.toml>
```
Expected: preflight を通過し、解決したデバイスレートと変換の遅延がログに出る。例:
`use input device 17: Microphone Array (Realtek(R) Au @48000Hz (WASAPI の同名デバイス (MME 経由)) -> 16000Hz 変換`

- [ ] **Step 3: ADR-0070 の事前充填の記述を実測に合わせて訂正**

ADR-0070 は実装前に書いたため、Decision に「入口では群遅延ぶんの無音で事前充填する」と書いてある。**実測（Task 1 の `test_fixed_hop_cadence_needs_no_priming`）で不要と判明**したので、Accepted へ昇格させる前に該当箇所を次に差し替える:

```markdown
- **固定ブロックへ再ブロック化する入口でも、事前充填は要らない。** 因果ポリフェーズは出力本数が `ceil(L*n/M)` で欠けないため、1 device tick あたり 1 ブロックがそのまま出る(実測で配信遅れ min=max=0)。事前充填が要るのは soxr のように滞留を内部に抱える実装で、そこでは滞留が丸ごと 1 hop の遅延に量子化される(実測 +160ms)。この差が自前実装を選んだ理由そのものなので、Alternatives rejected の soxr 項と合わせて読むこと。
```

Alternatives rejected の「入口で事前充填せず、滞留を許容する — 実測 +160ms」の項は、soxr を採った場合の話であることが分かるよう「**soxr を採ったうえで**入口で事前充填せず」に直す。

- [ ] **Step 4: ADR を Accepted へ昇格**

`docs/adr/0070-*.md` と `docs/adr/0071-*.md` の `- Status: Proposed` を `- Status: Accepted` に変え、`docs/adr/README.md` の索引 2 行の Status 列も `Accepted` にする。

- [ ] **Step 5: 実機の耳確認（ユーザーに依頼する）**

自動テストでは決着しない。次を確認してもらう:
1. stream_vc を有効にして 1〜2 分話し、変換音に**耳障りな高域の付帯音（折り返し）やブロック境界のプチプチが無い**こと。変更前と比べて悪化していないこと。
2. 発話系で TTS と VC を交互に鳴らし、**デバイスのリオープンによる頭切れ・無音ギャップが消えている**こと。
3. 遅延が体感で増えていないこと（設計上の増分は入口 3.1ms + 出口 1.3ms）。

耳確認が通るまで `finishing-a-development-branch` へ進まない。

- [ ] **Step 6: コミット**

```bash
git add docs/adr/
git commit -m "docs(adr): ADR-0070/0071 を Accepted へ昇格し、事前充填の記述を実測に合わせる"
```

---

## 自己レビュー結果

**Spec 受入基準 → タスクの対応:**

| 受入基準 | タスク |
|---|---|
| 4 デバイスがネイティブレートで開かれる | 5, 6, 7, 8 |
| レートを設定で明示指定でき、未指定なら自動 | 3, 4 |
| 自動で決められないとき起動時に失敗し、指定すべきキーを示す | 3, 9 |
| 解決したレートで開けないことが起動時に検出される | 9 |
| 実際に開いたレートと決定根拠がログに残る | 5, 6, 7, 8 |
| レート一致時は変換せずビット不変 | 1 (`make_resampler` → None), 5, 6, 7, 8 |
| 阻止域減衰・通過帯平坦性が自動テストで検証される | 1 |
| ブロック分割と一括変換が一致する | 1 |
| 追加遅延が 1 ブロック分まるごとにならない | 1 (`test_fixed_hop_cadence_needs_no_priming`), 5 |
| 範囲外はラップせず飽和する | 2 (`encode_pcm`), 6, 7, 8 |
| 1 発話の末尾が欠けない | 1 (`resample_full`), 8 |
| ソースのレート変化で再生デバイスが開き直されない | 8 |
| 録音の無音判定・区間長の時間換算が実時間と一致する | 7 |
| 下流が受け取るレートが変わらない | 7 (契約不変), 8 |
| 依存パッケージが追加されない | Global Constraints |
| consumer ロールが torch 無しで動く | 6 |

**未解決事項:** なし。

**Task 5-9 の書き直しについて:** 当初この 5 task にも実装コードとテストコードを書いていたが、それを撤去して契約・制約・受入基準に絞った。理由は実測である — Task 1-4 は旧形式で実行し、**4 task すべてで plan 内のコードに誤りが見つかった**（Task 1: `resample_full` の長さ契約が Interfaces 欄と矛盾、Task 2: 往復誤差の許容値が解析的に到達不能かつ `numpy` の import 欠落の 2 件、Task 3: テスト fixture が同じ plan 内の実装と矛盾、Task 4: フィールド順序が 2 節間で不一致）。plan 段階のコードは推測でしかないのに、brief 経由で「逐語で使う唯一の要件源」として実装者を拘束し、reviewer はそれへの準拠で判定するため、**plan 時点の推測の質が実装品質の上限になる**。Task 5-9 は最も統合度が高く（既存の非同期ループ、デバイス再接続の契約、ワーカーの状態機械に触る）投機が最も外れやすいので、そこを実装者の判断に委ねる。

実測で得た事実のうちコードでないもの（hop 量子化が起きる条件、`retry.py` の `[T: _Closable]` 束縛、録音のフレーム計数が定数である点）は、実装指示ではなく各 task の「制約」と「受入基準」として残してある。
