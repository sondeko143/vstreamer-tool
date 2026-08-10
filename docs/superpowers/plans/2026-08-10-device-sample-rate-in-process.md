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

**Files:**
- Modify: `vspeech/stream_vc/capture.py:58-70`（`open_stream_vc_input_stream`）、`73-125`（`_capture_read_loop`）、`128-167`（`capture_loop`）
- Test: `tests/test_stream_vc_capture_resample.py`

**Interfaces:**
- Consumes: `make_resampler`（Task 1）, `resolve_device_rate`（Task 3）, `StreamVcConfig.input_device_rate`（Task 4）
- Produces:
  - `open_stream_vc_input_stream(config: StreamVcConfig, hop: int) -> tuple[sd.RawInputStream, PolyphaseResampler | None, int]` — `(stream, resampler, device_hop)`
  - `class HopAccumulator(hop: int)` — `push(x) -> list[NDArray[np.float32]]`（正確に `hop` サンプルのブロックだけ返す）, `reset()`

デバイスは `dev_rate` で開き、1 回の read で `device_hop = round(hop * dev_rate / CAPTURE_RATE)` フレーム読む。リサンプル後は `HopAccumulator` でちょうど `hop` サンプルへ切り直す（比が割り切れる常用ケースでは 1 read = 1 ブロックだが、割り切れない設定でも壊れないようにする）。**事前充填は入れない** — Task 1 の `test_fixed_hop_cadence_needs_no_priming` が示すとおり不要。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_stream_vc_capture_resample.py` を新規作成:

```python
"""Capture-side resampling and re-blocking (ADR-0070)."""

import numpy as np

from vspeech.stream_vc.capture import CAPTURE_RATE
from vspeech.stream_vc.capture import HopAccumulator
from vspeech.stream_vc.capture import device_hop_size


def test_device_hop_maps_the_pipeline_hop_onto_the_device_clock() -> None:
    assert device_hop_size(2560, 48000) == 7680
    assert device_hop_size(2560, 44100) == 7056
    assert device_hop_size(2560, CAPTURE_RATE) == 2560


def test_accumulator_emits_only_whole_hops() -> None:
    acc = HopAccumulator(4)
    assert acc.push(np.arange(3, dtype=np.float32)) == []
    blocks = acc.push(np.arange(3, 9, dtype=np.float32))
    assert len(blocks) == 2
    assert np.array_equal(blocks[0], np.arange(0, 4, dtype=np.float32))
    assert np.array_equal(blocks[1], np.arange(4, 8, dtype=np.float32))
    assert acc.push(np.zeros(0, dtype=np.float32)) == []


def test_accumulator_reset_drops_the_partial_block() -> None:
    acc = HopAccumulator(4)
    acc.push(np.ones(3, dtype=np.float32))
    acc.reset()
    assert acc.push(np.zeros(3, dtype=np.float32)) == []


def test_one_device_read_yields_exactly_one_block_at_48k() -> None:
    """The capture cadence must stay 1:1 with the device clock -- otherwise the
    conversion loop starves on some ticks and gets two blocks on others."""
    from vspeech.lib.resample import make_resampler

    hop = 2560
    dev_rate = 48000
    resampler = make_resampler(dev_rate, CAPTURE_RATE)
    assert resampler is not None
    acc = HopAccumulator(hop)
    chunk = np.zeros(device_hop_size(hop, dev_rate), dtype=np.float32)
    for tick in range(100):
        blocks = acc.push(resampler.process(chunk))
        assert len(blocks) == 1, f"tick {tick}: {len(blocks)} block(s)"
        assert blocks[0].shape == (hop,)


def test_one_device_read_yields_exactly_one_block_at_44k1() -> None:
    from vspeech.lib.resample import make_resampler

    hop = 2560
    resampler = make_resampler(44100, CAPTURE_RATE)
    assert resampler is not None
    acc = HopAccumulator(hop)
    chunk = np.zeros(device_hop_size(hop, 44100), dtype=np.float32)
    for tick in range(100):
        assert len(acc.push(resampler.process(chunk))) == 1, f"tick {tick}"


def test_native_rate_device_needs_no_resampler() -> None:
    from vspeech.lib.resample import make_resampler

    assert make_resampler(CAPTURE_RATE, CAPTURE_RATE) is None
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc_capture_resample.py -q`
Expected: FAIL — `ImportError: cannot import name 'HopAccumulator'`

- [ ] **Step 3: `capture.py` を実装**

先頭の import に追加:

```python
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
```

`ms_to_samples` の直後（50 行目付近）に追加:

```python
def device_hop_size(hop: int, device_rate: int) -> int:
    """The pipeline hop expressed in device-clock frames."""
    return round(hop * device_rate / CAPTURE_RATE)


class HopAccumulator:
    """Re-blocks a variable-length sample stream into exact `hop`-sized blocks.

    The conversion core takes a fixed block (ADR-0053), but a resampler's output
    length is not tied to the device's read size once the ratio is not exact. Buffer
    the remainder and emit only whole hops.
    """

    def __init__(self, hop: int) -> None:
        self._hop = hop
        self._buf: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        if samples.shape[0]:
            self._buf = np.concatenate([self._buf, samples])
        blocks: list[NDArray[np.float32]] = []
        while self._buf.shape[0] >= self._hop:
            blocks.append(np.ascontiguousarray(self._buf[: self._hop]))
            self._buf = self._buf[self._hop :]
        return blocks
```

`open_stream_vc_input_stream`（58-70 行）を差し替え:

```python
def open_stream_vc_input_stream(
    config: StreamVcConfig, hop: int
) -> tuple[sd.RawInputStream, PolyphaseResampler | None, int]:
    """Open the mic at its native rate and build the resampler down to CAPTURE_RATE.

    Opening at CAPTURE_RATE instead would hand the conversion to the OS, whose
    filter we cannot inspect or test, and would make WASAPI unopenable outright
    (shared mode rejects any rate but the mix format) -- ADR-0070.
    """
    device = resolve_stream_vc_input_device(config)
    device_rate, how = resolve_device_rate(
        device,
        config.input_device_rate,
        input=True,
        config_key="stream_vc.input_device_rate",
    )
    resampler = make_resampler(device_rate, CAPTURE_RATE)
    logger.info(
        "stream_vc input device %s: %s @%dHz (%s)%s",
        device.index,
        device.name,
        device_rate,
        how,
        ""
        if resampler is None
        else f" -> {CAPTURE_RATE}Hz 変換 (遅延 {resampler.delay_samples} sample)",
    )
    stream = sd.RawInputStream(
        samplerate=device_rate,
        blocksize=device_hop_size(hop, device_rate),
        device=device.index,
        channels=1,
        dtype="int16",
        latency="low",
    )
    stream.start()
    return stream, resampler, device_hop_size(hop, device_rate)
```

`run_with_device_retry` は `[T: _Closable]` に束縛されていて `close_quietly(stream)` を呼ぶので、`open_stream` はストリームだけを返さなければならない（タプルは返せない）。`retry.py` の契約は変えず、per-connection の値は保持オブジェクトで渡す。`ms_to_samples` の import 群のあとに追加:

```python
@dataclass
class _CaptureState:
    """Per-connection values that open_stream produces and the read loop consumes.

    run_with_device_retry owns the stream (it closes it), so its open_stream must
    return the stream itself. The resampler and the device-clock hop travel through
    this holder instead of widening that contract.
    """

    resampler: PolyphaseResampler | None = None
    device_hop: int = 0
```

`from dataclasses import dataclass` を import に足す。

`_capture_read_loop`（73-125 行）を丸ごと差し替え（docstring は既存のものを保持しつつ、リサンプルと再ブロック化の説明を足す）:

```python
async def _capture_read_loop(
    stream: sd.RawInputStream,
    state: _CaptureState,
    accumulator: HopAccumulator,
    out_queue: Queue[CaptureItem],
    running: Event,
) -> None:
    """Steady state: keep reading a device-clock hop at a time until a device fault.

    The device runs at its own rate, so each read is `state.device_hop` frames and the
    resampler maps them onto CAPTURE_RATE; `accumulator` cuts the result back into the
    exact `hop`-sized blocks the conversion core needs (ADR-0053/0070). At the common
    ratios one read yields exactly one block, but the accumulator keeps a ratio that
    does not divide evenly from desynchronising the pipeline.

    Device loss surfaces as stream.read() raising (OSError, sd.PortAudioError). It is not
    caught here; it escapes to run_with_device_retry, which recovers within the subsystem
    via close -> backoff -> reopen (without dragging in the sibling vc/playback tasks or
    the utterance path, ADR-0050). `while stream.active` would return silently on
    deactivate and could stall siblings waiting in get()/recv() without a word, so this
    is `while True`.

    `running` is the pause/resume gate shared with the utterance path
    (`context.running`). Capture is **not** stopped by it -- ADR-0050 decided that
    capture keeps running while paused and drop_oldest_put discards the backlog; it is
    consulted here only to avoid misreporting those drops as an anomaly.
    """
    # A drop while running = real backpressure. Throttle by time (ADR-0062).
    drop_throttle = LogThrottle()
    # An input overflow means the reader was late, which persists once it starts, so this
    # fires on every block (about 6 a second at block_ms=160) until it clears. Thin it by
    # time and meter it every occurrence -- exactly what its counterpart on the sink side
    # (playback.py's paOutputUnderflowed) already does.
    overflow_throttle = LogThrottle()
    while True:
        data, overflowed = await to_thread(stream.read, state.device_hop)
        if overflowed:
            telemetry.record("stream_vc_capture_overflow", 1.0)
            if (n := overflow_throttle.hit()) is not None:
                logger.warning("stream_vc capture input overflow (total %d)", n)
        samples = pcm16_to_float32(bytes(data))
        if state.resampler is not None:
            samples = state.resampler.process(samples)
        for block in accumulator.push(samples):
            if not drop_oldest_put(out_queue, block):
                if not running.is_set():
                    # While paused vc_loop stops consuming, so the queue stays full and
                    # every subsequent block is dropped. That is exactly the behaviour
                    # ADR-0050 intended (do not accumulate paused audio) and not an
                    # anomaly, so no warning. Warning every time would emit about 6 lines
                    # a second at block_ms=160 for the whole pause and make the warning
                    # meaningless. They are still not discarded silently: they are counted
                    # under a pause-specific stage -- mixing them into the same stage would
                    # pollute the backpressure metric (stream_vc_capture_drop, used to
                    # assess RTF) with the length of the pause.
                    telemetry.record("stream_vc_capture_drop_paused", 1.0)
                    continue
                telemetry.record("stream_vc_capture_drop", 1.0)
                if (n := drop_throttle.hit()) is not None:
                    logger.warning(
                        "stream_vc capture queue full; dropped oldest block (total %d)",
                        n,
                    )
```

`capture_loop`（128-167 行）の末尾、`_signal_reopen` の定義から `run_with_device_retry` の呼び出しまでを差し替え（先頭の docstring はそのまま残す）:

```python
    state = _CaptureState()
    accumulator = HopAccumulator(hop)

    def _open() -> sd.RawInputStream:
        stream, resampler, device_hop = open_stream_vc_input_stream(config, hop)
        state.resampler = resampler
        state.device_hop = device_hop
        # A reopened device restarts from silence, so the filter tail and the partial
        # block of the pre-fault stream must not be spliced onto it. open_stream_vc_
        # input_stream hands back a fresh resampler, so only the accumulator needs
        # clearing here.
        accumulator.reset()
        return stream

    def _signal_reopen() -> None:
        drop_oldest_put(out_queue, CaptureSignal.REOPEN)

    # Wait for the VC warmup to finish before opening the mic. Opening earlier lets the
    # audio that accumulated in real time during model loading flood the queue right
    # after startup, causing a storm of drops and filling the first few hundred ms with
    # stale audio (confirmed in the logs on real hardware).
    await ready.wait()
    await run_with_device_retry(
        open_stream=_open,
        run=lambda stream: _capture_read_loop(
            stream, state, accumulator, out_queue, running
        ),
        worker="stream_vc",
        label="stream vc capture",
        on_reopen=_signal_reopen,
        reopen_metric="stream_vc_capture_reopen",
    )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc_capture_resample.py tests/test_stream_vc_capture.py -q`
Expected: PASS

- [ ] **Step 5: stream_vc 全体の既存テストが緑か確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc.py tests/test_stream_vc_capture.py tests/test_stream_vc_config.py tests/test_stream_vc_consumer.py tests/test_stream_vc_entrypoint.py tests/test_stream_vc_envelope.py tests/test_stream_vc_gate.py -q`
Expected: PASS

- [ ] **Step 6: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/stream_vc/capture.py tests/test_stream_vc_capture_resample.py
git commit -m "feat(stream_vc): 入口をネイティブレートで開き 16k へ自前リサンプルする (ADR-0070)"
```

---

### Task 6: stream_vc の出口 (`playback.py` / `consumer.py`)

**Files:**
- Modify: `vspeech/stream_vc/playback.py:42-55`（`open_stream_vc_output_stream`）、`58-173`（`playback_loop`）
- Modify: `vspeech/stream_vc/consumer.py:87-100`（セッション変更時の扱い）、`115-136`（書き込み）
- Test: `tests/test_stream_vc_output_resample.py`

**Interfaces:**
- Consumes: `make_resampler`, `resolve_device_rate`, `StreamVcConfig.output_device_rate`, `decode_pcm` / `encode_pcm`
- Produces:
  - `open_stream_vc_output_stream(config: StreamVcConfig) -> tuple[sd.RawOutputStream, int]` — `(stream, device_rate)`。**`sample_rate` 引数が無くなる**（呼び出し側 2 箇所を直す）
  - `class OutputResampler(device_rate: int)` — `convert(pcm: bytes, src_rate: int) -> bytes`（int16 モノラル前提）, `reset()`

出力ストリームのレートがパケット由来でなくなるので、**セッション変更でストリームを閉じる必要がなくなる**（`consumer.py:91-94`）。閉じるのはデバイス障害時だけになる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_stream_vc_output_resample.py` を新規作成:

```python
"""Output-side resampling for the streaming path (ADR-0070)."""

import numpy as np

from vspeech.stream_vc.playback import OutputResampler


def _tone_pcm(freq: float, rate: int, seconds: float = 0.16) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16).tobytes()


def test_converts_the_model_rate_to_the_device_rate() -> None:
    out = OutputResampler(48000)
    pcm = _tone_pcm(440.0, 40000)
    converted = out.convert(pcm, 40000)
    in_frames = len(pcm) // 2
    assert len(converted) // 2 == round(in_frames * 48000 / 40000)


def test_matching_rates_pass_the_bytes_through_untouched() -> None:
    out = OutputResampler(48000)
    pcm = _tone_pcm(440.0, 48000)
    assert out.convert(pcm, 48000) is pcm


def test_state_is_continuous_across_packets() -> None:
    """Two consecutive packets must convert the same as one long buffer, or every
    packet boundary becomes a click."""
    whole = _tone_pcm(440.0, 40000, seconds=0.32)
    half = len(whole) // 2
    half -= half % 2
    one = OutputResampler(48000)
    joined = one.convert(whole[:half], 40000) + one.convert(whole[half:], 40000)
    other = OutputResampler(48000)
    single = other.convert(whole, 40000)
    a = np.frombuffer(joined, dtype=np.int16).astype(np.float32)
    b = np.frombuffer(single, dtype=np.int16).astype(np.float32)
    assert len(a) == len(b)
    assert np.max(np.abs(a - b)) <= 1.0


def test_source_rate_change_rebuilds_the_resampler() -> None:
    out = OutputResampler(48000)
    out.convert(_tone_pcm(440.0, 40000), 40000)
    converted = out.convert(_tone_pcm(440.0, 24000), 24000)
    assert len(converted) // 2 == round(int(24000 * 0.16) * 48000 / 24000)


def test_overshoot_saturates_instead_of_wrapping() -> None:
    """Resampling a near-full-scale signal overshoots; a wrapping cast would flip the
    sign and click."""
    out = OutputResampler(48000)
    # A near-full-scale tone whose resampled peaks exceed the input peaks.
    t = np.arange(4000) / 40000
    loud = np.rint(np.sin(2 * np.pi * 3000 * t) * 32700).astype(np.int16)
    converted = np.frombuffer(out.convert(loud.tobytes(), 40000), dtype=np.int16)
    # A wrapping cast turns an overshoot of +1.02 into roughly -32700: the sign flips
    # while the magnitude stays large. Saturation cannot produce that, because the
    # resampled envelope never actually swings that fast between adjacent samples.
    flipped = (np.abs(np.diff(converted.astype(np.int32))) > 60000).sum()
    assert flipped == 0, f"{flipped} wrap-around discontinuities"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc_output_resample.py -q`
Expected: FAIL — `ImportError: cannot import name 'OutputResampler'`

- [ ] **Step 3: `playback.py` を実装**

import に追加:

```python
from vspeech.config import SampleFormat
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
```

`detect_gap` の直後に追加:

```python
class OutputResampler:
    """Converts a packet's PCM from the producer's rate to the output device's rate.

    The stream is continuous across packets, so the filter state is kept and only
    rebuilt when the source rate actually changes (a new producer session with a
    different model). Rebuilding per packet would put a filter edge at every packet
    boundary.
    """

    def __init__(self, device_rate: int) -> None:
        self._device_rate = device_rate
        self._src_rate: int | None = None
        self._resampler: PolyphaseResampler | None = None

    def reset(self) -> None:
        if self._resampler is not None:
            self._resampler.reset()

    def convert(self, pcm: bytes, src_rate: int) -> bytes:
        if src_rate != self._src_rate:
            self._src_rate = src_rate
            self._resampler = make_resampler(src_rate, self._device_rate)
            if self._resampler is not None:
                logger.info(
                    "stream_vc output resample %dHz -> %dHz (遅延 %d sample)",
                    src_rate,
                    self._device_rate,
                    self._resampler.delay_samples,
                )
        if self._resampler is None:
            return pcm
        samples = decode_pcm(pcm, SampleFormat.INT16, channels=1)
        return encode_pcm(self._resampler.process(samples), SampleFormat.INT16)
```

`open_stream_vc_output_stream`（42-55 行）を差し替え:

```python
def open_stream_vc_output_stream(
    config: StreamVcConfig,
) -> tuple[sd.RawOutputStream, int]:
    """Open the sink at its native rate (ADR-0070).

    The rate no longer comes from the packet, so the stream survives a producer
    session that switches to a model with a different sample rate -- only the
    resampler is rebuilt.
    """
    device = resolve_stream_vc_output_device(config)
    device_rate, how = resolve_device_rate(
        device,
        config.output_device_rate,
        input=False,
        config_key="stream_vc.output_device_rate",
    )
    logger.info(
        "stream_vc output device %s: %s @%dHz (%s)",
        device.index,
        device.name,
        device_rate,
        how,
    )
    stream = sd.RawOutputStream(
        samplerate=device_rate,
        channels=1,
        device=device.index,
        dtype="int16",
        latency="low",
    )
    stream.start()
    return stream, device_rate
```

`playback_loop` 内の変更:
- `stream = open_stream_vc_output_stream(config, packet.sample_rate)` の 2 箇所（117・123 行）を `stream, device_rate = open_stream_vc_output_stream(config)` にし、直後に `converter = OutputResampler(device_rate)` を作る（`converter` はループ外で `None` 初期化し、ここで代入）。
- `await to_thread(stream.write, packet.pcm)`（141 行）を次に:

```python
                pcm = converter.convert(packet.pcm, packet.sample_rate)
                underflowed = await to_thread(stream.write, pcm)
```

- デバイス障害の `except` 節（148-163 行）で `stream = None` にするとき、`converter = None` も一緒にクリアする（次のオープンで作り直す）。

- [ ] **Step 4: `consumer.py` を実装**

import に `from vspeech.stream_vc.playback import OutputResampler` を追加。

- `open_stream_vc_output_stream(config, packet.sample_rate)` の 2 箇所（118-120・124-126 行）を `stream, device_rate = open_stream_vc_output_stream(config)` にし、`converter = OutputResampler(device_rate)` を作る。
- セッション変更の分岐（87-96 行）から**ストリームを閉じる処理を外す**。コメントを次に差し替え:

```python
            if packet.session_id != session:
                if session is not None:
                    logger.info("stream_vc consumer: producer session changed; reset")
                    # The output stream now runs at the device's own rate, so a new
                    # session with a different model rate no longer needs a reopen
                    # (ADR-0070). Only the resampler is rebuilt, which convert() does
                    # on its own when the source rate changes; reset it so the filter
                    # state from the old session is not spliced onto the new one.
                    if converter is not None:
                        converter.reset()
                session = packet.session_id
                buffer.reset()
```

- `await to_thread(stream.write, result.pcm)`（130 行）を:

```python
                pcm = converter.convert(result.pcm, packet.sample_rate)
                underflowed = await to_thread(stream.write, pcm)
```

- 障害時の `except` 節で `converter = None` もクリアする。

**注意**: `result.pcm` は jitter buffer の concealment を通っていることがあり、その場合も int16 モノラルなので扱いは同じ。ただし concealment のブロック長は `_block_bytes` 由来なので、`packet.sample_rate` と整合していることを確認すること。

- [ ] **Step 5: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_stream_vc_output_resample.py tests/test_stream_vc_consumer.py tests/test_stream_vc.py -q`
Expected: PASS

- [ ] **Step 6: consumer が torch を引かないことを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_forbidden_imports.py -q`
Expected: PASS

- [ ] **Step 7: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/stream_vc/playback.py vspeech/stream_vc/consumer.py tests/test_stream_vc_output_resample.py
git commit -m "feat(stream_vc): 出口を固定のデバイスレートで開き自前リサンプルする (ADR-0070)"
```

---

### Task 7: 発話系の録音 (`vspeech/worker/recording.py`)

**Files:**
- Modify: `vspeech/worker/recording.py:33-44`（`open_input_stream`）、`76-161`（`sd_recording_worker`）
- Test: `tests/test_recording_resample.py`

**Interfaces:**
- Consumes: `make_resampler`, `resolve_device_rate`, `decode_pcm` / `encode_pcm`, `RecordingConfig.input_device_rate`
- Produces: `open_input_stream(config: RecordingConfig) -> tuple[sd.RawInputStream, PolyphaseResampler | None, int]`

**下流の契約は不変**: `SoundOutput.rate` は今までどおり `config.rate`。transcription も vc も無改造。

**必ず直す罠**: `interval_frame_count += config.chunk`（102 行）。リサンプル後は 1 read が `config.chunk` フレームとは限らないので、**実際に得られたフレーム数**で数える。ここを直さないと `interval_sec` / `silence_threshold` / `max_recording_sec` の時間換算が全部ずれる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_recording_resample.py` を新規作成:

```python
"""Recording-side resampling keeps the downstream contract and the time base
(ADR-0070)."""

import numpy as np

from vspeech.config import RecordingConfig
from vspeech.config import SampleFormat
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import make_resampler
from vspeech.worker.recording import device_chunk_size
from vspeech.worker.recording import utterance_capture_sec


def test_device_chunk_maps_the_configured_chunk_onto_the_device_clock() -> None:
    assert device_chunk_size(1024, 48000, 16000) == 3072
    assert device_chunk_size(1024, 16000, 16000) == 1024


def test_resampled_frames_carry_the_configured_rate_downstream() -> None:
    """A one-second capture at 48 kHz must become one second at config.rate."""
    config = RecordingConfig(rate=16000, channels=1, format=SampleFormat.INT16)
    resampler = make_resampler(48000, config.rate)
    assert resampler is not None
    chunk = device_chunk_size(1024, 48000, config.rate)
    frames = b""
    for _ in range(48000 // chunk):
        block = np.zeros(chunk, dtype=np.float32)
        frames += encode_pcm(resampler.process(block), config.format)
    seconds = utterance_capture_sec(frames, config)
    assert abs(seconds - (48000 // chunk) * chunk / 48000) < 0.01


def test_interval_counting_uses_resampled_frames_not_the_device_chunk() -> None:
    """Counting the device chunk would make the silence/length clock run 3x fast at
    48 kHz. Guard the arithmetic the worker relies on."""
    config = RecordingConfig(rate=16000)
    resampler = make_resampler(48000, config.rate)
    assert resampler is not None
    device_chunk = device_chunk_size(1024, 48000, config.rate)
    produced = sum(
        len(resampler.process(np.zeros(device_chunk, dtype=np.float32)))
        for _ in range(50)
    )
    # 50 device reads of 3072 frames = 153600 device frames = 3.2 s -> 51200 at 16 kHz.
    assert abs(produced - 51200) <= 2
    assert produced != 50 * device_chunk


def test_multichannel_capture_keeps_its_channels() -> None:
    config = RecordingConfig(rate=16000, channels=2, format=SampleFormat.INT16)
    resampler = make_resampler(48000, config.rate)
    assert resampler is not None
    interleaved = np.zeros(3072 * config.channels, dtype=np.float32)
    decoded = decode_pcm(
        encode_pcm(interleaved, config.format), config.format, config.channels
    )
    assert decoded.shape == (3072, 2)
    out = resampler.process(decoded)
    assert out.shape == (1024, 2)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_recording_resample.py -q`
Expected: FAIL — `ImportError: cannot import name 'device_chunk_size'`

- [ ] **Step 3: `recording.py` を実装**

import に追加:

```python
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import PolyphaseResampler
from vspeech.lib.resample import make_resampler
```

`open_input_stream`（33-44 行）を差し替え:

```python
def device_chunk_size(chunk: int, device_rate: int, config_rate: int) -> int:
    """The configured chunk expressed in device-clock frames."""
    return round(chunk * device_rate / config_rate)


def open_input_stream(
    config: RecordingConfig,
) -> tuple[sd.RawInputStream, PolyphaseResampler | None, int]:
    """Open the mic at its native rate; resample down to config.rate in-process.

    Opening at config.rate would hand the conversion to the OS (ADR-0070). The
    downstream contract is unchanged: what leaves this worker is still config.rate.
    """
    device = resolve_input_device(config)
    device_rate, how = resolve_device_rate(
        device,
        config.input_device_rate,
        input=True,
        config_key="recording.input_device_rate",
    )
    resampler = make_resampler(device_rate, config.rate)
    chunk = device_chunk_size(config.chunk, device_rate, config.rate)
    logger.info(
        "use input device %s: %s @%dHz (%s)%s",
        device.index,
        device.name,
        device_rate,
        how,
        "" if resampler is None else f" -> {config.rate}Hz 変換",
    )
    stream = sd.RawInputStream(
        samplerate=device_rate,
        blocksize=chunk,
        device=device.index,
        channels=config.channels,
        dtype=get_sd_dtype(config.format),
    )
    stream.start()
    return stream, resampler, chunk
```

`sd_recording_worker`（76-161 行）の変更:

- 89-90 行を差し替え:

```python
        with worker_startup("recording"):
            stream, resampler, device_chunk = open_input_stream(config)
```

- 96-103 行（read から `interval_frames += in_data` まで）を差し替え:

```python
                chunk_data, overflowed = await to_thread(stream.read, device_chunk)
                if overflowed:
                    # sounddevice reports an overflow with a flag rather than an
                    # exception, so at least leave a log line.
                    logger.warning("recording input overflow: samples were dropped")
                in_data = bytes(chunk_data)
                if resampler is not None:
                    decoded = decode_pcm(in_data, config.format, config.channels)
                    in_data = encode_pcm(resampler.process(decoded), config.format)
                # Count the frames we actually produced, not the device read size.
                # After resampling they differ (3072 device frames -> 1024 at 16 kHz),
                # and counting the device size would run the silence / interval_sec /
                # max_recording_sec clock at the device's rate instead of config.rate.
                frame_count = len(in_data) // (
                    get_sample_size(config.format) * config.channels
                )
                interval_frame_count += frame_count
                interval_frames += in_data
```

- `finally: stream.close()`（160-161 行）はそのまま。ジェネレータが作り直されるたびに `open_input_stream` が新しい resampler を返すので、明示的な `reset()` は不要。

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_recording_resample.py tests/test_recording_metrics.py tests/test_recording_trace.py -q`
Expected: PASS

- [ ] **Step 5: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/worker/recording.py tests/test_recording_resample.py
git commit -m "feat(recording): 録音デバイスをネイティブレートで開き config.rate へ自前リサンプルする (ADR-0070)"
```

---

### Task 8: 発話系の再生 (`vspeech/worker/playback.py`)

**Files:**
- Modify: `vspeech/worker/playback.py:64-135`（`OutputStream`）
- Test: `tests/test_playback_resample.py`

**Interfaces:**
- Consumes: `make_resampler`, `resolve_device_rate`, `decode_pcm` / `encode_pcm`, `PlaybackConfig.output_device_rate`
- Produces: `OutputStream.device_rate: int`, `OutputStream.prepare(data, rate, format, channels) -> bytes`

**中心の変更**: デバイスは `device_rate` 固定で開く。`update_stream_if_changed` の再オープン条件から **rate が外れる**（format / channels / デバイス同一性の変化だけ）。ソースのレート変化はリサンプラの作り直しで吸収する。

**ワンショットである点に注意**: 発話は 1 件ずつ独立した buffer なので `process()` ではなく **`resample_full()`** を使う。`process()` だと末尾 `delay_samples` ぶん（3ms 前後）が毎回フィルタ内に残って出てこない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_playback_resample.py` を新規作成:

```python
"""Utterance playback: fixed device rate + one-shot resampling (ADR-0070)."""

import numpy as np
import pytest

from vspeech.config import SampleFormat


def _tone(freq: float, rate: int, seconds: float) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16).tobytes()


@pytest.fixture
def output_stream(monkeypatch: pytest.MonkeyPatch):
    """An OutputStream whose device resolution and PortAudio calls are stubbed."""
    import vspeech.worker.playback as pb
    from vspeech.config import PlaybackConfig
    from vspeech.lib.audio import DeviceInfo

    device = DeviceInfo(
        host_api=2,
        max_input_channels=0,
        max_output_channels=2,
        name="Stub Speakers",
        index=7,
    )
    monkeypatch.setattr(pb, "get_output_device", lambda config: device)
    monkeypatch.setattr(
        pb, "resolve_device_rate", lambda *a, **k: (48000, "テスト用スタブ")
    )
    return pb.OutputStream(PlaybackConfig())


def test_device_rate_is_resolved_once(output_stream) -> None:
    assert output_stream.device_rate == 48000


def test_prepare_converts_the_source_rate_to_the_device_rate(output_stream) -> None:
    pcm = _tone(440.0, 24000, 0.5)
    out = output_stream.prepare(pcm, 24000, SampleFormat.INT16, 1)
    assert len(out) // 2 == round((len(pcm) // 2) * 48000 / 24000)


def test_prepare_passes_matching_rates_through_untouched(output_stream) -> None:
    pcm = _tone(440.0, 48000, 0.5)
    assert output_stream.prepare(pcm, 48000, SampleFormat.INT16, 1) is pcm


def test_one_shot_keeps_the_tail_of_the_utterance(output_stream) -> None:
    """The whole utterance comes out, tail included.

    Keeping streaming state across utterances would drop the trailing
    `delay_samples` of each one. Note the very last samples legitimately taper: any
    FIR reconstructs the signal edge from a window that is half past the end. So
    assert the last 20 ms still carries the tone, not that every sample is full
    scale. The precise numeric fidelity of one-shot mode is Task 1's job
    (test_resample_full_keeps_length_and_alignment, -60 dB against a reference).
    """
    rate = 24000
    n = int(rate * 0.3)
    pcm = _tone(1000.0, rate, 0.3)
    out = np.frombuffer(
        output_stream.prepare(pcm, rate, SampleFormat.INT16, 1), dtype=np.int16
    )
    assert len(out) == round(n * 48000 / rate)
    assert np.max(np.abs(out[-960:])) > 18000


def test_consecutive_utterances_are_independent(output_stream) -> None:
    """No state may leak between utterances -- they are separated by silence."""
    pcm = _tone(440.0, 24000, 0.2)
    first = output_stream.prepare(pcm, 24000, SampleFormat.INT16, 1)
    second = output_stream.prepare(pcm, 24000, SampleFormat.INT16, 1)
    assert first == second


def test_source_rate_change_does_not_reopen_the_device(output_stream, monkeypatch) -> None:
    """TTS at 24 kHz followed by VC at 40 kHz used to close and reopen the device."""
    opened: list[int] = []

    class _FakeStream:
        def __init__(self, **kwargs) -> None:
            opened.append(kwargs["samplerate"])

        def start(self) -> None: ...
        def close(self) -> None: ...
        def write(self, data) -> None: ...

    import vspeech.worker.playback as pb

    monkeypatch.setattr(pb.sd, "RawOutputStream", _FakeStream)
    monkeypatch.setattr(
        pb, "get_device_info", lambda index: output_stream.device
    )
    monkeypatch.setattr(
        output_stream, "search_appropriate_device", lambda: output_stream.device
    )
    output_stream.update_stream_if_changed(24000, SampleFormat.INT16, 1)
    output_stream.update_stream_if_changed(40000, SampleFormat.INT16, 1)
    assert opened == [48000]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_playback_resample.py -q`
Expected: FAIL — `AttributeError: module 'vspeech.worker.playback' has no attribute 'resolve_device_rate'`

- [ ] **Step 3: `playback.py` を実装**

import に追加:

```python
from vspeech.lib.audio import resolve_device_rate
from vspeech.lib.pcm import decode_pcm
from vspeech.lib.pcm import encode_pcm
from vspeech.lib.resample import make_resampler
```

`OutputStream`（64-135 行）を差し替え。`rate` フィールドは「デバイスを開いているレート」の意味になる:

```python
@dataclass
class OutputStream:
    config: InitVar[PlaybackConfig]
    rate: int = 0
    format: SampleFormat = SampleFormat.INVALID
    channels: int = 0
    stream: sd.RawOutputStream | None = None
    device: DeviceInfo = field(init=False)
    device_rate: int = field(init=False, default=0)

    def __post_init__(self, config: PlaybackConfig) -> None:
        self.device = get_output_device(config=config)
        self.device_rate, how = resolve_device_rate(
            self.device,
            config.output_device_rate,
            input=False,
            config_key="playback.output_device_rate",
        )
        logger.info(
            "setting device %s: %s @%dHz (%s)",
            self.device.index,
            self.device.name,
            self.device_rate,
            how,
        )

    def prepare(
        self, data: bytes, rate: int, format: SampleFormat, channels: int
    ) -> bytes:
        """Convert one utterance's PCM to the device rate.

        One-shot, not streaming: utterances are independent buffers separated by
        silence, so the resampler must flush its tail here. Keeping state across
        utterances would clip the last few ms of every one of them.
        """
        if rate == self.device_rate:
            return data
        resampler = make_resampler(rate, self.device_rate)
        if resampler is None:
            return data
        decoded = decode_pcm(data, format, channels)
        return encode_pcm(resampler.resample_full(decoded), format)

    def update_stream_if_changed(
        self,
        rate: int,
        format: SampleFormat,
        channels: int,
    ):
        """Open the sink if needed. `rate` is the SOURCE rate and deliberately does
        NOT take part in the decision: the device always runs at device_rate, so a
        source whose rate changed (TTS 24 kHz -> VC 40 kHz) no longer reopens it
        (ADR-0070). Only the format, the channel count and the device identity do.
        """
        output_device = get_device_info(self.device.index)
        if (
            self.stream
            and self.format == format
            and self.channels == channels
            and output_device.name == self.device.name
        ):
            logger.debug("stream is reused.")
            return

        if self.stream:
            self.stream.close()
        self.device = self.search_appropriate_device()
        logger.info("use device %s: %s", self.device.index, self.device.name)
        self.rate = self.device_rate
        self.format = format
        self.channels = channels
        self.stream = sd.RawOutputStream(
            samplerate=self.device_rate,
            channels=channels,
            device=self.device.index,
            dtype=get_sd_dtype(format),
        )
        self.stream.start()
```

`search_appropriate_device` と `playback` メソッドはそのまま。

`sd_playback_worker`（150-163 行）の呼び出しを差し替え:

```python
                output_stream.update_stream_if_changed(
                    rate=speech.sound.rate,
                    format=speech.sound.format,
                    channels=speech.sound.channels,
                )
                given_volume = speech.current_event.params.volume
                logger.debug("playback... %s", speech.text)
                pcm = output_stream.prepare(
                    speech.sound.data,
                    speech.sound.rate,
                    speech.sound.format,
                    speech.sound.channels,
                )
                with telemetry.timer("playback", trace_id=speech.trace_id):
                    await output_stream.playback(
                        volume=given_volume
                        if given_volume is not None
                        else config.volume,
                        data=pcm,
                    )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_playback_resample.py tests/test_playback_e2e.py -q`
Expected: PASS

- [ ] **Step 5: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/worker/playback.py tests/test_playback_resample.py
git commit -m "feat(playback): 再生デバイスを固定レートで開きワンショットで自前リサンプルする (ADR-0070)"
```

---

### Task 9: preflight でレート解決とデバイス開通を検証

**Files:**
- Modify: `vspeech/preflight.py:113-137`（`_check_recording`）、`139-149`（`_check_playback`）、`312-414`（`_check_stream_vc`）
- Test: `tests/test_preflight_device_rate.py`

**Interfaces:**
- Consumes: `resolve_device_rate`, `DeviceRateUnresolvedError`
- Produces: `_check_device_rate(device, override, *, input, worker, config_key) -> list[ConfigProblem]`（`preflight.py` 内の private ヘルパ）

出力ストリームのレートが静的になったので、**再生系も起動時に `check_output_settings` で検証できるようになる**（これまでは実行時にしかレートが決まらず不可能だった）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_preflight_device_rate.py` を新規作成:

```python
"""Preflight validates the resolved device rate before anything opens a stream
(ADR-0038 + ADR-0071)."""

import pytest

from vspeech.config import Config
from vspeech.exceptions import DeviceRateUnresolvedError
from vspeech.lib.audio import DeviceInfo
from vspeech.preflight import collect_problems

_DEVICE = DeviceInfo(
    host_api=2, max_input_channels=2, max_output_channels=2, name="Stub", index=3
)


@pytest.fixture
def base_config() -> Config:
    config = Config()
    config.recording.enable = True
    config.playback.enable = True
    return config


def test_unresolvable_rate_is_reported_with_the_config_key(
    base_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vspeech.lib.audio as audio

    monkeypatch.setattr(audio, "resolve_input_device", lambda config: _DEVICE)
    monkeypatch.setattr(audio, "resolve_output_device", lambda config: _DEVICE)

    def _boom(*args, **kwargs):
        raise DeviceRateUnresolvedError("判定できません")

    monkeypatch.setattr(audio, "resolve_device_rate", _boom)
    fields = {p.field for p in collect_problems(base_config)}
    assert "recording.input_device_rate" in fields
    assert "playback.output_device_rate" in fields


def test_undopenable_rate_is_reported(
    base_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sounddevice as sd

    import vspeech.lib.audio as audio

    monkeypatch.setattr(audio, "resolve_input_device", lambda config: _DEVICE)
    monkeypatch.setattr(audio, "resolve_output_device", lambda config: _DEVICE)
    monkeypatch.setattr(audio, "resolve_device_rate", lambda *a, **k: (192000, "stub"))

    def _reject(**kwargs):
        raise sd.PortAudioError("Invalid sample rate [PaErrorCode -9997]")

    monkeypatch.setattr(sd, "check_input_settings", _reject)
    monkeypatch.setattr(sd, "check_output_settings", _reject)
    messages = [p.message for p in collect_problems(base_config)]
    assert any("192000" in m for m in messages)


def test_resolvable_and_openable_rate_reports_nothing(
    base_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sounddevice as sd

    import vspeech.lib.audio as audio

    monkeypatch.setattr(audio, "resolve_input_device", lambda config: _DEVICE)
    monkeypatch.setattr(audio, "resolve_output_device", lambda config: _DEVICE)
    monkeypatch.setattr(audio, "resolve_device_rate", lambda *a, **k: (48000, "stub"))
    monkeypatch.setattr(sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(sd, "check_output_settings", lambda **kwargs: None)
    fields = {p.field for p in collect_problems(base_config)}
    assert "recording.input_device_rate" not in fields
    assert "playback.output_device_rate" not in fields
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_preflight_device_rate.py -q`
Expected: FAIL — レート関連の ConfigProblem がまだ出ない

- [ ] **Step 3: `preflight.py` に共通ヘルパを足す**

`_check_vad_gate` の直後に追加:

```python
def _check_device_rate(
    device,
    override: int | None,
    *,
    input: bool,
    worker: str,
    config_key: str,
) -> list[ConfigProblem]:
    """Resolve the rate the device will be opened at, then check it can be opened.

    Both halves are layer A: they need no model load and no audio to flow. The output
    side only became checkable here once the stream stopped taking its rate from the
    incoming packet (ADR-0070).
    """
    import sounddevice as sd

    from vspeech.exceptions import DeviceRateUnresolvedError
    from vspeech.lib.audio import resolve_device_rate

    try:
        rate, _ = resolve_device_rate(
            device, override, input=input, config_key=config_key
        )
    except DeviceRateUnresolvedError as e:
        return [ConfigProblem(worker, str(e), field=config_key)]
    try:
        if input:
            sd.check_input_settings(device=device.index, samplerate=rate)
        else:
            sd.check_output_settings(device=device.index, samplerate=rate)
    except Exception as e:
        return [
            ConfigProblem(
                worker,
                f"デバイス '{device.name}' を {rate}Hz で開けません: {e}",
                field=config_key,
            )
        ]
    return []
```

- [ ] **Step 4: 3 つの checker から呼ぶ**

`_check_recording`: `resolve_input_device` が成功した場合にその戻り値を使って呼ぶ。

```python
    try:
        device = resolve_input_device(config.recording)
    except DeviceNotFoundError as e:
        problems.append(ConfigProblem(w, str(e), field="recording.input_device_index"))
    else:
        problems += _check_device_rate(
            device,
            config.recording.input_device_rate,
            input=True,
            worker=w,
            config_key="recording.input_device_rate",
        )
```

`_check_playback`: 同様に `resolve_output_device` の戻り値を使い、`playback.output_device_rate` で呼ぶ（早期 return をやめて problems リストにする）。

`_check_stream_vc`: `does_vc` の入力側（359-364 行）と `does_play` の出力側（365-371 行）を同じ形にし、`stream_vc.input_device_rate` / `stream_vc.output_device_rate` で呼ぶ。

- [ ] **Step 5: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 uv run pytest tests/test_preflight_device_rate.py tests/test_preflight.py -q`
Expected: PASS

- [ ] **Step 6: lint / 型検査 / コミット**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add vspeech/preflight.py tests/test_preflight_device_rate.py
git commit -m "feat(preflight): 解決したデバイスレートで開けるかを起動時に検証する (ADR-0071)"
```

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
| 範囲外はラップせず飽和する | 2 (`encode_pcm`), 6 |
| 1 発話の末尾が欠けない | 1 (`resample_full`), 8 |
| ソースのレート変化で再生デバイスが開き直されない | 8 |
| 録音の無音判定・区間長の時間換算が実時間と一致する | 7 |
| 下流が受け取るレートが変わらない | 7 (契約不変), 8 |
| 依存パッケージが追加されない | Global Constraints |
| consumer ロールが torch 無しで動く | 6 (Step 6) |

**未解決事項:** なし。`retry.py` の `open_stream` 戻り値の扱いは Task 5 Step 3 で確定済み（`[T: _Closable]` の束縛と `close_quietly(stream)` の呼び出しがあるためタプルは返せず、`_CaptureState` 保持オブジェクトで渡す。`retry.py` は無改造）。
