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
    assert min(levels) > -0.5


@pytest.mark.parametrize(("src", "dst"), ALL)
def test_fixed_hop_blocks_match_one_shot_exactly(src: int, dst: int) -> None:
    """The pipeline's own access pattern (fixed 160 ms hops) must be bit-exact.

    This is the core continuity guarantee: no discontinuity at block seams.

    Note: this bit-exactness is an empirical property of this numpy/BLAS build,
    not a mathematical theorem -- feeding the same signal through a *different*
    block size can reorder the underlying BLAS summation and differ by up to
    4.8e-7 relative (measured on the 44100->16000 pair; see
    test_irregular_blocks_match_one_shot, which tolerates it via np.allclose).
    If this starts failing after a numpy/BLAS upgrade on a fixed block size,
    that is a real regression; a difference only when the block size itself
    changes is not.
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
def test_different_block_segmentations_agree(src: int, dst: int) -> None:
    """Two different ways of chopping the same signal into blocks must agree.

    Unlike replaying the identical computation twice (which cannot fail), this
    is a real property: it would catch state (tail/`_fed`/`_emitted`) that gets
    corrupted at some but not all block boundaries.
    """
    x = np.random.default_rng(11).standard_normal(src).astype(np.float32)
    hop_a = int(src * 0.160)
    hop_b = int(src * 0.080)
    ra = PolyphaseResampler(src, dst)
    out_a = np.concatenate(
        [ra.process(x[i : i + hop_a]) for i in range(0, len(x), hop_a)]
    )
    rb = PolyphaseResampler(src, dst)
    out_b = np.concatenate(
        [rb.process(x[i : i + hop_b]) for i in range(0, len(x), hop_b)]
    )
    assert len(out_a) == len(out_b)
    assert np.allclose(out_a, out_b, atol=1e-5, rtol=0)


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
    assert peak - (src // 2) * dst / src == pytest.approx(r.delay_samples, abs=0.5)


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
    """One-shot mode returns the full duration with the group delay removed.

    `n = src // 2 + 1` is deliberately not a multiple of `down` for any of the
    parametrized ratios (down in {1, 3, 5, 441}): it pins the contract to
    `round(n*dst/src)`, not `ceil(n*dst/src)` -- the two disagree exactly when
    n is not a multiple of `down` (e.g. 48000->16000, n=1000: round=333,
    ceil=334).
    """
    for n in (src // 2, src // 2 + 1):
        r = PolyphaseResampler(src, dst)
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


def test_channel_layout_change_after_first_block_raises() -> None:
    """A channel-count change mid-stream must raise a domain ValueError -- not
    silently reinitialise the tail (a discontinuity) and not numpy's raw
    concatenate ValueError (not a domain error)."""
    r = PolyphaseResampler(48000, 16000)
    r.process(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="channel layout"):
        r.process(np.zeros((100, 2), dtype=np.float32))

    r2 = PolyphaseResampler(48000, 16000)
    r2.process(np.zeros((100, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="channel layout"):
        r2.process(np.zeros((100, 3), dtype=np.float32))


def test_invalid_transition_width_raises() -> None:
    with pytest.raises(ValueError):
        PolyphaseResampler(48000, 16000, transition_width=2.0)
    with pytest.raises(ValueError):
        PolyphaseResampler(48000, 16000, transition_width=0.0)


def test_invalid_stopband_db_raises() -> None:
    with pytest.raises(ValueError):
        PolyphaseResampler(48000, 16000, stopband_db=0.0)
    with pytest.raises(ValueError):
        PolyphaseResampler(48000, 16000, stopband_db=-5.0)
