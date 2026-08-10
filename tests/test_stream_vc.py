import os
from pathlib import Path

import numpy as np
import pytest

from vspeech.lib.stream_vc import next_context


def test_next_context_returns_tail():
    seq = np.arange(5)
    assert list(next_context(seq, 2)) == [3, 4]


def test_next_context_zero_is_empty():
    seq = np.arange(3)
    assert len(next_context(seq, 0)) == 0


def test_next_context_clamps_when_longer_than_seq():
    # context_len > len(seq): return the whole buffer (not a negative-index slice)
    seq = np.arange(3)
    assert list(next_context(seq, 5)) == [0, 1, 2]


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


def test_emit_with_crossfade_hop_is_realtime_clock_not_render_ratio():
    """The emit length is exactly the real-time clock (block_len*sr/16000) and does not
    depend on the render length.

    Deriving it as a ratio of the render length (out_total * block_len / seq_len) shortens
    the hop by however much HuBERT's receptive field truncates off the tail (about 320
    input samples) and permanently starves the output device (measured 3.03% = 30.3ms/s).
    A regression test that pins the actual value without a GPU. "The same length every
    tick" alone would let a constant but wrong value through, so both the actual value and
    independence from out_total are asserted.
    """
    block_len, sr = 2560, 48000
    seq_len = 8000 + block_len
    expected = round(block_len * sr / 16000)
    assert expected == 7680

    # The ideal length with no truncation, and the length real hardware actually returns
    # (shorter by the receptive field).
    ideal_total = round(seq_len * sr / 16000)
    truncated_total = round((seq_len - 320) * sr / 16000)
    assert ideal_total != truncated_total

    lengths: list[int] = []
    for out_total in (ideal_total, truncated_total):
        sv = _bare_streaming_vc(block_len=block_len, target_sample_rate=sr)
        out = np.arange(out_total, dtype=np.int16)
        emitted = [sv._emit_with_crossfade(out).shape[0] for _ in range(4)]
        assert len(set(emitted)) == 1  # constant across ticks = rate lock
        lengths.append(emitted[0])

    # Exactly the expected value, and independent of the render length out_total
    # (this is the assert that catches the bug)
    assert lengths == [expected, expected]


def test_emit_delay_is_the_offset_from_the_block_start():
    """The emit's content starts `emit_delay_samples` before the start of the input block.

    The decoder render is aligned to the start of the analysis window (the truncation is
    at the tail) and the read is anchored at the tail, so the emit sounds earlier by the
    crossfade plus SOLA plus the receptive-field truncation. The VAD gate shifts its mask
    by this value when overlaying (ADR-0059), so the contract is pinned on CPU.
    """
    block_len, ctx_len, sr = 2560, 8000, 48000
    xf_len, sola_len = 400, 0
    sv = _bare_streaming_vc(
        block_len=block_len,
        context_len=ctx_len,
        crossfade_len=xf_len,
        sola_search_len=sola_len,
        target_sample_rate=sr,
    )
    # A render length shortened by the receptive field (320 input samples), as on real
    # hardware.
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    sv._emit_with_crossfade(np.arange(out_total, dtype=np.int16))

    out_hop = round(block_len * sr / 16000)
    out_xf = round(xf_len * sr / 16000)
    ctx_out = round(ctx_len * sr / 16000)
    assert sv.emit_delay_samples == ctx_out - (out_total - out_hop - out_xf)
    # The truncation (20ms) plus the crossfade (25ms). Consistent with the measured ~52ms
    # (which includes SOLA).
    assert sv.emit_delay_samples == round(0.045 * sr)


def test_emit_delay_does_not_move_with_the_sola_lag():
    """The emit delay does not move with the lag SOLA picks (it is derived from the
    nominal read position).

    Folding the lag into the time axis would re-anchor the time axis of whoever overlays
    the output with it (the VAD gate's mask) on every tick, making the gain jump at the
    emit seam (a click).
    """
    block_len, ctx_len, sr = 2560, 8000, 48000
    xf_len, sola_len = 400, 80
    sv = _bare_streaming_vc(
        block_len=block_len,
        context_len=ctx_len,
        crossfade_len=xf_len,
        sola_search_len=sola_len,
        target_sample_rate=sr,
    )
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    out_hop = round(block_len * sr / 16000)
    out_xf = round(xf_len * sr / 16000)
    out_sola = round(sola_len * sr / 16000)
    expected = round(ctx_len * sr / 16000) - (out_total - out_hop - out_xf - out_sola)

    rng = np.random.default_rng(0)
    delays = []
    for _ in range(5):
        # different content every tick -> SOLA picks a different lag each time
        out = (rng.standard_normal(out_total) * 8000).astype(np.int16)
        sv._emit_with_crossfade(out)
        delays.append(sv.emit_delay_samples)
    assert delays == [expected] * 5


def test_emit_delay_without_crossfade_is_the_receptive_field_truncation():
    """With crossfade disabled, the emit delay is exactly the receptive-field
    truncation."""
    block_len, ctx_len, sr = 2560, 8000, 48000
    sv = _bare_streaming_vc(
        block_len=block_len,
        context_len=ctx_len,
        crossfade_len=0,
        sola_search_len=0,
        target_sample_rate=sr,
    )
    trunc_in = 320
    out_total = round((ctx_len + block_len - trunc_in) * sr / 16000)
    out = np.arange(out_total, dtype=np.int16)
    emitted = sv._emit_no_crossfade(out)
    assert emitted.shape[0] == round(block_len * sr / 16000)
    assert sv.emit_delay_samples == round(trunc_in * sr / 16000)


def test_emit_with_crossfade_raises_when_output_shorter_than_hop():
    """When the render is shorter than one hop, fail and name the cause rather than
    silently emitting something short."""
    sv = _bare_streaming_vc()
    out = np.arange(4000, dtype=np.int16)  # < hop(7680)
    with pytest.raises(ValueError, match="context_ms"):
        sv._emit_with_crossfade(out)


def test_helpers_work_on_torch_tensors():
    # docstring claims numpy/torch agnosticism; verify the torch path.
    import pytest

    torch = pytest.importorskip("torch")
    seq = torch.arange(5)
    assert next_context(seq, 2).tolist() == [3, 4]
    assert next_context(seq, 0).numel() == 0


_CONFIG_ENV = "VSPEECH_RVC_GOLDEN_CONFIG"
_config_path = os.environ.get(_CONFIG_ENV)
_GOLDEN_CONFIG = Path(_config_path) if _config_path else None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


_gpu_gate = pytest.mark.skipif(
    not _cuda_available() or _GOLDEN_CONFIG is None or not _GOLDEN_CONFIG.exists(),
    reason=f"CUDA / ${_CONFIG_ENV} config not available",
)


@_gpu_gate
def test_streaming_vc_process_block_shape_and_finite():
    from scripts import capture_change_voice_golden as cap
    from vspeech.lib.stream_vc import StreamingVc

    assert _GOLDEN_CONFIG is not None  # gate guarantees; narrows for ty
    rt = cap.build_rvc_runtime(_GOLDEN_CONFIG)

    block_len = 640  # 40ms @ 16k
    context_len = 3200  # 200ms @ 16k
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
    )
    sv.warmup()

    import numpy as np

    from scripts.stream_vc_rtf import make_voiced_signal

    signal = make_voiced_signal(16000, 1.0, seed=0)
    out1 = sv.process_block(signal[:block_len])
    out2 = sv.process_block(signal[block_len : 2 * block_len])

    assert out1.dtype == np.int16 and out2.dtype == np.int16
    assert out1.shape[0] > 0 and out2.shape[0] > 0
    assert np.all(np.isfinite(out1)) and np.all(np.isfinite(out2))


@_gpu_gate
def test_streaming_vc_crossfade_rate_locked_and_finite():
    from scripts import capture_change_voice_golden as cap
    from vspeech.lib.stream_vc import StreamingVc

    assert _GOLDEN_CONFIG is not None
    rt = cap.build_rvc_runtime(_GOLDEN_CONFIG)

    block_len = 1280  # 80ms @ 16k
    context_len = 1600  # 100ms @ 16k
    crossfade_len = 160  # 10ms @ 16k
    sola_search_len = 80  # 5ms @ 16k -> exercise the SOLA path, not just lag 0
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
        sola_search_len=sola_search_len,
    )
    sv.warmup()

    from scripts.stream_vc_rtf import make_voiced_signal

    signal = make_voiced_signal(16000, 2.0, seed=0)
    outs = [
        sv.process_block(signal[i * block_len : (i + 1) * block_len]) for i in range(3)
    ]
    # Rate-lock invariant: emit length is the real-time hop derived from the
    # sample-rate clock, so every tick emits exactly one hop -> no drift and no
    # starvation. SOLA only moves *where* we read, never *how much* we emit, so
    # this must hold with the search window on as well. Assert the real value,
    # not just equality: a constant-but-short hop (the render-ratio bug) also
    # passes an all-equal check while starving the sink.
    expected = round(block_len * rt["target_sample_rate"] / 16000)
    assert outs[0].shape[0] == expected
    lengths = {out.shape[0] for out in outs}
    assert len(lengths) == 1  # all equal -> rate-locked, no drift
    for out in outs:
        assert out.dtype == np.int16
        assert out.shape[0] > 0
        assert np.all(np.isfinite(out))
    assert any(np.any(out != 0) for out in outs)


@pytest.mark.parametrize("correlated", [True, False])
def test_crossfade_weights_direction_and_endpoints(correlated):
    from vspeech.lib.stream_vc import crossfade_weights

    fade_in, fade_out = crossfade_weights(64, correlated=correlated)
    # Both laws (sin/cos and sin²/cos²) rise/fall the same way at the endpoints.
    assert fade_in[0] < fade_in[-1]
    assert fade_out[0] > fade_out[-1]
    assert fade_in[0] < 0.1 and fade_out[0] > 0.9
    assert fade_in[-1] > 0.9 and fade_out[-1] < 0.1


def test_crossfade_weights_correlated_is_amplitude_preserving():
    from vspeech.lib.stream_vc import crossfade_weights

    # SOLA on (correlated renders): sum-to-1 (sin²/cos²) is unity-gain for
    # correlated signals. sin²(πx/2)+cos²(πx/2)==1 exactly.
    fade_in, fade_out = crossfade_weights(64, correlated=True)
    assert np.allclose(fade_in + fade_out, 1.0, atol=1e-6)
    # Odd n places a cell centre exactly at x=0.5, where sin²=cos²=0.5.
    fi, fo = crossfade_weights(101, correlated=True)
    assert np.isclose(fi[50], 0.5, atol=1e-6)
    assert np.isclose(fo[50], 0.5, atol=1e-6)


def test_crossfade_weights_uncorrelated_is_equal_power():
    from vspeech.lib.stream_vc import crossfade_weights

    # SOLA off (uncorrelated renders): equal-power (sin/cos) keeps total power
    # flat (fi²+fo²==1). Using sum-to-1 here would notch the band by ~1.25 dB.
    fade_in, fade_out = crossfade_weights(64, correlated=False)
    assert np.allclose(fade_in**2 + fade_out**2, 1.0, atol=1e-6)
    # Centre of the equal-power law is 0.707/0.707, NOT 0.5/0.5.
    fi, fo = crossfade_weights(101, correlated=False)
    assert np.isclose(fi[50], np.sqrt(0.5), atol=1e-6)
    assert np.isclose(fo[50], np.sqrt(0.5), atol=1e-6)


def test_crossfade_weights_laws_differ_at_centre():
    from vspeech.lib.stream_vc import crossfade_weights

    # The two branches must genuinely differ: sum-to-1 centre 0.5 vs
    # equal-power centre 0.707. This is the whole point of the conditional law.
    corr_in, _ = crossfade_weights(101, correlated=True)
    uncorr_in, _ = crossfade_weights(101, correlated=False)
    assert not np.isclose(corr_in[50], uncorr_in[50], atol=1e-3)
    assert corr_in[50] < uncorr_in[50]  # 0.5 < 0.707


@pytest.mark.parametrize("correlated", [True, False])
def test_crossfade_weights_zero_is_empty(correlated):
    from vspeech.lib.stream_vc import crossfade_weights

    fade_in, fade_out = crossfade_weights(0, correlated=correlated)
    assert fade_in.shape == (0,) and fade_out.shape == (0,)


def test_overlap_add_boundaries():
    from vspeech.lib.stream_vc import crossfade_weights
    from vspeech.lib.stream_vc import overlap_add

    n = 100
    fade_in, fade_out = crossfade_weights(n, correlated=True)
    prev = np.full(n, 100.0, dtype=np.float32)
    head = np.full(n, 0.0, dtype=np.float32)
    blended = overlap_add(prev, head, fade_in, fade_out)
    # start dominated by prev (fade_out ~1), end by head (fade_out ~0)
    assert blended[0] > 99.0
    assert blended[-1] < 1.0


def test_sola_offset_finds_known_shift():
    from vspeech.lib.stream_vc import sola_offset

    rng = np.random.default_rng(0)
    # The amplitude is in int16 units (what is actually passed is
    # out.astype(np.float32)). The silence test is a ratio of full scale, so leaving unit
    # variance would read as -90dBFS = effectively silent.
    sig = (rng.standard_normal(4096) * 3000.0).astype(np.float32)
    tail = sig[1000:1500]
    shift = 37
    region = sig[1000 - 100 + shift : 1500 + 100 + shift]
    # region is sig[937:1637], so it matches tail (= sig[1000:1500]) at index
    # 1000 - 937 = 63 = 100 - shift from the start of region.
    assert sola_offset(tail, region) == 100 - shift


def test_sola_offset_centers_when_tail_digitally_silent():
    from vspeech.lib.stream_vc import sola_offset

    tail = np.zeros(100, dtype=np.float32)
    region = (np.random.default_rng(1).standard_normal(300) * 3000.0).astype(np.float32)
    # "No shift" is the centre (len(region)-n)//2, not index 0. The caller cuts region
    # starting one search half-width earlier, so 0 is the largest negative shift.
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_centers_when_tail_is_near_silent_noise_floor():
    """Even at a realistic noise floor that is not digital silence, it does not search and
    returns the centre.

    The previous absolute 1e-9 test only fired on perfect digital silence and let argmax
    correlate noise against noise, picking an effectively random lag (with no phase to
    align to, every lag looks equally plausible).
    """
    from vspeech.lib.stream_vc import sola_offset

    rng = np.random.default_rng(7)
    # A level even lower than the measured noise floor of RMS 0.000298 * 32768 ~ 9.8 int16
    # units, but not zero.
    tail = (rng.standard_normal(100) * 1.0).astype(np.float32)
    assert np.any(tail != 0.0)  # not digital silence
    region = (rng.standard_normal(300) * 1.0).astype(np.float32)
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_breaks_flat_ties_toward_center():
    """On a perfectly flat correlation surface, pick the centre (the nominal lag), not
    index 0."""
    from vspeech.lib.stream_vc import sola_offset

    # Constant DC, so the normalized correlation is 1.0 at every lag = a perfect tie.
    tail = np.full(100, 5000.0, dtype=np.float32)
    region = np.full(300, 5000.0, dtype=np.float32)
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_zero_when_region_too_short():
    from vspeech.lib.stream_vc import sola_offset

    # Not a single window fits, so a centre cannot be defined. 0 is the only sensible
    # return value.
    tail = np.full(100, 5000.0, dtype=np.float32)
    assert sola_offset(tail, np.full(50, 5000.0, dtype=np.float32)) == 0


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
    latency and RTF alone (ADR-0070). If this broke, preflight would need a new check --
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


def test_lookahead_raises_when_the_window_is_not_extended_for_it():
    """A lookahead that outruns the render (context_len not extended for it) fails loud
    rather than silently clamping.

    A silent clamp would make the measured lookahead differ from the configured one,
    defeating the point of an A/B (ADR-0070). This pins the raise side of the guard added
    in `_emit_with_crossfade`, which `test_a_large_lookahead_with_the_extended_window_...`
    only exercises from the non-raising side.
    """
    sr, block_len, ctx_len = 48000, 2560, 8000
    out_total = round((ctx_len + block_len - 320) * sr / 16000)
    out = np.arange(out_total, dtype=np.int16)
    # Far beyond what out_total - hop - xf - 2*sola leaves room for, with context_len left
    # at its un-extended default.
    lookahead_len = 20000
    out_look = round(lookahead_len * sr / 16000)
    sv = _bare_streaming_vc(target_sample_rate=sr, lookahead_len=lookahead_len)
    with pytest.raises(ValueError, match="lookahead") as excinfo:
        sv._emit_with_crossfade(out)
    assert str(out_look) in str(excinfo.value)


def test_lookahead_without_crossfade_delays_the_emit_and_keeps_the_hop_length():
    """The no-crossfade path also honours lookahead_len, symmetrically with the crossfade
    path: the emit delay moves by exactly the lookahead and the emit length stays exactly
    one hop."""
    sr, block_len, ctx_len = 48000, 2560, 8000
    trunc_in = 320
    expected_hop = round(block_len * sr / 16000)
    delays: dict[float, int] = {}
    for look_ms in (0.0, 40.0, 160.0):
        lookahead_len = round(look_ms * 16)
        ctx_len_ext = (
            ctx_len + lookahead_len
        )  # caller extends context_len, as documented
        out_total = round((ctx_len_ext + block_len - trunc_in) * sr / 16000)
        sv = _bare_streaming_vc(
            block_len=block_len,
            context_len=ctx_len_ext,
            crossfade_len=0,
            sola_search_len=0,
            target_sample_rate=sr,
            lookahead_len=lookahead_len,
        )
        out = np.arange(out_total, dtype=np.int16)
        emitted = sv._emit_no_crossfade(out)
        assert emitted.shape[0] == expected_hop
        delays[look_ms] = sv.emit_delay_samples
    for look_ms in (40.0, 160.0):
        out_look = round(round(look_ms * 16) * sr / 16000)
        assert delays[look_ms] - delays[0.0] == out_look
