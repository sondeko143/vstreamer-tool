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
):
    """モデル/GPU 無しで `_emit_with_crossfade` だけを駆動する StreamingVc。

    `__init__` は torch / rvc extra を要求するので、必要な属性だけ手で埋めた
    素のインスタンスを作る(CPU のみで emit 長の契約を固定するため)。
    """
    from vspeech.lib.stream_vc import StreamingVc

    sv = object.__new__(StreamingVc)
    sv.block_len = block_len
    sv.context_len = context_len
    sv.crossfade_len = crossfade_len
    sv.sola_search_len = sola_search_len
    sv.target_sample_rate = target_sample_rate
    sv._xfade_cache = None
    sv._output_tail = None
    return sv


def test_emit_with_crossfade_hop_is_realtime_clock_not_render_ratio():
    """emit 長は実時間クロック(block_len*sr/16000)ちょうどで、描画長に依存しない。

    描画長からの比率導出(out_total * block_len / seq_len)だと、HuBERT の受容野が
    末尾を一定量(約 320 入力サンプル)切り詰めるぶんだけ hop が短くなり、出力
    デバイスを永続的に飢えさせる(実測 3.03% = 30.3ms/s)。GPU 無しで実値を固定する
    回帰テスト。「毎tick同じ長さ」だけでは一定だが誤った値を素通しするので、
    実値と「out_total 非依存」の両方を assert する。
    """
    block_len, sr = 2560, 48000
    seq_len = 8000 + block_len
    expected = round(block_len * sr / 16000)
    assert expected == 7680

    # 切り詰め無しの理想長と、実機で実際に返ってくる長さ(受容野ぶん短い)。
    ideal_total = round(seq_len * sr / 16000)
    truncated_total = round((seq_len - 320) * sr / 16000)
    assert ideal_total != truncated_total

    lengths: list[int] = []
    for out_total in (ideal_total, truncated_total):
        sv = _bare_streaming_vc(block_len=block_len, target_sample_rate=sr)
        out = np.arange(out_total, dtype=np.int16)
        emitted = [sv._emit_with_crossfade(out).shape[0] for _ in range(4)]
        assert len(set(emitted)) == 1  # tick 間で一定 = レートロック
        lengths.append(emitted[0])

    # 実値ちょうど、かつ描画長 out_total に依存しない(← バグを捕まえる assert)
    assert lengths == [expected, expected]


def test_emit_with_crossfade_raises_when_output_shorter_than_hop():
    """描画長が 1 hop に満たないときは黙って短く出さず、原因を言って落ちる。"""
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


def test_crossfade_weights_sum_to_one():
    from vspeech.lib.stream_vc import crossfade_weights

    # New law is amplitude-preserving (sum-to-1), NOT equal-power (sum-of-squares).
    # sin²(πx/2) + cos²(πx/2) == 1 exactly, so fade_in + fade_out == 1 to float precision.
    fade_in, fade_out = crossfade_weights(64)
    assert np.allclose(fade_in + fade_out, 1.0, atol=1e-6)


def test_crossfade_weights_center_is_half():
    from vspeech.lib.stream_vc import crossfade_weights

    # Odd n places a cell centre exactly at x=0.5, where sin²=cos²=0.5. This pins the
    # sum-to-1 law: equal-power gave 0.707/0.707 at the centre, sum-to-1 gives 0.5/0.5.
    fade_in, fade_out = crossfade_weights(101)
    assert np.isclose(fade_in[50], 0.5, atol=1e-6)
    assert np.isclose(fade_out[50], 0.5, atol=1e-6)


def test_crossfade_weights_direction():
    from vspeech.lib.stream_vc import crossfade_weights

    fade_in, fade_out = crossfade_weights(64)
    # fade_in rises 0->1, fade_out falls 1->0 (endpoints: sin²(small)≈0, sin²(≈π/2)≈1)
    assert fade_in[0] < fade_in[-1]
    assert fade_out[0] > fade_out[-1]
    assert fade_in[0] < 0.1 and fade_out[0] > 0.9
    assert fade_in[-1] > 0.9 and fade_out[-1] < 0.1


def test_crossfade_weights_zero_is_empty():
    from vspeech.lib.stream_vc import crossfade_weights

    fade_in, fade_out = crossfade_weights(0)
    assert fade_in.shape == (0,) and fade_out.shape == (0,)


def test_overlap_add_boundaries():
    from vspeech.lib.stream_vc import crossfade_weights
    from vspeech.lib.stream_vc import overlap_add

    n = 100
    fade_in, fade_out = crossfade_weights(n)
    prev = np.full(n, 100.0, dtype=np.float32)
    head = np.full(n, 0.0, dtype=np.float32)
    blended = overlap_add(prev, head, fade_in, fade_out)
    # start dominated by prev (fade_out ~1), end by head (fade_out ~0)
    assert blended[0] > 99.0
    assert blended[-1] < 1.0


def test_sola_offset_finds_known_shift():
    from vspeech.lib.stream_vc import sola_offset

    rng = np.random.default_rng(0)
    # 振幅は int16 単位(実際に渡るのは out.astype(np.float32))。無音判定は
    # フルスケール比なので、単位分散のままだと -90dBFS = 実質無音扱いになる。
    sig = (rng.standard_normal(4096) * 3000.0).astype(np.float32)
    tail = sig[1000:1500]
    shift = 37
    region = sig[1000 - 100 + shift : 1500 + 100 + shift]
    # region は sig[937:1637] なので、tail (= sig[1000:1500]) と一致するのは
    # region 先頭からの index 1000 - 937 = 63 = 100 - shift。
    assert sola_offset(tail, region) == 100 - shift


def test_sola_offset_centers_when_tail_digitally_silent():
    from vspeech.lib.stream_vc import sola_offset

    tail = np.zeros(100, dtype=np.float32)
    region = (np.random.default_rng(1).standard_normal(300) * 3000.0).astype(np.float32)
    # 「シフト無し」は index 0 ではなく中央 (len(region)-n)//2。呼び出し側が
    # 探索半幅ぶん手前から region を切るので、0 は最大の負シフトになる。
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_centers_when_tail_is_near_silent_noise_floor():
    """デジタル無音ではない現実のノイズフロアでも探索せず中央を返す。

    従来の絶対値 1e-9 判定は完全なデジタル無音でしか発火せず、ノイズ同士を
    相関させて argmax が事実上ランダムな lag を選んでいた(合わせるべき位相が
    そもそも無いので、どの lag も等価に「それらしい」)。
    """
    from vspeech.lib.stream_vc import sola_offset

    rng = np.random.default_rng(7)
    # 実測のノイズフロア RMS 0.000298 * 32768 ≈ 9.8 int16 単位より更に小さい、
    # しかし 0 ではないレベル。
    tail = (rng.standard_normal(100) * 1.0).astype(np.float32)
    assert np.any(tail != 0.0)  # デジタル無音ではない
    region = (rng.standard_normal(300) * 1.0).astype(np.float32)
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_breaks_flat_ties_toward_center():
    """相関面が完全に平坦なとき index 0 ではなく中央(公称 lag)を選ぶ。"""
    from vspeech.lib.stream_vc import sola_offset

    # DC 一定なので全 lag の正規化相関が 1.0 = 完全な同点。
    tail = np.full(100, 5000.0, dtype=np.float32)
    region = np.full(300, 5000.0, dtype=np.float32)
    assert sola_offset(tail, region) == (300 - 100) // 2


def test_sola_offset_zero_when_region_too_short():
    from vspeech.lib.stream_vc import sola_offset

    # 窓が 1 つも取れないので中央は定義できない。0 だけが妥当な戻り値。
    tail = np.full(100, 5000.0, dtype=np.float32)
    assert sola_offset(tail, np.full(50, 5000.0, dtype=np.float32)) == 0
