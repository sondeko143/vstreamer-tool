import os
from pathlib import Path

import numpy as np
import pytest

from vspeech.lib.stream_vc import next_context
from vspeech.lib.stream_vc import slice_block_output


def test_next_context_returns_tail():
    seq = np.arange(5)
    assert list(next_context(seq, 2)) == [3, 4]


def test_next_context_zero_is_empty():
    seq = np.arange(3)
    assert len(next_context(seq, 0)) == 0


def test_slice_block_output_takes_block_tail():
    out = np.arange(10)
    # block is last block_len/seq_len = 2/10 of the sequence -> last 2 output samples
    assert list(slice_block_output(out, block_len=2, seq_len=10)) == [8, 9]


def test_slice_block_output_rounds_proportionally():
    out = np.arange(100)
    # 40 / (200+40) of 100 -> round(16.67) = 17
    assert len(slice_block_output(out, block_len=40, seq_len=240)) == 17


def test_next_context_clamps_when_longer_than_seq():
    # context_len > len(seq): return the whole buffer (not a negative-index slice)
    seq = np.arange(3)
    assert list(next_context(seq, 5)) == [0, 1, 2]


def test_slice_block_output_block_len_zero_returns_all():
    out = np.arange(10)
    assert list(slice_block_output(out, block_len=0, seq_len=10)) == list(range(10))


def test_slice_block_output_clamps_when_block_exceeds_seq():
    # block_len > seq_len: clamp to the whole output, not a negative-index slice
    out = np.arange(100)
    assert list(slice_block_output(out, block_len=300, seq_len=240)) == list(range(100))


def test_helpers_work_on_torch_tensors():
    # docstring claims numpy/torch agnosticism; verify the torch path.
    import pytest

    torch = pytest.importorskip("torch")
    seq = torch.arange(5)
    assert next_context(seq, 2).tolist() == [3, 4]
    assert next_context(seq, 0).numel() == 0
    out = torch.arange(10)
    assert slice_block_output(out, block_len=2, seq_len=10).tolist() == [8, 9]


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
