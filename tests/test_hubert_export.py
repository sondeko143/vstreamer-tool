"""Pin the layer-index arithmetic of the export wrapper.

The real assets have layer_offset 0, so a regression that drops `+ layer_offset` from
`L9 + layer_offset` is not caught even by the export's own self-verification (the golden
comparison), because both sides then agree. This is the only place that pins the
off-by-one.

scripts/export_hubert_onnx.py imports transformers / safetensors lazily inside its
functions, so this module imports even when they are not installed. The checks are done by
passing a dummy nn.Module.

torch itself is an offline-tool-only dependency now (ADR-0081 took it out of the
runtime); an environment without it skips this whole module rather than failing
collection.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# 0..13, so that 10/13 (offset=+1) can be indexed in addition to 9/12
HIDDEN_STATES = 14


class _StubModel(torch.nn.Module):
    """A model where every element of hidden_states[i] is i, so the value reveals which
    layer was indexed."""

    def forward(self, source, output_hidden_states=False):
        hidden = tuple(torch.full((1, 2, 4), float(i)) for i in range(HIDDEN_STATES))
        return SimpleNamespace(hidden_states=hidden)


def test_layer_indices_apply_the_offset():
    from scripts.export_hubert_onnx import layer_indices

    assert layer_indices(0) == (9, 12)
    assert layer_indices(1) == (10, 13)
    assert layer_indices(-1) == (8, 11)


def test_wrapper_indexes_hidden_states_with_a_zero_offset():
    from scripts.export_hubert_onnx import HubertOnnxWrapper

    wrapper = HubertOnnxWrapper(_StubModel(), torch.nn.Identity(), layer_offset=0)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 9.0
    assert out12[0, 0, 0].item() == 12.0


def test_wrapper_indexes_hidden_states_with_a_nonzero_offset():
    """Only this test catches a regression that drops `+ layer_offset`."""
    from scripts.export_hubert_onnx import HubertOnnxWrapper

    wrapper = HubertOnnxWrapper(_StubModel(), torch.nn.Identity(), layer_offset=1)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 10.0  # 9 + 1
    assert out12[0, 0, 0].item() == 13.0  # 12 + 1
    # An implementation that ignores the offset would return 9.0 / 12.0.
    assert out9[0, 0, 0].item() != 9.0
    assert out12[0, 0, 0].item() != 12.0


def test_wrapper_applies_final_proj_only_to_the_l9_output():
    from scripts.export_hubert_onnx import HubertOnnxWrapper

    class _Doubler(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    wrapper = HubertOnnxWrapper(_StubModel(), _Doubler(), layer_offset=0)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 18.0  # final_proj(hidden_states[9]) = 9 * 2
    assert out12[0, 0, 0].item() == 12.0  # raw, unchanged
