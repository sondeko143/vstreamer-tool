"""Pin the layer-index arithmetic of the export wrapper, and its exit-code contract.

The real assets have layer_offset 0, so a regression that drops `+ layer_offset` from
`L9 + layer_offset` is not caught even by the export's own self-verification (the golden
comparison), because both sides then agree. This is the only place that pins the
off-by-one.

The second concern is `should_write_assets`: whether the run is judged before
`--measure-only` is honoured. Nothing else would catch that inversion, because it only
shows up on a *failing* export, which no green run produces.

The two halves are gated differently on purpose:

- The exit-code contract needs no torch, and it guards a defect that **already shipped
  once**. It is imported from scripts/hubert_metrics.py (which is torch-free) and runs in
  the ordinary suite. Gating it behind torch is what made this whole module collect zero
  tests once ADR-0080 took torch out of the dependency table.
- The wrapper tests genuinely need torch: `HubertOnnxWrapper` subclasses `torch.nn.Module`
  and scripts/export_hubert_onnx.py therefore imports torch at module level. They skip
  when torch is absent, which is the normal state of this project's environment -- torch
  comes only from the export task's `uv run --with` overlay.

scripts/export_hubert_onnx.py imports transformers / safetensors lazily inside its
functions, so those do not have to be installed for the wrapper tests either. The checks
are done by passing a dummy nn.Module.
"""

from types import SimpleNamespace

import pytest

from scripts.hubert_metrics import should_write_assets

requires_torch = pytest.mark.requires_torch

# 0..13, so that 10/13 (offset=+1) can be indexed in addition to 9/12
HIDDEN_STATES = 14


def _stub_model():
    """A model where every element of hidden_states[i] is i, so the value reveals which
    layer was indexed.

    Built inside a function rather than declared at module level: the class statement
    needs `torch.nn.Module` at definition time, and a module-level one would take the
    whole file down with it in a torch-free environment -- exactly the failure this file
    was rewritten to fix.
    """
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)

    class _StubModel(torch.nn.Module):
        def forward(self, source, output_hidden_states=False):
            hidden = tuple(
                torch.full((1, 2, 4), float(i)) for i in range(HIDDEN_STATES)
            )
            return SimpleNamespace(hidden_states=hidden)

    return _StubModel()


@requires_torch
def test_layer_indices_apply_the_offset():
    from scripts.export_hubert_onnx import layer_indices

    assert layer_indices(0) == (9, 12)
    assert layer_indices(1) == (10, 13)
    assert layer_indices(-1) == (8, 11)


@requires_torch
def test_wrapper_indexes_hidden_states_with_a_zero_offset():
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)

    from scripts.export_hubert_onnx import HubertOnnxWrapper

    wrapper = HubertOnnxWrapper(_stub_model(), torch.nn.Identity(), layer_offset=0)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 9.0
    assert out12[0, 0, 0].item() == 12.0


@requires_torch
def test_wrapper_indexes_hidden_states_with_a_nonzero_offset():
    """Only this test catches a regression that drops `+ layer_offset`."""
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)

    from scripts.export_hubert_onnx import HubertOnnxWrapper

    wrapper = HubertOnnxWrapper(_stub_model(), torch.nn.Identity(), layer_offset=1)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 10.0  # 9 + 1
    assert out12[0, 0, 0].item() == 13.0  # 12 + 1
    # An implementation that ignores the offset would return 9.0 / 12.0.
    assert out9[0, 0, 0].item() != 9.0
    assert out12[0, 0, 0].item() != 12.0


@requires_torch
def test_wrapper_applies_final_proj_only_to_the_l9_output():
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)

    from scripts.export_hubert_onnx import HubertOnnxWrapper

    class _Doubler(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    wrapper = HubertOnnxWrapper(_stub_model(), _Doubler(), layer_offset=0)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 18.0  # final_proj(hidden_states[9]) = 9 * 2
    assert out12[0, 0, 0].item() == 12.0  # raw, unchanged


def test_a_failed_gate_exits_non_zero_in_both_modes():
    """The exit code is the verdict, `--measure-only` included.

    The previous order honoured --measure-only first and returned, so an export whose
    equivalence gate had just printed FAIL still exited 0 -- and CLAUDE.md points people
    at exactly that mode.
    """
    for measure_only in (True, False):
        with pytest.raises(SystemExit) as exc:
            should_write_assets(False, measure_only=measure_only)
        assert exc.value.code != 0


def test_a_passing_gate_writes_only_when_not_measure_only():
    assert should_write_assets(True, measure_only=False) is True
    assert should_write_assets(True, measure_only=True) is False
