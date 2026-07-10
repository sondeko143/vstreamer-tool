"""export ラッパの層インデックス算術を固定する。

実資産の layer_offset は 0 なので、`L9 + layer_offset` から `+ layer_offset` を落とす退行は
export の自己検証（golden 比較）でも捕まらない。両辺が一致してしまうからである。ここだけが
off-by-one を pin している。

scripts/export_hubert_onnx.py は transformers / safetensors を関数内で遅延 import するので、
それらが未インストールでもこのモジュールは import できる。ダミーの nn.Module を渡して検査する。
"""

from types import SimpleNamespace

import torch

HIDDEN_STATES = 14  # 0..13。9/12 に加えて offset=+1 の 10/13 も引けるように


class _StubModel(torch.nn.Module):
    """hidden_states[i] の全要素が i になるモデル。どの層を引いたか値で分かる。"""

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
    """`+ layer_offset` を落とす退行はこのテストだけが捕まえる。"""
    from scripts.export_hubert_onnx import HubertOnnxWrapper

    wrapper = HubertOnnxWrapper(_StubModel(), torch.nn.Identity(), layer_offset=1)
    out9, out12 = wrapper(torch.zeros(1, 8))
    assert out9[0, 0, 0].item() == 10.0  # 9 + 1
    assert out12[0, 0, 0].item() == 13.0  # 12 + 1
    # offset を無視する実装は 9.0 / 12.0 を返す。
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
    assert out12[0, 0, 0].item() == 12.0  # 生のまま
