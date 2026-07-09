"""transformers ベース HuBERT runtime の単体テスト。

実物の hubert_base.pt を使わず、ごく小さな HubertConfig でランダム初期化した
資産をその場で作って検証する。層選択・final_proj 適用・エラー経路を固定する。
"""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import HubertConfig
from transformers import HubertModel

HIDDEN = 32
PROJ_OUT = 8
NUM_LAYERS = 4


def _tiny_config() -> HubertConfig:
    return HubertConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=(HIDDEN, HIDDEN),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
    )


@pytest.fixture
def asset_dir(tmp_path):
    """Task 3 の変換ツールが書き出すのと同じレイアウトの合成資産。"""
    torch.manual_seed(0)
    model = HubertModel(_tiny_config())
    model.eval()
    model.save_pretrained(tmp_path)
    save_file(
        {
            "weight": torch.randn(PROJ_OUT, HIDDEN).contiguous(),
            "bias": torch.randn(PROJ_OUT).contiguous(),
        },
        str(tmp_path / "final_proj.safetensors"),
    )
    with open(tmp_path / "mapping.json", "w", encoding="utf-8") as f:
        json.dump({"layer_offset": 0, "num_hidden_layers": NUM_LAYERS}, f)
    return tmp_path


def _wav() -> torch.Tensor:
    t = np.arange(4000, dtype=np.float32) / 16000.0
    return torch.from_numpy(np.sin(2 * np.pi * 220.0 * t).astype(np.float32)).unsqueeze(
        0
    )


def test_load_hubert_model_returns_bundle(asset_dir):
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    assert bundle.final_proj is not None
    assert bundle.layer_offset == 0


def test_extract_features_applies_final_proj(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(
        bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=True
    )
    assert out.shape[0] == 1
    assert out.shape[2] == PROJ_OUT


def test_extract_features_without_final_proj_returns_hidden(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(
        bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=False
    )
    assert out.shape[2] == HIDDEN


def test_extract_features_selects_the_requested_layer(asset_dir):
    """layer_offset=0 なら hidden_states[N] がそのまま返ること。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    wav = _wav()
    with torch.inference_mode():
        expected = bundle.model(
            input_values=wav, output_hidden_states=True
        ).hidden_states[2]
    out = extract_features(
        bundle, wav, torch.device("cpu"), emb_output_layer=2, use_final_proj=False
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_extract_features_raises_when_final_proj_missing(asset_dir):
    """useFinalProj=True を要求されたのに資産に final_proj が無い場合。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    (asset_dir / "final_proj.safetensors").unlink()
    bundle = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    assert bundle.final_proj is None
    with pytest.raises(RuntimeError, match="final_proj"):
        extract_features(
            bundle, _wav(), torch.device("cpu"), emb_output_layer=2, use_final_proj=True
        )
