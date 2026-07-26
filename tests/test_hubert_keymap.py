"""Tests for the fairseq -> transformers state_dict key translation (no torch needed)."""

import pytest

from scripts.hubert_keymap import build_key_map
from scripts.hubert_keymap import translate_key


@pytest.mark.parametrize(
    ("fairseq_key", "expected"),
    [
        (
            "feature_extractor.conv_layers.0.0.weight",
            "feature_extractor.conv_layers.0.conv.weight",
        ),
        (
            "feature_extractor.conv_layers.0.2.weight",
            "feature_extractor.conv_layers.0.layer_norm.weight",
        ),
        (
            "feature_extractor.conv_layers.4.0.weight",
            "feature_extractor.conv_layers.4.conv.weight",
        ),
        ("post_extract_proj.weight", "feature_projection.projection.weight"),
        ("post_extract_proj.bias", "feature_projection.projection.bias"),
        ("layer_norm.weight", "feature_projection.layer_norm.weight"),
        ("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias"),
        ("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.weight_g"),
        (
            "encoder.layers.3.self_attn.q_proj.weight",
            "encoder.layers.3.attention.q_proj.weight",
        ),
        (
            "encoder.layers.3.self_attn_layer_norm.bias",
            "encoder.layers.3.layer_norm.bias",
        ),
        (
            "encoder.layers.11.fc1.weight",
            "encoder.layers.11.feed_forward.intermediate_dense.weight",
        ),
        (
            "encoder.layers.11.fc2.bias",
            "encoder.layers.11.feed_forward.output_dense.bias",
        ),
        ("mask_emb", "masked_spec_embed"),
        # passed through (the same name on both sides)
        ("encoder.layer_norm.weight", "encoder.layer_norm.weight"),
        (
            "encoder.layers.7.final_layer_norm.weight",
            "encoder.layers.7.final_layer_norm.weight",
        ),
    ],
)
def test_translate_key(fairseq_key, expected):
    assert translate_key(fairseq_key) == expected


@pytest.mark.parametrize(
    "dropped", ["final_proj.weight", "final_proj.bias", "label_embs_concat"]
)
def test_translate_key_drops_non_encoder_params(dropped):
    assert translate_key(dropped) is None


def test_build_key_map_matches_and_drops():
    fairseq_keys = [
        "post_extract_proj.weight",
        "encoder.layer_norm.weight",
        "final_proj.weight",  # drop
        "label_embs_concat",  # drop
    ]
    hf_keys = ["feature_projection.projection.weight", "encoder.layer_norm.weight"]
    assert build_key_map(hf_keys, fairseq_keys) == {
        "feature_projection.projection.weight": "post_extract_proj.weight",
        "encoder.layer_norm.weight": "encoder.layer_norm.weight",
    }


def test_build_key_map_resolves_weight_norm_parametrization_alias():
    """On newer torch, weight_norm appears as parametrizations.*."""
    fairseq_keys = ["encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"]
    hf_keys = [
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ]
    assert build_key_map(hf_keys, fairseq_keys) == {
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1": "encoder.pos_conv.0.weight_v",
    }


def test_build_key_map_raises_when_a_transformers_param_is_unsourced():
    with pytest.raises(KeyError, match="encoder.layer_norm.weight"):
        build_key_map(["encoder.layer_norm.weight"], ["post_extract_proj.weight"])


def test_build_key_map_raises_when_two_fairseq_params_collide():
    """When two land on the same transformers parameter, the later one must not silently
    overwrite the earlier.

    The "no source" net cannot catch this: when a wrong rule's output happens to coincide
    with another legitimate key, the correct source is silently discarded and the weights
    are corrupted.
    """
    hf_keys = ["feature_projection.projection.weight"]
    fairseq_keys = [
        "post_extract_proj.weight",  # lands there after rule 3 translates it
        "feature_projection.projection.weight",  # lands on the same name by passthrough
    ]
    with pytest.raises(KeyError, match="same transformers param"):
        build_key_map(hf_keys, fairseq_keys)
