"""Map fairseq HuBERT/ContentVec state_dict keys onto transformers HubertModel names.

Structurally strict: `build_key_map` raises if even one parameter on the transformers side
has no source. This closes the path by which a silently mis-converted encoder could reach
the runtime.
"""

import re
from collections.abc import Iterable

# fairseq -> transformers. Applied top to bottom; only the first matching rule is used.
_RULES: tuple[tuple[str, str], ...] = (
    (
        r"^feature_extractor\.conv_layers\.(\d+)\.0\.",
        r"feature_extractor.conv_layers.\1.conv.",
    ),
    (
        r"^feature_extractor\.conv_layers\.(\d+)\.2\.",
        r"feature_extractor.conv_layers.\1.layer_norm.",
    ),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^encoder\.pos_conv\.0\.", "encoder.pos_conv_embed.conv."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (
        r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.",
        r"encoder.layers.\1.layer_norm.",
    ),
    (
        r"^encoder\.layers\.(\d+)\.fc1\.",
        r"encoder.layers.\1.feed_forward.intermediate_dense.",
    ),
    (
        r"^encoder\.layers\.(\d+)\.fc2\.",
        r"encoder.layers.\1.feed_forward.output_dense.",
    ),
    (r"^mask_emb$", "masked_spec_embed"),
)

# fairseq parameters with no counterpart in transformers' HubertModel.
# final_proj is extracted as a separate tensor, so it is dropped here.
# label_embs_concat is HuBERT's pre-training label embedding and is unused at inference.
DROPPED_PREFIXES: tuple[str, ...] = ("final_proj.", "label_embs_concat")

# On torch>=2.1, weight_norm appears in the state_dict as parametrizations.*.
_WEIGHT_NORM_ALIASES: dict[str, str] = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def translate_key(fairseq_key: str) -> str | None:
    """Translate a fairseq key name into a transformers key name.

    `None` means "a key we deliberately drop (DROPPED_PREFIXES)", not "no rule matched".
    Keys that match no rule are passed through unchanged (some keys have the same name on
    both sides).
    """
    if fairseq_key.startswith(DROPPED_PREFIXES):
        return None
    for pattern, replacement in _RULES:
        translated, hits = re.subn(pattern, replacement, fairseq_key)
        if hits:
            return translated
    # encoder.layer_norm.* and encoder.layers.N.final_layer_norm.* have the same name on
    # both sides.
    return fairseq_key


def build_key_map(
    hf_keys: Iterable[str], fairseq_keys: Iterable[str]
) -> dict[str, str]:
    """Return {transformers_key: fairseq_key}.

    The fail-closed net is cast twice:
      - Unfilled: KeyError if any parameter on the transformers side has no source at all.
      - Collision: KeyError if two fairseq parameters land on the same transformers
        parameter. A plain dict assignment would silently let the last one win. The
        "no source" net cannot catch this (the case where a wrong rule's output happens to
        coincide with another legitimate key), so it is detected separately.
    """
    hf_key_set = set(hf_keys)
    mapping: dict[str, str] = {}
    for fairseq_key in fairseq_keys:
        translated = translate_key(fairseq_key)
        if translated is None:
            continue
        if translated not in hf_key_set:
            translated = _WEIGHT_NORM_ALIASES.get(translated, translated)
        if translated in hf_key_set:
            if translated in mapping:
                raise KeyError(
                    f"two fairseq params map to the same transformers param "
                    f"{translated!r}: {mapping[translated]!r} and {fairseq_key!r}"
                )
            mapping[translated] = fairseq_key
    unsourced = sorted(hf_key_set - set(mapping))
    if unsourced:
        raise KeyError(f"no fairseq source for transformers params: {unsourced}")
    return mapping
