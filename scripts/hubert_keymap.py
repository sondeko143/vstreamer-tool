"""fairseq HuBERT/ContentVec の state_dict キーを transformers HubertModel 名へ写す。

構造的に strict: `build_key_map` は transformers 側のパラメータに供給元が
1 つでも無ければ例外を投げる。黙って誤変換された encoder が runtime に届く経路を塞ぐ。
"""

import re
from collections.abc import Iterable

# fairseq -> transformers。先頭から順に最初にマッチした 1 本だけを適用する。
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

# transformers HubertModel に対応物が無い fairseq パラメータ。
# final_proj は別テンソルとして抽出するのでここでは捨てる。
# label_embs_concat は HuBERT の事前学習用ラベル埋め込みで、推論では使わない。
DROPPED_PREFIXES: tuple[str, ...] = ("final_proj.", "label_embs_concat")

# torch>=2.1 の weight_norm は parametrizations.* として state_dict に現れる。
_WEIGHT_NORM_ALIASES: dict[str, str] = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def translate_key(fairseq_key: str) -> str | None:
    """fairseq のキー名を transformers のキー名へ。

    `None` は「意図的に捨てるキー（DROPPED_PREFIXES）」の意味であって、「規則に一致しなかった」
    の意味ではない。規則に一致しないキーはそのまま素通しする（両者で同名のキーがあるため）。
    """
    if fairseq_key.startswith(DROPPED_PREFIXES):
        return None
    for pattern, replacement in _RULES:
        translated, hits = re.subn(pattern, replacement, fairseq_key)
        if hits:
            return translated
    # encoder.layer_norm.* / encoder.layers.N.final_layer_norm.* は両者で同名。
    return fairseq_key


def build_key_map(
    hf_keys: Iterable[str], fairseq_keys: Iterable[str]
) -> dict[str, str]:
    """{transformers_key: fairseq_key} を返す。

    fail-closed の網は二重に張る:
      - 未充足: transformers 側のパラメータに供給元が 1 つも無ければ KeyError。
      - 衝突: 2 つの fairseq パラメータが同じ transformers パラメータに着地したら KeyError。
        素の dict 代入だと後勝ちで黙って上書きされる。「供給元が無い」網はこれを捕まえられない
        （間違った規則の出力が別の正当なキーと偶然一致するケース）ので、専用に検出する。
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
