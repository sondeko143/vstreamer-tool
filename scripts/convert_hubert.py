"""fairseq ContentVec (hubert_base.pt) を transformers HubertModel 資産へ変換する。

Python 3.11 + fairseq で **一度だけ** 走らせるオフライン処理。runtime には含めない。
fairseq / transformers は pyproject にも uv.lock にも無い。poe task が `uv run --with` で
使い捨ての 3.11 環境に供給する（`--python 3.11` は必須。省くとプロジェクトの処理系に落ち、
cp311 の fairseq wheel が入らない）。

    uv run poe convert-hubert \
        --input  <path>/hubert_base.pt \
        --output <path>/hubert_contentvec \
        --golden <path>/hubert_golden

poe task は `python -m scripts.convert_hubert` の形で起動する。ファイルパスで叩くと
sys.path[0] が scripts/ になり `from scripts.hubert_metrics import ...` を解決できない。

このあと `uv run poe export-hubert-onnx` で ONNX を書き出すまで、runtime は資産を読めない。

出力:
  <output>/config.json, model.safetensors  transformers HubertModel の encoder
  <output>/final_proj.safetensors          fairseq の final_proj (768->256)
  <output>/mapping.json                    fairseq output_layer -> hidden_states の対応
  <golden>/hubert_golden.npz               fairseq 側の正解特徴量（fp32）

変換の正しさはこのスクリプト自身がアサートする。通らなければ資産を書かない。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import HubertConfig
from transformers import HubertModel

from scripts.hubert_keymap import build_key_map
from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

# しきい値は scripts/hubert_metrics.py が単一情報源。ここで再定義しないこと。
# 変換時点でこれを満たさなければ資産を書き出さない。

# 検証・golden 捕獲に使う代表音声（決定論的。RNG を使わない）。
GOLDEN_SAMPLE_RATE = 16000
GOLDEN_SECONDS = 1.0


def make_fixed_audio() -> np.ndarray:
    """220Hz + 440Hz の決定論的な mono float32 波形（[-1, 1]）。"""
    n = int(GOLDEN_SAMPLE_RATE * GOLDEN_SECONDS)
    t = np.arange(n, dtype=np.float64) / GOLDEN_SAMPLE_RATE
    wave = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    return wave.astype(np.float32)


def load_fairseq_model(checkpoint: Path):
    """fairseq 側の ContentVec を eval モードで読む（現行 rvc.py と同一手順）。

    saved_cfg も返す。HubertConfig の一部（活性化関数名）はモジュール属性からは
    綺麗に取れないため、チェックポイントが保持している設定値から読む。
    """
    import fairseq.data.dictionary
    from fairseq import checkpoint_utils

    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
    models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint.expanduser())], suffix=""
    )
    model = models[0]
    model.eval()
    return model, saved_cfg


def derive_hf_config(fs_model, saved_cfg) -> HubertConfig:
    """fairseq モデルの実属性 / 保存済み設定から HubertConfig を組む（ハードコードしない）。"""
    conv_dim: list[int] = []
    conv_kernel: list[int] = []
    conv_stride: list[int] = []
    for conv_layer in fs_model.feature_extractor.conv_layers:
        conv = conv_layer[0]
        conv_dim.append(int(conv.out_channels))
        conv_kernel.append(int(conv.kernel_size[0]))
        conv_stride.append(int(conv.stride[0]))

    first_layer = fs_model.feature_extractor.conv_layers[0]
    has_group_norm = any(isinstance(m, torch.nn.GroupNorm) for m in first_layer)
    conv_bias = first_layer[0].bias is not None

    pos_conv = fs_model.encoder.pos_conv[0]
    first_block = fs_model.encoder.layers[0]

    return HubertConfig(
        hidden_size=int(fs_model.encoder.embedding_dim),
        num_hidden_layers=len(fs_model.encoder.layers),
        num_attention_heads=int(first_block.self_attn.num_heads),
        intermediate_size=int(first_block.fc1.out_features),
        hidden_act=str(saved_cfg.model.activation_fn),
        conv_dim=tuple(conv_dim),
        conv_kernel=tuple(conv_kernel),
        conv_stride=tuple(conv_stride),
        conv_bias=conv_bias,
        feat_extract_norm="group" if has_group_norm else "layer",
        feat_extract_activation="gelu",
        do_stable_layer_norm=bool(getattr(fs_model.encoder, "layer_norm_first", False)),
        num_conv_pos_embeddings=int(pos_conv.kernel_size[0]),
        num_conv_pos_embedding_groups=int(pos_conv.groups),
        # 推論専用。dropout は全て 0 にして eval と一致させる。
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
    )


def convert_encoder(fs_model, hf_config: HubertConfig) -> HubertModel:
    """fairseq の重みを transformers HubertModel へ strict に流し込む。"""
    hf_model = HubertModel(hf_config)
    hf_model.eval()

    fairseq_sd = fs_model.state_dict()
    key_map = build_key_map(hf_model.state_dict().keys(), fairseq_sd.keys())
    hf_model.load_state_dict(
        {hf: fairseq_sd[fs] for hf, fs in key_map.items()}, strict=True
    )
    return hf_model


def fairseq_features(
    fs_model, source: torch.Tensor, layer: int, use_final_proj: bool
) -> np.ndarray:
    """現行 rvc.py の extract_features と同一の呼び出しで fp32 特徴量を得る。"""
    padding_mask = torch.zeros(source.shape, dtype=torch.bool)
    with torch.inference_mode():
        logits = fs_model.extract_features(
            source=source, padding_mask=padding_mask, output_layer=layer
        )
        feats = fs_model.final_proj(logits[0]) if use_final_proj else logits[0]
    return feats.squeeze(0).float().cpu().numpy()


def hf_hidden_states(
    hf_model: HubertModel, source: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        # fairseq の padding_mask (True=パディング) と transformers の attention_mask
        # (1=有効) は意味が反転している。パディング無しなので何も渡さない = 全有効。
        outputs = hf_model(input_values=source, output_hidden_states=True)
    return outputs.hidden_states


def resolve_layer_offset(fs_model, hf_model: HubertModel, source: torch.Tensor) -> int:
    """fairseq output_layer=N が transformers hidden_states[N + offset] に対応する offset。

    候補を総当たりして誤差最小を選び、複数の層で一致することを要求する（off-by-one 対策）。
    """
    hidden_states = hf_hidden_states(hf_model, source)
    resolved: set[int] = set()
    for layer in (9, 12):
        reference = fairseq_features(fs_model, source, layer, use_final_proj=False)
        best_offset = None
        best_diff = float("inf")
        for offset in (-1, 0, 1):
            index = layer + offset
            if not 0 <= index < len(hidden_states):
                continue
            candidate = hidden_states[index].squeeze(0).float().cpu().numpy()
            if candidate.shape != reference.shape:
                continue
            diff = feature_max_abs_diff(candidate, reference)
            if diff < best_diff:
                best_diff = diff
                best_offset = offset
        if best_offset is None or best_diff > MAX_ABS_MAX:
            raise SystemExit(
                f"layer {layer}: no transformers hidden_states index matches fairseq "
                f"(best max-abs-diff={best_diff:.3e} > {MAX_ABS_MAX:.1e}). 変換が壊れている。"
            )
        resolved.add(best_offset)
    if len(resolved) != 1:
        raise SystemExit(f"layer offset が層ごとに矛盾している: {sorted(resolved)}")
    return resolved.pop()


def verify(
    fs_model, hf_model: HubertModel, source: torch.Tensor, layer_offset: int
) -> dict[str, np.ndarray]:
    """(9,True) と (12,False) で等価をアサートし、fairseq 側の golden を返す。"""
    final_proj = fs_model.final_proj
    hidden_states = hf_hidden_states(hf_model, source)
    golden: dict[str, np.ndarray] = {}

    for layer, use_final_proj, key in ((9, True, "l9_proj"), (12, False, "l12_raw")):
        reference = fairseq_features(fs_model, source, layer, use_final_proj)
        hidden = hidden_states[layer + layer_offset]
        with torch.inference_mode():
            candidate_t = final_proj(hidden) if use_final_proj else hidden
        candidate = candidate_t.squeeze(0).float().cpu().numpy()

        cosine = feature_cosine(candidate, reference)
        max_abs = feature_max_abs_diff(candidate, reference)
        print(
            f"{key}: cosine={cosine:.8f} max_abs={max_abs:.3e} shape={reference.shape}"
        )
        if cosine < COSINE_MIN or max_abs > MAX_ABS_MAX:
            raise SystemExit(
                f"{key}: equivalence FAILED (cosine={cosine:.8f} < {COSINE_MIN} "
                f"or max_abs={max_abs:.3e} > {MAX_ABS_MAX:.1e})。資産は書き出さない。"
            )
        golden[key] = reference.astype(np.float32)
    return golden


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fairseq ContentVec (hubert_base.pt) を transformers HubertModel 資産へ変換する (offline, 1/2 段目)。",
        epilog=(
            "HuBERT (ContentVec) 資産は 2 段のオフライン変換で用意する。\n"
            "入力 hubert_base.pt は RVC が配布する ContentVec (MIT, origin auspicious3000/contentvec)。\n"
            "\n"
            "手順:\n"
            "  1. uv run poe convert-hubert \\\n"
            "         --input  ~/.config/vstreamer/hubert_base.pt \\\n"
            "         --output ./hubert_contentvec \\\n"
            "         --golden ./hubert_golden\n"
            "  2. uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden\n"
            "  3. config の [rvc] に設定:\n"
            '       hubert_model_file = "./hubert_contentvec"   # 資産ディレクトリ (ファイルではない)\n'
            "\n"
            "ライセンスは THIRD_PARTY_NOTICES.md を参照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="hubert_base.pt")
    parser.add_argument(
        "--output", required=True, type=Path, help="変換済み資産ディレクトリ"
    )
    parser.add_argument(
        "--golden", required=True, type=Path, help="golden 出力ディレクトリ"
    )
    args = parser.parse_args()

    fs_model, saved_cfg = load_fairseq_model(args.input)
    hf_config = derive_hf_config(fs_model, saved_cfg)
    hf_model = convert_encoder(fs_model, hf_config)

    wav = make_fixed_audio()
    source = torch.from_numpy(wav).unsqueeze(0)

    layer_offset = resolve_layer_offset(fs_model, hf_model, source)
    print(f"resolved layer_offset={layer_offset}")

    golden = verify(fs_model, hf_model, source, layer_offset)

    args.output.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.output)
    save_file(
        {
            "weight": fs_model.final_proj.weight.detach().cpu().contiguous(),
            "bias": fs_model.final_proj.bias.detach().cpu().contiguous(),
        },
        str(args.output / "final_proj.safetensors"),
    )
    with open(args.output / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "layer_offset": layer_offset,
                "num_hidden_layers": hf_config.num_hidden_layers,
            },
            f,
            indent=2,
        )

    args.golden.mkdir(parents=True, exist_ok=True)
    # numpy 2 の savez スタブは allow_pickle:bool を持ち、**golden(ndarray dict)展開と衝突する型誤検知。実行時は正しい。
    np.savez(args.golden / "hubert_golden.npz", wav=wav, **golden)  # ty: ignore[invalid-argument-type]

    print(f"wrote asset -> {args.output}")
    print(f"wrote golden -> {args.golden / 'hubert_golden.npz'}")


if __name__ == "__main__":
    main()
