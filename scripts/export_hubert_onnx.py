"""transformers HubertModel 資産 (hubert_contentvec/) を ONNX へ export する。

**一度きり**のオフライン処理。runtime には含めない。依存 (transformers / onnx /
onnxscript) は poe task の `uv run --with` が一時環境で供給する。

    uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden

`python scripts/export_hubert_onnx.py` ではなく **`python -m scripts.export_hubert_onnx`**
で起動すること（前者は sys.path[0] が scripts/ になり `from scripts...` / `from vspeech...`
の import が解決しない）。

出力:
  <asset>/hubert_fp32.onnx    fp32 グラフ
  <asset>/hubert_fp16.onnx    CUDA 上で model.half() を export した fp16 グラフ
  <asset>/mapping.json        出力名 <-> (layer, use_final_proj) の対応表（上書き）
  <golden>/hubert_golden_fp16.npz  torch fp16 の出力（fp16 ゲートの参照）

ゲートの参照:
  fp32 グラフ -> <golden>/hubert_golden.npz（fairseq 由来の fp32 正解）
  fp16 グラフ -> torch fp16（置き換え対象の実装）。fp32 golden ではない。
                 半精度の絶対誤差は hidden state のスケールに対して 1e-1 オーダーで、
                 現行 runtime 自身が fp32 golden 比 cosine 0.987 / max_abs 0.435 を出す。

final_proj はグラフに焼き込む。したがって runtime は safetensors も
torch.nn.Linear も要らない。export の正しさはこのスクリプト自身がアサートし、
通らなければ資産を書き出さない。
"""

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import COSINE_MIN_FP16
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import MAX_ABS_MAX_FP16
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff
from vspeech.lib.rvc import FEATS_L9_PROJ
from vspeech.lib.rvc import FEATS_L12_RAW
from vspeech.lib.rvc import parse_output_names

# NOTE: transformers / safetensors / scripts.convert_hubert (transformers を引く) は
# **関数内で遅延 import する**。module 直下に置くと、spec ② でそれらを依存から外した後に
# このモジュールをテストから import できなくなり、layer_indices / HubertOnnxWrapper を
# 単体テストできなくなる。scripts/convert_hubert.py が fairseq に対して取っている手と同じ。

L9 = 9
L12 = 12
OPSET = 20

# golden npz のキー -> (ONNX 出力名, fairseq output_layer, use_final_proj)
GOLDEN_KEYS = {
    "l9_proj": (FEATS_L9_PROJ, L9, True),
    "l12_raw": (FEATS_L12_RAW, L12, False),
}


def layer_indices(layer_offset: int) -> tuple[int, int]:
    """fairseq の output_layer -> transformers hidden_states の添字。

    layer_offset は変換時に実測で確定して mapping.json に記録されている（実資産では 0）。
    """
    return L9 + layer_offset, L12 + layer_offset


class HubertOnnxWrapper(torch.nn.Module):
    """export 専用。runtime には入らない。

    final_proj をグラフに焼き込み、実在する 2 組合せだけを出力する。層インデックスは
    export 時に解決してグラフへ固定するので、runtime は推測しない。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        final_proj: torch.nn.Module,
        layer_offset: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.final_proj = final_proj
        self.l9, self.l12 = layer_indices(layer_offset)

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(source, output_hidden_states=True).hidden_states
        return self.final_proj(hidden_states[self.l9]), hidden_states[self.l12]


def fold_weight_norm(model: torch.nn.Module) -> None:
    """pos_conv の weight_norm パラメトリゼーションを畳み込む。

    parametrization が残ったままだと export されるグラフに余計な演算が乗り、
    exporter によっては失敗する。畳み込んでも数値は変わらない。
    """
    from torch.nn.utils import parametrize

    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )


def load_asset(asset_dir: Path) -> tuple[torch.nn.Module, torch.nn.Linear, int, int]:
    """(encoder, final_proj, layer_offset, num_hidden_layers) を返す。"""
    from safetensors.torch import load_file
    from transformers import HubertModel

    model = HubertModel.from_pretrained(  # nosec B615 - local dir only, no Hub download
        asset_dir, local_files_only=True
    )
    model.eval()

    tensors = load_file(str(asset_dir / "final_proj.safetensors"))
    weight, bias = tensors["weight"], tensors["bias"]
    final_proj = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        final_proj.weight.copy_(weight)
        final_proj.bias.copy_(bias)
    final_proj.eval()

    with open(asset_dir / "mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    layer_offset = int(mapping["layer_offset"])
    num_hidden_layers = int(mapping["num_hidden_layers"])

    fold_weight_norm(model)
    return model, final_proj, layer_offset, num_hidden_layers


def export_graph(wrapper: torch.nn.Module, source: torch.Tensor, path: Path) -> str:
    """ONNX を書き、使った exporter 名 ("dynamo" / "legacy") を返す。

    `external_data=False` で重みをグラフへ埋め込む。dynamo exporter の既定
    (`external_data=True`) は重みを `<path>.data` へ別出しし、`path` 自体は
    ポインタだけの小さなグラフになる。呼び出し側は `path` しか `shutil.move`
    しないので、既定のままだと `.data` は一時ディレクトリに取り残されたまま
    破棄され、移動後の資産は外部データが見つからず読み込めない
    （2026-07-10 に実測: `InferenceSession` が
    `External data path does not exist` で失敗するのを確認）。HuBERT の重みは
    fp32/fp16 とも ONNX の 2GB protobuf 上限に収まるので埋め込んで問題ない。
    """
    kwargs: dict[str, Any] = dict(
        input_names=["source"],
        output_names=[FEATS_L9_PROJ, FEATS_L12_RAW],
        dynamic_axes={
            "source": {1: "N"},
            FEATS_L9_PROJ: {1: "T"},
            FEATS_L12_RAW: {1: "T"},
        },
        opset_version=OPSET,
        external_data=False,
    )
    try:
        torch.onnx.export(wrapper, (source,), str(path), dynamo=True, **kwargs)
        return "dynamo"
    except Exception:  # exporter は多様な例外を投げるので広く捕まえる
        # **大声で報告すること。** 2026-07-10 にこの except が UnicodeEncodeError を飲み込み、
        # dynamo が成功できるのに黙って legacy へ落ちていた（torch.onnx が進捗の ✅ を
        # Windows の cp1252 stdout へ書こうとして落ちる）。main() の UTF-8 reconfigure が
        # その原因を潰すが、フォールバックが起きたときは必ず traceback を出す。
        print("!!! dynamo exporter failed; falling back to the legacy exporter !!!")
        traceback.print_exc()
        torch.onnx.export(wrapper, (source,), str(path), dynamo=False, **kwargs)
        return "legacy"


def run_session(path: Path, wav: np.ndarray, is_half: bool) -> dict[str, np.ndarray]:
    from onnxruntime import InferenceSession

    providers = ["CUDAExecutionProvider"] if is_half else ["CPUExecutionProvider"]
    session = InferenceSession(str(path), providers=providers)
    source = wav.astype(np.float16 if is_half else np.float32)[None, :]
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(names, {"source": source})
    return {name: np.asarray(out) for name, out in zip(names, outputs)}


def torch_fp16_reference(
    half_wrapper: torch.nn.Module, source: torch.Tensor
) -> dict[str, np.ndarray]:
    """置き換え対象である `HubertModel.half()` の出力（fp16 ゲートの参照）。

    fp32 golden を fp16 の参照にはできない。半精度の絶対誤差は hidden state のスケール
    (O(1)-O(2.5)) に対して 1e-1 オーダーになり、現行 runtime 自身が fp32 golden 比で
    cosine 0.987 / max_abs 0.435 を出す。問うべきは「ONNX 化で fp16 の振る舞いが
    変わっていないか」であり、参照は置き換え対象の torch fp16 である。

    GPU / カーネル依存の参照。テストは CUDA gating 済みなので開発機でのみ意味を持つ。

    **呼び出し順序が load-bearing**: `.half()` はモジュールを in-place で書き換えるので、
    fp32 グラフの export を済ませてから半精度化すること。半精度化した後に `.float()` で
    戻しても fp32 の重みは復元しない。ここでは既に半精度化済みのラッパをそのまま呼び、
    ONNX fp16 と厳密に同じ重み・同じ層から参照を取る。
    """
    with torch.inference_mode():
        out9, out12 = half_wrapper(source)
    return {
        "l9_proj": out9.squeeze(0).float().cpu().numpy(),
        "l12_raw": out12.squeeze(0).float().cpu().numpy(),
    }


def check(
    outputs: dict[str, np.ndarray],
    golden: dict[str, np.ndarray],
    label: str,
    cosine_min: float,
    max_abs_max: float,
) -> bool:
    ok = True
    for golden_key, (output_name, _, _) in GOLDEN_KEYS.items():
        reference = golden[golden_key].astype(np.float32)
        candidate = outputs[output_name].squeeze(0).astype(np.float32)
        if candidate.shape != reference.shape:
            print(f"{label} {golden_key}: shape {candidate.shape} != {reference.shape}")
            ok = False
            continue
        cosine = feature_cosine(candidate, reference)
        max_abs = feature_max_abs_diff(candidate, reference)
        verdict = "OK" if (cosine >= cosine_min and max_abs <= max_abs_max) else "FAIL"
        print(
            f"{label} {golden_key}: cosine={cosine:.8f} max_abs={max_abs:.3e} [{verdict}]"
        )
        ok = ok and verdict == "OK"
    return ok


def main() -> None:
    # torch.onnx の進捗表示は ✅ を含む。Windows の既定 stdout (cp1252) では
    # UnicodeEncodeError になり、export_graph の except がそれを「dynamo 失敗」と
    # 誤認して黙って legacy へ落ちる。ここで潰しておく。
    # typeshed types sys.stdout/stderr as TextIO, which lacks .reconfigure(); at
    # runtime CPython gives TextIOWrapper, which has it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path, help="hubert_contentvec/")
    parser.add_argument("--golden", required=True, type=Path, help="hubert_golden/")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="一時ディレクトリへ export して誤差を印字するだけ。資産は更新しない",
    )
    args = parser.parse_args()

    asset_dir = args.asset.expanduser()
    golden_dir = args.golden.expanduser()
    golden = dict(np.load(golden_dir / "hubert_golden.npz"))
    wav = golden["wav"].astype(np.float32)

    if not torch.cuda.is_available():
        raise SystemExit("fp16 export には CUDA が要ります")

    model, final_proj, layer_offset, num_hidden_layers = load_asset(asset_dir)
    print(f"layer_offset={layer_offset}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        fp32_path = tmp_dir / "hubert_fp32.onnx"
        fp16_path = tmp_dir / "hubert_fp16.onnx"

        # fp32 を先に出す。次の `.half()` はモジュールを in-place で壊す。
        wrapper = HubertOnnxWrapper(model, final_proj, layer_offset).eval()
        source = torch.from_numpy(wav).unsqueeze(0)
        exporter = export_graph(wrapper, source, fp32_path)
        print(f"exported fp32 with {exporter} exporter")

        ok = check(
            run_session(fp32_path, wav, is_half=False),
            golden,
            "fp32",
            COSINE_MIN,
            MAX_ABS_MAX,
        )

        half_wrapper = (
            HubertOnnxWrapper(model, final_proj, layer_offset).eval().half().cuda()
        )
        half_source = source.half().cuda()
        # fp16 ゲートの参照。ONNX fp16 と同じ重み・同じ層から取る。
        reference = torch_fp16_reference(half_wrapper, half_source)
        export_graph(half_wrapper, half_source, fp16_path)
        ok = (
            check(
                run_session(fp16_path, wav, is_half=True),
                reference,
                "fp16",
                COSINE_MIN_FP16,
                MAX_ABS_MAX_FP16,
            )
            and ok
        )

        if args.measure_only:
            print("--measure-only: 資産は更新していません")
            return
        if not ok:
            raise SystemExit("等価ゲートに落ちました。資産は書き出しません。")

        shutil.move(str(fp32_path), asset_dir / "hubert_fp32.onnx")
        shutil.move(str(fp16_path), asset_dir / "hubert_fp16.onnx")

    # fp16 ゲートの参照を golden 側へ保存する。テストは npz を読むだけで transformers を
    # 要らない（Task 8 でプロジェクト依存から外れるため、ここでしか捕獲できない）。
    # numpy 2 の savez スタブは allow_pickle:bool を持ち、**reference 展開と衝突する型誤検知。実行時は正しい。
    np.savez(golden_dir / "hubert_golden_fp16.npz", wav=wav, **reference)  # ty: ignore[invalid-argument-type]

    mapping = {
        "layer_offset": layer_offset,
        "num_hidden_layers": num_hidden_layers,
        "exporter": exporter,
        "opset": OPSET,
        "outputs": [
            {"name": FEATS_L9_PROJ, "layer": L9, "use_final_proj": True, "dim": 256},
            {"name": FEATS_L12_RAW, "layer": L12, "use_final_proj": False, "dim": 768},
        ],
    }
    parse_output_names(mapping)  # runtime が読める形であることをここで保証する
    with open(asset_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"wrote onnx + mapping.json -> {asset_dir}")


if __name__ == "__main__":
    main()
