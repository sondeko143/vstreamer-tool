"""Export the transformers HubertModel assets (hubert_contentvec/) to ONNX.

A **one-shot** offline step, not part of the runtime. Its dependencies (transformers /
onnx / onnxscript) are supplied in a temporary environment by the poe task's
`uv run --with`.

    uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden

Launch it as **`python -m scripts.export_hubert_onnx`**, not
`python scripts/export_hubert_onnx.py` (the latter makes sys.path[0] be scripts/, where
the `from scripts...` / `from vspeech...` imports do not resolve).

Outputs:
  <asset>/hubert_fp32.onnx    the fp32 graph
  <asset>/hubert_fp16.onnx    the fp16 graph, exported from model.half() on CUDA
  <asset>/mapping.json        output name <-> (layer, use_final_proj) table (overwritten)
  <golden>/hubert_golden_fp16.npz  the torch fp16 output (the fp16 gate's reference)

The gates' references:
  fp32 graph -> <golden>/hubert_golden.npz (the fp32 reference derived from fairseq)
  fp16 graph -> torch fp16 (the implementation being replaced), not the fp32 golden.
                Half precision's absolute error is on the order of 1e-1 relative to the
                scale of the hidden states, and the current runtime itself scores
                cosine 0.987 / max_abs 0.435 against the fp32 golden.

final_proj is baked into the graph, so the runtime needs neither safetensors nor
torch.nn.Linear. This script asserts the correctness of the export itself; if the
assertions fail, no assets are written **and the process exits non-zero** -- including
under `--measure-only`, whose job is to suppress the writes, not the verdict.
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
from scripts.hubert_metrics import should_write_assets
from vspeech.lib.rvc import FEATS_L9_PROJ
from vspeech.lib.rvc import FEATS_L12_RAW
from vspeech.lib.rvc import parse_output_names

# NOTE: transformers / safetensors / scripts.convert_hubert (which pulls in transformers)
# are **imported lazily inside the functions**. They are not dependencies, so putting them
# at module level would make this module unimportable from the tests and leave
# layer_indices / HubertOnnxWrapper untestable. The same move scripts/convert_hubert.py
# makes for fairseq.
#
# torch cannot get the same treatment: HubertOnnxWrapper's base class needs it at class
# definition time. So this module IS unimportable without torch, which since ADR-0080 is
# every environment that is not the export overlay. That is why the exit-code contract
# (`should_write_assets`) lives in scripts/hubert_metrics.py instead of here -- the tests
# that pin it must run in the ordinary suite.

L9 = 9
L12 = 12
OPSET = 20

# golden npz key -> (ONNX output name, fairseq output_layer, use_final_proj)
GOLDEN_KEYS = {
    "l9_proj": (FEATS_L9_PROJ, L9, True),
    "l12_raw": (FEATS_L12_RAW, L12, False),
}


def layer_indices(layer_offset: int) -> tuple[int, int]:
    """fairseq's output_layer -> the index into transformers' hidden_states.

    layer_offset was determined by measurement at conversion time and recorded in
    mapping.json (0 for the real assets).
    """
    return L9 + layer_offset, L12 + layer_offset


class HubertOnnxWrapper(torch.nn.Module):
    """Export only; never part of the runtime.

    It bakes final_proj into the graph and emits only the two combinations that really
    exist. The layer indices are resolved at export time and fixed into the graph, so the
    runtime never guesses.
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
    """Fold pos_conv's weight_norm parametrization.

    Leaving the parametrization in place puts extra operations into the exported graph and
    makes some exporters fail. Folding does not change the numbers.
    """
    from torch.nn.utils import parametrize

    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )


def load_asset(asset_dir: Path) -> tuple[torch.nn.Module, torch.nn.Linear, int, int]:
    """Return (encoder, final_proj, layer_offset, num_hidden_layers)."""
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
    """Write the ONNX and return the name of the exporter used ("dynamo" / "legacy").

    `external_data=False` embeds the weights into the graph. The dynamo exporter's default
    (`external_data=True`) writes the weights out separately to `<path>.data` and leaves
    `path` as a small graph of pointers. The caller only `shutil.move`s `path`, so with
    the default the `.data` is left behind in the temporary directory and discarded, and
    the moved asset cannot be loaded because its external data is missing (measured
    2026-07-10: `InferenceSession` failing with `External data path does not exist`).
    HuBERT's weights fit within ONNX's 2GB protobuf limit in both fp32 and fp16, so
    embedding them is fine.
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
    except Exception:  # exporters raise a wide variety of exceptions, so catch broadly
        # **Report this loudly.** If this except swallowed the exception, we could fall
        # back to legacy silently even though dynamo would have succeeded (e.g. torch.onnx
        # dying while writing its progress ✅ to a Windows cp1252 stdout). main()'s UTF-8
        # reconfigure removes that cause, but always print a traceback whenever the
        # fallback happens.
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
    """The output of `HubertModel.half()`, the implementation being replaced (the fp16
    gate's reference).

    The fp32 golden cannot serve as an fp16 reference. Half precision's absolute error
    lands on the order of 1e-1 relative to the scale of the hidden states (O(1)-O(2.5)),
    and the current runtime itself scores cosine 0.987 / max_abs 0.435 against the fp32
    golden. The question to ask is "did going to ONNX change the fp16 behaviour", and the
    reference for that is the torch fp16 being replaced.

    A GPU- and kernel-dependent reference. The tests are CUDA-gated, so it only means
    anything on a development machine.

    **The call order is load-bearing**: `.half()` rewrites the module in place, so export
    the fp32 graph first and only then go to half precision. Calling `.float()` afterwards
    does not restore the fp32 weights. Here the already-halved wrapper is called as-is, so
    the reference comes from exactly the same weights and the same layers as the ONNX
    fp16.
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
    # torch.onnx's progress display contains ✅. On the default Windows stdout (cp1252)
    # that raises UnicodeEncodeError, which export_graph's except misreads as "dynamo
    # failed" and silently falls back to legacy. Remove the cause here.
    # typeshed types sys.stdout/stderr as TextIO, which lacks .reconfigure(); at
    # runtime CPython gives TextIOWrapper, which has it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]

    parser = argparse.ArgumentParser(
        description="transformers HubertModel 資産 (hubert_contentvec/) を ONNX へ export する (offline, 2/2 段目)。",
        epilog=(
            "これは HuBERT 資産づくりの 2 段目。先に `uv run poe convert-hubert` で\n"
            "hubert_base.pt を資産ディレクトリへ変換しておくこと (詳細は convert-hubert --help)。\n"
            "\n"
            "手順:\n"
            "  uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden\n"
            "  -> <asset>/hubert_fp32.onnx + hubert_fp16.onnx + mapping.json を書き出す\n"
            "\n"
            "config の [rvc] は資産ディレクトリを指す:\n"
            '  hubert_model_file = "./hubert_contentvec"\n'
            "\n"
            "ライセンスは THIRD_PARTY_NOTICES.md を参照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asset", required=True, type=Path, help="hubert_contentvec/")
    parser.add_argument("--golden", required=True, type=Path, help="hubert_golden/")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="一時ディレクトリへ export して誤差を印字するだけ。資産は更新しない "
        "(等価ゲートに落ちれば、このモードでも非ゼロで終了する)",
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

        # Emit fp32 first. The `.half()` that follows destroys the module in place.
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
        # The fp16 gate's reference, taken from the same weights and layers as the ONNX
        # fp16.
        reference = torch_fp16_reference(half_wrapper, half_source)
        # Deferred: the return value of the fp16 export_graph (the exporter name) is
        # discarded. mapping.json's "exporter" records only the fp32 side, so an fp16-only
        # fall back to legacy leaves no trace (provenance only; the runtime never reads
        # it, and the fallback itself prints a traceback, so it is noticeable).
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

        if not should_write_assets(ok, args.measure_only):
            return

        # Deferred: the two moves are not atomic. With fp32 succeeding and fp16 failing, a
        # new fp32 could coexist with an old fp16 and an old mapping.json (unlikely, since
        # these are renames on the same FS). It does not break silently: the next load
        # makes parse_output_names raise ValueError.
        shutil.move(str(fp32_path), asset_dir / "hubert_fp32.onnx")
        shutil.move(str(fp16_path), asset_dir / "hubert_fp16.onnx")

    # Save the fp16 gate's reference on the golden side. The tests only read the npz and
    # need no transformers (it leaves the project dependencies in Task 8, so this is the
    # only place it can be captured).
    # numpy 2's savez stub declares allow_pickle:bool, which collides with the **reference
    # expansion -- a false positive from the type checker. It is correct at runtime.
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
    # guarantees here that the shape is one the runtime can read
    parse_output_names(mapping)
    with open(asset_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"wrote onnx + mapping.json -> {asset_dir}")


if __name__ == "__main__":
    main()
