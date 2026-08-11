"""Unit tests for the ONNX-based HuBERT runtime.

Uses neither the real HuBERT nor transformers. It builds a tiny two-output graph on the
spot through `onnx`'s graph API and pins only the runtime's contract (resolving output
names, the error paths, and file selection).
"""

import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto
from onnx import helper

from vspeech.lib.cuda_util import Device

L9_DIM = 2
L12_DIM = 3


def _tiny_graph(elem_type: int):
    """source (1,N) -> feats_l9_proj (1,N,2), feats_l12_raw (1,N,3)。

    The values are copies of the input, so the test can verify the contents. The
    dimensions differ (2 vs 3), so the shape uniquely reveals which output was taken.
    """
    source = helper.make_tensor_value_info("source", elem_type, [1, "N"])
    out9 = helper.make_tensor_value_info("feats_l9_proj", elem_type, [1, "N", L9_DIM])
    out12 = helper.make_tensor_value_info("feats_l12_raw", elem_type, [1, "N", L12_DIM])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])
    nodes = [
        helper.make_node("Unsqueeze", ["source", "axes"], ["u"]),
        helper.make_node("Concat", ["u"] * L9_DIM, ["feats_l9_proj"], axis=2),
        helper.make_node("Concat", ["u"] * L12_DIM, ["feats_l12_raw"], axis=2),
    ]
    graph = helper.make_graph(nodes, "tiny_hubert", [source], [out9, out12], [axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _write_asset(path, *, fp16: bool = False):
    """A synthetic asset with the same layout scripts/export_hubert_onnx.py writes."""
    onnx.save(_tiny_graph(TensorProto.FLOAT), str(path / "hubert_fp32.onnx"))
    if fp16:
        onnx.save(_tiny_graph(TensorProto.FLOAT16), str(path / "hubert_fp16.onnx"))
    with open(path / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "layer_offset": 0,
                "num_hidden_layers": 12,
                "outputs": [
                    {"name": "feats_l9_proj", "layer": 9, "use_final_proj": True},
                    {"name": "feats_l12_raw", "layer": 12, "use_final_proj": False},
                ],
            },
            f,
        )
    return path


@pytest.fixture
def asset_dir(tmp_path):
    return _write_asset(tmp_path)


def _wav() -> torch.Tensor:
    t = np.arange(64, dtype=np.float32) / 16000.0
    return torch.from_numpy(np.sin(2 * np.pi * 220.0 * t).astype(np.float32)).unsqueeze(
        0
    )


def test_load_hubert_model_opens_the_fp32_graph(asset_dir):
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, Device("cpu"), is_half=False)
    assert model.is_half is False
    assert model.output_names == {
        (9, True): "feats_l9_proj",
        (12, False): "feats_l12_raw",
    }


def test_select_onnx_file_prefers_fp16_on_cuda(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=True)
    path, is_half = _select_onnx_file(asset, Device("cuda", 0), is_half=True)
    assert path.name == "hubert_fp16.onnx"
    assert is_half is True


def test_select_onnx_file_uses_fp32_on_cpu_even_when_half_requested(tmp_path):
    """An fp16 graph is effectively unusable on CPUExecutionProvider. CPU always gets
    fp32."""
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=True)
    path, is_half = _select_onnx_file(asset, Device("cpu"), is_half=True)
    assert path.name == "hubert_fp32.onnx"
    assert is_half is False


def test_select_onnx_file_falls_back_to_fp32_when_fp16_absent(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=False)
    path, is_half = _select_onnx_file(asset, Device("cuda", 0), is_half=True)
    assert path.name == "hubert_fp32.onnx"
    assert is_half is False


def test_select_onnx_file_raises_when_asset_missing(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    with pytest.raises(FileNotFoundError, match="hubert_fp32.onnx"):
        _select_onnx_file(tmp_path, Device("cpu"), is_half=False)


def test_extract_features_picks_the_projected_output(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, Device("cpu"), is_half=False)
    out = extract_features(
        model, _wav(), torch.device("cpu"), emb_output_layer=9, use_final_proj=True
    )
    assert out.shape == (1, 64, L9_DIM)
    assert out.dtype == torch.float32


def test_extract_features_picks_the_raw_output(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, Device("cpu"), is_half=False)
    out = extract_features(
        model, _wav(), torch.device("cpu"), emb_output_layer=12, use_final_proj=False
    )
    assert out.shape == (1, 64, L12_DIM)


def test_extract_features_returns_the_graph_values(asset_dir):
    """Not only is the output name resolved -- the contents of that output are returned."""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, Device("cpu"), is_half=False)
    wav = _wav()
    out = extract_features(
        model, wav, torch.device("cpu"), emb_output_layer=9, use_final_proj=True
    )
    expected = wav.unsqueeze(-1).expand(1, 64, L9_DIM)
    assert torch.allclose(out, expected, atol=1e-6)


def test_extract_features_rejects_an_unsupported_combination(asset_dir):
    """(9, False) was never exported. Fail with the mapping table attached, never
    guess."""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, Device("cpu"), is_half=False)
    with pytest.raises(RuntimeError) as excinfo:
        extract_features(
            model, _wav(), torch.device("cpu"), emb_output_layer=9, use_final_proj=False
        )
    message = str(excinfo.value)
    assert "(9, False)" in message
    assert "(9, True)" in message  # the mapping table is shown
    assert "(12, False)" in message


def test_parse_output_names_builds_the_lookup_table():
    from vspeech.lib.rvc import parse_output_names

    mapping = {
        "layer_offset": 0,
        "outputs": [
            {"name": "feats_l9_proj", "layer": 9, "use_final_proj": True, "dim": 256},
            {"name": "feats_l12_raw", "layer": 12, "use_final_proj": False, "dim": 768},
        ],
    }
    assert parse_output_names(mapping) == {
        (9, True): "feats_l9_proj",
        (12, False): "feats_l12_raw",
    }


def test_parse_output_names_rejects_an_empty_table():
    import pytest

    from vspeech.lib.rvc import parse_output_names

    with pytest.raises(ValueError, match="outputs"):
        parse_output_names({"outputs": []})


def test_parse_output_names_rejects_a_missing_outputs_key():
    """Today's real mapping.json (written by scripts/convert_hubert.py) has no 'outputs'
    key at all."""
    import pytest

    from vspeech.lib.rvc import parse_output_names

    with pytest.raises(ValueError, match="outputs"):
        parse_output_names({"layer_offset": 0, "num_hidden_layers": 12})


def test_parse_output_names_rejects_a_duplicate_key():
    import pytest

    from vspeech.lib.rvc import parse_output_names

    mapping = {
        "outputs": [
            {"name": "feats_l9_proj", "layer": 9, "use_final_proj": True, "dim": 256},
            {
                "name": "feats_l9_proj_dup",
                "layer": 9,
                "use_final_proj": True,
                "dim": 256,
            },
        ],
    }
    with pytest.raises(ValueError):
        parse_output_names(mapping)


def test_parse_output_names_rejects_a_string_use_final_proj():
    """The JSON string "false" gives bool("false") == True, so it must not pass
    silently."""
    import pytest

    from vspeech.lib.rvc import parse_output_names

    mapping = {
        "outputs": [
            {
                "name": "feats_l9_proj",
                "layer": 9,
                "use_final_proj": "false",
                "dim": 256,
            },
        ],
    }
    with pytest.raises(ValueError):
        parse_output_names(mapping)


def test_parse_output_names_rejects_a_bool_layer():
    """isinstance(True, int) is True, so JSON's true must not be read as layer number 1."""
    import pytest

    from vspeech.lib.rvc import parse_output_names

    mapping = {
        "outputs": [
            {
                "name": "feats_l9_proj",
                "layer": True,
                "use_final_proj": True,
                "dim": 256,
            },
        ],
    }
    with pytest.raises(ValueError):
        parse_output_names(mapping)
