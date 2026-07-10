"""ONNX ベース HuBERT runtime の単体テスト。

実物の HuBERT も transformers も使わない。`onnx` のグラフ API で 2 出力の極小
グラフをその場で組み、runtime の契約（出力名の引き当て・エラー経路・ファイル選択）
だけを固定する。
"""

import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto
from onnx import helper

L9_DIM = 2
L12_DIM = 3


def _tiny_graph(elem_type: int):
    """source (1,N) -> feats_l9_proj (1,N,2), feats_l12_raw (1,N,3)。

    値は入力の複製なので、テスト側で中身を検算できる。次元を 2 / 3 と変えてあるので
    どちらの出力を引いたかが shape から一意に分かる。
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
    """scripts/export_hubert_onnx.py が書き出すのと同じレイアウトの合成資産。"""
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

    model = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    assert model.is_half is False
    assert model.output_names == {
        (9, True): "feats_l9_proj",
        (12, False): "feats_l12_raw",
    }


def test_select_onnx_file_prefers_fp16_on_cuda(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=True)
    path, is_half = _select_onnx_file(asset, torch.device("cuda", 0), is_half=True)
    assert path.name == "hubert_fp16.onnx"
    assert is_half is True


def test_select_onnx_file_uses_fp32_on_cpu_even_when_half_requested(tmp_path):
    """fp16 グラフは CPUExecutionProvider で実質動かない。CPU では必ず fp32。"""
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=True)
    path, is_half = _select_onnx_file(asset, torch.device("cpu"), is_half=True)
    assert path.name == "hubert_fp32.onnx"
    assert is_half is False


def test_select_onnx_file_falls_back_to_fp32_when_fp16_absent(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    asset = _write_asset(tmp_path, fp16=False)
    path, is_half = _select_onnx_file(asset, torch.device("cuda", 0), is_half=True)
    assert path.name == "hubert_fp32.onnx"
    assert is_half is False


def test_select_onnx_file_raises_when_asset_missing(tmp_path):
    from vspeech.lib.rvc import _select_onnx_file

    with pytest.raises(FileNotFoundError, match="hubert_fp32.onnx"):
        _select_onnx_file(tmp_path, torch.device("cpu"), is_half=False)


def test_extract_features_picks_the_projected_output(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(
        model, _wav(), torch.device("cpu"), emb_output_layer=9, use_final_proj=True
    )
    assert out.shape == (1, 64, L9_DIM)
    assert out.dtype == torch.float32


def test_extract_features_picks_the_raw_output(asset_dir):
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    out = extract_features(
        model, _wav(), torch.device("cpu"), emb_output_layer=12, use_final_proj=False
    )
    assert out.shape == (1, 64, L12_DIM)


def test_extract_features_returns_the_graph_values(asset_dir):
    """出力名を引き当てるだけでなく、その出力の中身が返ること。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    wav = _wav()
    out = extract_features(
        model, wav, torch.device("cpu"), emb_output_layer=9, use_final_proj=True
    )
    expected = wav.unsqueeze(-1).expand(1, 64, L9_DIM)
    assert torch.allclose(out, expected, atol=1e-6)


def test_extract_features_rejects_an_unsupported_combination(asset_dir):
    """(9, False) は export されていない。推測せず、対応表を添えて落ちること。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    model = load_hubert_model(asset_dir, torch.device("cpu"), is_half=False)
    with pytest.raises(RuntimeError) as excinfo:
        extract_features(
            model, _wav(), torch.device("cpu"), emb_output_layer=9, use_final_proj=False
        )
    message = str(excinfo.value)
    assert "(9, False)" in message
    assert "(9, True)" in message  # 対応表が示されること
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
    """今の実物 mapping.json (scripts/convert_hubert.py 出力) には 'outputs' 自体がない。"""
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
    """JSON の文字列 "false" は bool("false") == True になるので黙って通してはいけない。"""
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
    """isinstance(True, int) は True なので JSON の true を層番号 1 として読んではいけない。"""
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
