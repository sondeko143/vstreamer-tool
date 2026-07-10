# RVC HuBERT の ONNX 化（spec ②）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RVC の content encoder を `transformers.HubertModel`（eager）から ONNX / `onnxruntime` へ移し、`fairseq` と `transformers` を `pyproject.toml` と `uv.lock` から完全に取り除く。

**Architecture:** `final_proj` を焼き込んだ 2 出力（`feats_l9_proj` / `feats_l12_raw`）の ONNX グラフを fp32 / fp16 の 2 本 export する。runtime は `InferenceSession` を 1 つ開き、RVC モデルのメタデータ `(embOutputLayer, useFinalProj)` から出力名を引くだけになる。変換（fairseq→transformers）と export（transformers→ONNX）は一度きりのオフライン工程で、依存は extra ではなく `uv run --with` のオーバーレイで供給する。

**Tech Stack:** Python 3.11 / uv / onnxruntime-gpu 1.26 / torch 2.10.0+cu128 / transformers 4.57（export 時のみ）/ onnx + onnxscript（export 時のみ）/ pytest / ruff / ty / poethepoet

**設計書:** [docs/superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md](../specs/2026-07-10-rvc-hubert-onnx-design.md)
**ブランチ:** `feat/rvc-hubert-onnx`（作成済み。設計書は `9731a2b` / `7656279` で commit 済み）

## Global Constraints

- `requires-python = ">=3.11,<3.12"` は**変更しない**。Python 3.12 化は後続 spec ③。
- 等価しきい値の**単一情報源は `scripts/hubert_metrics.py`**。他所で数値をコピーせず必ず import する。
- しきい値を緩めるときは **export の実測値の 10 倍**を上限とし、実測値を根拠としてコメントに残す。**理由なく緩めない。**
- `change_voice` の音声 golden（`tests/assets/rvc_golden/change_voice_golden.npz`）は**再ベースラインしない**。`corr ≥ 0.999` / `SNR ≥ 40 dB` のまま通すこと。通らなければ export が壊れている。
- `extract_features()` の**戻り値の契約（shape / dtype / device）を変えない**。後段の `functional.interpolate` と `infer()` は無改修。
- `load_hubert_model(file_name, device, is_half)` の**シグネチャを変えない**。
- runtime（`vspeech/` 配下）は `fairseq` / `transformers` / `safetensors` を **import しない**。
- **runtime は層インデックスを推測しない。** 対応表は `mapping.json` から読み、未登録の組合せは明示的に例外にする。
- **fp16 の参照は fp32 golden ではなく torch fp16。** 実測 (2026-07-10, RTX 4060): 現行 runtime の `HubertModel.half()` 自身が fp32 golden 比 `cosine 0.987 / max_abs 0.435` を出す。fp32 golden に fp16 を絶対誤差で照らすゲートは原理的に成立しない。
- **`.half()` はモジュールを in-place で壊す。** fp32 グラフの export を必ず先に済ませること。`.float()` で戻しても fp32 の重みは復元しない。
- **`scripts/export_hubert_onnx.py` は起動時に stdout/stderr を UTF-8 へ `reconfigure` する。** torch.onnx の進捗表示に `✅` が含まれ、Windows の cp1252 stdout では `UnicodeEncodeError` になる。これを `except Exception` が「dynamo 失敗」と誤認して黙って legacy exporter へ落とす事故が実際に起きた。フォールバック時は traceback を必ず印字する。
- **`layer_offset` の off-by-one は `tests/test_hubert_export.py` だけが pin する。** 実資産の offset は 0 なので、`+ layer_offset` を落とす退行は golden ゲートすら通過する。このテストを消さないこと。
- `scripts/export_hubert_onnx.py` は `transformers` / `safetensors` を**関数内で遅延 import** する。module 直下に置くと Task 8 の依存撤去後にテストから import できなくなる。
- `torch.compile` を再導入しない（Windows/CUDA で `TritonMissing`）。
- `uv sync --extra rvc` は使わない（他の extra を巻き添えでアンインストールする）。**`uv sync --all-extras`** を使う。
- fairseq wheel URL（`poe convert-hubert` 内）から `--python 3.11` を落とさない。落とすとプロジェクトの処理系に落ちて cp311 wheel が入らない。
- 資産（`hubert_contentvec/`、`hubert_golden/`）は gitignore 済み。commit しない。

## File Structure

| ファイル | 役割 | 変更 |
|---|---|---|
| `vspeech/lib/rvc.py` | runtime。ONNX セッション化 + io_binding ヘルパ | Modify |
| `vspeech/worker/vc.py` | `change_voice` 呼び出しから `half_available` を落とす | Modify |
| `scripts/export_hubert_onnx.py` | オフライン export ツール（新規） | Create |
| `scripts/hubert_metrics.py` | fp16 用しきい値を追加 | Modify |
| `scripts/capture_change_voice_golden.py` | `half_available` 撤去に追随 | Modify |
| `tests/test_rvc_helpers.py` | io_binding ヘルパの単体テスト | Modify |
| `tests/test_hubert_runtime.py` | 極小 ONNX で runtime 契約を固定（transformers 不要へ） | Rewrite |
| `tests/test_hubert_equivalence.py` | fp32 / fp16 の等価ゲート | Rewrite |
| `tests/test_no_fairseq_import.py` → `tests/test_forbidden_imports.py` | 構造ゲートを `transformers` にも拡張 | Rename + Modify |
| `pyproject.toml` | 依存手術 + poe task + ty override | Modify |
| `CLAUDE.md` | オフライン工程の運用を追記 | Modify |

---

## Task 1: io_binding ヘルパの抽出（純リファクタ）

`infer()` の中の io_binding 分岐を、`extract_features()` と共有できる 2 つのヘルパに括り出す。この時点では挙動を一切変えない。

**Files:**
- Modify: `vspeech/lib/rvc.py:111-200`（`infer`）
- Test: `tests/test_rvc_helpers.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `_element_type(dtype: torch.dtype) -> type` — torch dtype → numpy 型。未対応なら `ValueError`
  - `_bind_torch_input(io_binding, name: str, tensor: torch.Tensor) -> torch.Tensor` — contiguous 化して CUDA ポインタを bind。返り値は**生存させるための参照**
  - `_ort_output_to_torch(ort_output, device: torch.device) -> torch.Tensor` — dlpack で受け取り、失敗時は numpy 経由

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_rvc_helpers.py` の末尾に追記:

```python
def test_element_type_maps_supported_dtypes():
    import numpy as np
    import torch

    from vspeech.lib.rvc import _element_type

    assert _element_type(torch.float16) is np.float16
    assert _element_type(torch.float32) is np.float32
    assert _element_type(torch.int64) is np.int64


def test_element_type_rejects_unsupported_dtype():
    import pytest
    import torch

    from vspeech.lib.rvc import _element_type

    with pytest.raises(ValueError, match="Unsupported dtype"):
        _element_type(torch.bfloat16)


def test_ort_output_to_torch_falls_back_to_numpy():
    """dlpack が使えない ORT 値でも numpy 経由で torch tensor を返すこと。"""
    import numpy as np
    import torch

    from vspeech.lib.rvc import _ort_output_to_torch

    class _NoDlpack:
        def numpy(self):
            return np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    out = _ort_output_to_torch(_NoDlpack(), torch.device("cpu"))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 3)
    assert out.dtype == torch.float32
    assert out[0, 1, 2].item() == 5.0
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run --all-extras pytest tests/test_rvc_helpers.py -k "element_type or ort_output" -v`
Expected: FAIL — `ImportError: cannot import name '_element_type' from 'vspeech.lib.rvc'`

- [ ] **Step 3: ヘルパを実装し `infer()` を書き換える**

`vspeech/lib/rvc.py` の `infer()` の直前にヘルパを置く:

```python
_ORT_ELEMENT_TYPES: dict[torch.dtype, type] = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int64: np.int64,
}


def _element_type(dtype: torch.dtype) -> type:
    try:
        return _ORT_ELEMENT_TYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}") from None


def _bind_torch_input(io_binding: Any, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """torch の CUDA バッファを ORT の入力へゼロコピーで bind する。

    返り値は呼び出し側で**参照を保持する**こと。contiguous 化で新しい tensor が
    生まれる場合があり、束縛したポインタの寿命がそれに依存する。
    """
    tensor = tensor.contiguous()
    device = tensor.device
    io_binding.bind_input(
        name=name,
        device_type="cuda",
        device_id=device.index if device.index is not None else 0,
        element_type=_element_type(tensor.dtype),
        shape=tuple(tensor.shape),
        buffer_ptr=tensor.data_ptr(),
    )
    return tensor


def _ort_output_to_torch(ort_output: Any, device: torch.device) -> torch.Tensor:
    try:
        from torch.utils import dlpack

        try:
            dlp = ort_output._ortvalue.to_dlpack()
        except AttributeError:
            dlp = ort_output.to_dlpack()
        return dlpack.from_dlpack(dlp).clone()
    except Exception:
        return torch.tensor(ort_output.numpy(), device=device)
```

`infer()` の CUDA 分岐を差し替える（`bind` ローカル関数と dlpack ブロックを削除）:

```python
def infer(
    is_half: bool,
    session: InferenceSession,
    feats: torch.Tensor,
    pitch_length: torch.Tensor,
    pitch: torch.Tensor | None,
    pitchf: torch.Tensor | None,
    sid: torch.Tensor,
):
    device = feats.device
    if device.type == "cuda":
        io_binding = session.io_binding()
        tensors = [
            _bind_torch_input(
                io_binding, "feats", feats.half() if is_half else feats.float()
            ),
            _bind_torch_input(io_binding, "p_len", pitch_length),
            _bind_torch_input(io_binding, "sid", sid),
        ]
        if pitch is not None and pitchf is not None:
            tensors.append(_bind_torch_input(io_binding, "pitch", pitch))
            tensors.append(_bind_torch_input(io_binding, "pitchf", pitchf))

        io_binding.bind_output(
            "audio", "cuda", device_id=device.index if device.index is not None else 0
        )
        session.run_with_iobinding(io_binding)
        audio1 = _ort_output_to_torch(io_binding.get_outputs()[0], device)
        del tensors
        return audio1.unsqueeze(0)

    # Fallback for CPU
    ...  # 以降は無変更
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run --all-extras pytest tests/test_rvc_helpers.py -v`
Expected: PASS（既存テストも含めて全て）

- [ ] **Step 5: 既存スイートが壊れていないことを確認**

Run: `uv run --all-extras pytest tests/ -q`
Expected: PASS（資産を要するテストは skip）

- [ ] **Step 6: commit**

```bash
git add vspeech/lib/rvc.py tests/test_rvc_helpers.py
git commit -m "refactor(rvc): extract io_binding helpers from infer()"
```

---

## Task 2: 出力名の定数と `mapping.json` パーサ

export ツールと runtime が同じ出力名・同じ対応表解釈を使えるよう、`vspeech/lib/rvc.py` に単一情報源を置く。純関数なので先に固定する。

**Files:**
- Modify: `vspeech/lib/rvc.py`
- Test: `tests/test_hubert_runtime.py`（末尾に追記。本ファイルは Task 4 で全面書き換えするが、この 2 関数のテストはそのまま残す）

**Interfaces:**
- Consumes: なし
- Produces:
  - `FEATS_L9_PROJ: str = "feats_l9_proj"`
  - `FEATS_L12_RAW: str = "feats_l12_raw"`
  - `parse_output_names(mapping: dict[str, Any]) -> dict[tuple[int, bool], str]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_hubert_runtime.py` の末尾に追記:

```python
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
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run --all-extras pytest tests/test_hubert_runtime.py -k parse_output_names -v`
Expected: FAIL — `ImportError: cannot import name 'parse_output_names'`

- [ ] **Step 3: 実装**

`vspeech/lib/rvc.py` の `HUBERT_SAMPLE_RATE` の直後に追記:

```python
# ONNX グラフの出力名。scripts/export_hubert_onnx.py がこの名前で export し、
# mapping.json が (emb_output_layer, use_final_proj) との対応を記録する。
# 実在する RVC モデルは v1 = (9, True) と v2 = (12, False) の 2 種類だけ。
FEATS_L9_PROJ = "feats_l9_proj"
FEATS_L12_RAW = "feats_l12_raw"


def parse_output_names(mapping: dict[str, Any]) -> dict[tuple[int, bool], str]:
    """mapping.json の `outputs` を (emb_output_layer, use_final_proj) -> 出力名 に開く。

    runtime は層インデックスを推測しない。ここで読んだ対応表だけを信じる。
    """
    outputs = mapping.get("outputs") or []
    if not outputs:
        raise ValueError("mapping.json に 'outputs' がありません。再 export が必要です")
    return {
        (int(o["layer"]), bool(o["use_final_proj"])): str(o["name"]) for o in outputs
    }
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run --all-extras pytest tests/test_hubert_runtime.py -k parse_output_names -v`
Expected: PASS（2 件）

- [ ] **Step 5: commit**

```bash
git add vspeech/lib/rvc.py tests/test_hubert_runtime.py
git commit -m "feat(rvc): add ONNX output-name constants and mapping.json parser"
```

---

## Task 3: fp16 しきい値の追加

`scripts/hubert_metrics.py` に fp16 用しきい値を足す。初期値で置き、Task 5 の実測で確定させる。

**Files:**
- Modify: `scripts/hubert_metrics.py:15-21`
- Test: `tests/test_hubert_metrics.py`

**Interfaces:**
- Consumes: なし
- Produces: `COSINE_MIN_FP16: float`、`MAX_ABS_MAX_FP16: float`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_hubert_metrics.py` の末尾に追記:

```python
def test_fp16_thresholds_are_looser_than_fp32_but_still_tight():
    """fp16 ゲートは fp32 より緩いが、無意味に緩くはないこと。

    `1e-1` / `0.999` は**動かさない硬い上限**。実測 x 10 がこれを超えるなら
    fp16 export が壊れているということなので、しきい値ではなく export を疑う。
    """
    from scripts.hubert_metrics import COSINE_MIN
    from scripts.hubert_metrics import COSINE_MIN_FP16
    from scripts.hubert_metrics import MAX_ABS_MAX
    from scripts.hubert_metrics import MAX_ABS_MAX_FP16

    assert MAX_ABS_MAX_FP16 > MAX_ABS_MAX
    assert MAX_ABS_MAX_FP16 <= 1e-1
    assert COSINE_MIN_FP16 <= COSINE_MIN
    assert COSINE_MIN_FP16 >= 0.999
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run --all-extras pytest tests/test_hubert_metrics.py -k fp16 -v`
Expected: FAIL — `ImportError: cannot import name 'COSINE_MIN_FP16'`

- [ ] **Step 3: 実装**

`scripts/hubert_metrics.py` の `SNR_MIN_DB = 40.0` の直後に追記:

```python
# fp16 ONNX グラフ vs fp32 golden。12 層ぶんの半精度累積があるため fp32 の 1e-4 は
# 原理的に通らない。値は `scripts/export_hubert_onnx.py --measure-only` の実測の 10 倍。
# 実測 (2026-07-10, RTX 4060, hubert_fp16.onnx): TASK 5 で記入すること。
COSINE_MIN_FP16 = 0.9999
MAX_ABS_MAX_FP16 = 1e-2
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run --all-extras pytest tests/test_hubert_metrics.py -v`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add scripts/hubert_metrics.py tests/test_hubert_metrics.py
git commit -m "feat(rvc): add fp16 equivalence thresholds to hubert_metrics"
```

---

## Task 4: runtime を ONNX セッションへ置換

`HubertBundle`（transformers + Linear + offset）を `HubertSession`（`InferenceSession` + 出力名表）へ置き換える。`tests/test_hubert_runtime.py` を極小 ONNX ベースに書き換え、transformers をテストから切る。

**Files:**
- Modify: `vspeech/lib/rvc.py`（import 群、`HubertBundle`、`extract_features`、`load_hubert_model`、`_extract_hubert_feats`、`change_voice`）
- Modify: `vspeech/worker/vc.py:210`, `vspeech/worker/vc.py:270`
- Modify: `scripts/capture_change_voice_golden.py:91`, `:109`
- Modify: `pyproject.toml`（dev group に `onnx`）
- Rewrite: `tests/test_hubert_runtime.py`

**Interfaces:**
- Consumes: `FEATS_L9_PROJ` / `FEATS_L12_RAW` / `parse_output_names`（Task 2）、`_bind_torch_input` / `_ort_output_to_torch`（Task 1）
- Produces:
  - `class HubertSession` — フィールド `session: InferenceSession`, `output_names: dict[tuple[int, bool], str]`, `is_half: bool`
  - `_select_onnx_file(asset_dir: Path, device: torch.device, is_half: bool) -> tuple[Path, bool]`
  - `load_hubert_model(file_name: Path, device: torch.device, is_half: bool) -> HubertSession`（シグネチャ不変）
  - `extract_features(model: HubertSession, feats, dev, emb_output_layer=9, use_final_proj=True) -> torch.Tensor`（契約不変）
  - `change_voice(...)` — **`half_available` 引数を削除**（他は不変）

- [ ] **Step 1: dev group に `onnx` を追加**

`pyproject.toml` の `[dependency-groups] dev` に 1 行足す（極小 ONNX をテストで組み立てるため。runtime extras には入れない）:

```toml
    "beniget>=0.5.0",
    "onnx>=1.19,<2",
]
```

Run: `uv sync --all-extras`
Expected: `onnx` が入る

- [ ] **Step 2: 失敗するテストを書く（`tests/test_hubert_runtime.py` を全面置換）**

```python
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
```

（Task 2 で追記した `test_parse_output_names_*` の 2 件はこのファイルの末尾にそのまま残すこと。）

- [ ] **Step 3: 失敗を確認**

Run: `uv run --all-extras pytest tests/test_hubert_runtime.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_onnx_file'` および `HubertSession` 不在

- [ ] **Step 4: `vspeech/lib/rvc.py` を実装**

import 群から `safetensors` / `transformers` を落とす:

```python
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import torch
import torchaudio.transforms as T
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel
from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from torch.nn import functional

from vspeech.config import RvcConfig
from vspeech.lib.pitch_extract import pitch_extract
from vspeech.logger import logger
```

`HubertBundle` を `HubertSession` へ置き換える:

```python
@dataclass
class HubertSession:
    """ONNX 化した ContentVec の runtime 表現。

    `final_proj` はグラフに焼き込まれているので runtime には持たない。どの出力が
    どの (emb_output_layer, use_final_proj) に対応するかは mapping.json が唯一の情報源。
    """

    session: InferenceSession
    output_names: dict[tuple[int, bool], str]
    is_half: bool
```

`extract_features` を置き換える:

```python
def extract_features(
    model: HubertSession,
    feats: torch.Tensor,
    dev: torch.device,
    emb_output_layer: int = 9,
    use_final_proj: bool = True,
) -> torch.Tensor:
    key = (emb_output_layer, use_final_proj)
    try:
        output_name = model.output_names[key]
    except KeyError:
        supported = ", ".join(
            f"({layer}, {proj})" for layer, proj in sorted(model.output_names)
        )
        raise RuntimeError(
            f"HuBERT ONNX 資産は (emb_output_layer, use_final_proj)={key} を出力しません。"
            f" 利用可能な組合せ: {supported}。"
            " scripts/export_hubert_onnx.py で再 export してください。"
        ) from None

    source = feats.to(
        device=dev, dtype=torch.float16 if model.is_half else torch.float32
    )
    if dev.type == "cuda":
        io_binding = model.session.io_binding()
        # bind したポインタの寿命は `bound` が握る。run が終わるまで捨てないこと。
        bound = _bind_torch_input(io_binding, "source", source)
        io_binding.bind_output(
            output_name, "cuda", device_id=dev.index if dev.index is not None else 0
        )
        model.session.run_with_iobinding(io_binding)
        out = _ort_output_to_torch(io_binding.get_outputs()[0], dev)
        del bound
        return out

    result = cast(
        list,
        model.session.run(
            output_names=[output_name], input_feed={"source": source.cpu().numpy()}
        ),
    )
    return torch.from_numpy(np.asarray(result[0])).to(dev)
```

`load_hubert_model` を置き換える:

```python
def _select_onnx_file(
    asset_dir: Path, device: torch.device, is_half: bool
) -> tuple[Path, bool]:
    """使う ONNX ファイルと、それが fp16 かどうかを返す。

    fp16 グラフは CPUExecutionProvider では実質動かないので、CPU では必ず fp32。
    """
    if is_half and device.type == "cuda":
        fp16 = asset_dir / "hubert_fp16.onnx"
        if fp16.exists():
            return fp16, True
    fp32 = asset_dir / "hubert_fp32.onnx"
    if not fp32.exists():
        raise FileNotFoundError(
            f"HuBERT ONNX 資産がありません: {fp32}。"
            " `uv run poe export-hubert-onnx` で生成してください。"
        )
    return fp32, False


def load_hubert_model(
    file_name: Path, device: torch.device, is_half: bool
) -> HubertSession:
    """ONNX 化済み ContentVec 資産ディレクトリを読む（scripts/export_hubert_onnx.py の出力）。"""
    asset_dir = file_name.expanduser()
    model_file, half = _select_onnx_file(asset_dir, device, is_half)
    session = create_session(
        model_file, gpu_id=device.index if device.index is not None else 0
    )
    with open(asset_dir / "mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    return HubertSession(
        session=session,
        output_names=parse_output_names(mapping),
        is_half=half,
    )
```

`_extract_hubert_feats` から `half_available` を落とす（dtype は `HubertSession.is_half` が単一情報源）:

```python
def _extract_hubert_feats(
    hubert_model: HubertSession,
    audio_pad: torch.Tensor,
    device: torch.device,
    emb_output_layer: int,
    use_final_proj: bool,
) -> torch.Tensor:
    feats = audio_pad
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()  # nosec B101 - internal shape invariant
    feats = feats.view(1, -1)
    feats = extract_features(
        model=hubert_model,
        feats=feats,
        dev=device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    return functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
        0, 2, 1
    )
```

`change_voice` の引数から `half_available: bool,` を削除し、`hubert_model: HubertBundle` を `hubert_model: HubertSession` に変え、`_extract_hubert_feats` 呼び出しから `half_available=half_available,` を削除する。

- [ ] **Step 5: 呼び出し側を追随させる**

`vspeech/worker/vc.py` の 2 箇所（warmup と本処理）から `half_available=half_available,` の行を削除する。`half_precision_available()` の呼び出しと `load_hubert_model(is_half=half_available)` は**残す**。

`scripts/capture_change_voice_golden.py`:
- `build_rvc_runtime` の返す dict から `"half_available": half_available,` の行を削除
- `run_change_voice` の `change_voice(...)` 呼び出しから `half_available=rt["half_available"],` を削除

- [ ] **Step 6: テストが通ることを確認**

Run: `uv run --all-extras pytest tests/test_hubert_runtime.py -v`
Expected: PASS（10 件）

Run: `uv run --all-extras pytest tests/ -q`
Expected: PASS（`tests/test_hubert_equivalence.py` はまだ古い API を参照して FAIL するので、Task 6 まで一時的に落ちてよい。落ちる場合は次の Step で確認する）

- [ ] **Step 7: `vspeech/` から transformers が消えたことを確認**

Run: `uv run --all-extras python -c "import ast,pathlib; [print(p) for p in pathlib.Path('vspeech').rglob('*.py') for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))) if isinstance(n,(ast.Import,ast.ImportFrom)) and 'transformers' in (getattr(n,'module',None) or '') + ''.join(a.name for a in getattr(n,'names',[]))]"`
Expected: 何も出力されない

- [ ] **Step 8: commit**

```bash
git add vspeech/lib/rvc.py vspeech/worker/vc.py scripts/capture_change_voice_golden.py tests/test_hubert_runtime.py pyproject.toml uv.lock
git commit -m "feat(rvc)!: run HuBERT through onnxruntime instead of transformers"
```

---

## Task 5: オフライン export ツール

`hubert_contentvec/`（transformers 資産）から fp32 / fp16 の ONNX を作り、`hubert_golden.npz` に対して自己検証する。**通らなければ資産を書かない。**

**Files:**
- Create: `scripts/export_hubert_onnx.py`
- Create: `tests/test_hubert_export.py`
- Modify: `scripts/hubert_metrics.py`（Task 3 で置いた fp16 しきい値を実測で確定）

**Interfaces:**
- Consumes: `FEATS_L9_PROJ` / `FEATS_L12_RAW` / `parse_output_names`（Task 2）、`COSINE_MIN` / `MAX_ABS_MAX` / `COSINE_MIN_FP16` / `MAX_ABS_MAX_FP16` / `feature_cosine` / `feature_max_abs_diff`（Task 3）。検証波形は golden npz の `wav` を読むので `scripts/convert_hubert.py` は import しない（あれが transformers を module 直下で引くため）。
- Produces:
  - `layer_indices(layer_offset: int) -> tuple[int, int]` — `(9 + offset, 12 + offset)`
  - `class HubertOnnxWrapper(torch.nn.Module)` — `__init__(model, final_proj, layer_offset)`、`forward(source) -> tuple[Tensor, Tensor]`
  - `hubert_fp32.onnx` / `hubert_fp16.onnx` / 更新された `mapping.json`

**重い import は関数内に遅延させる。** `transformers` / `safetensors` を module 直下で import すると、Task 8 で依存を外した後にこのモジュールを**テストから import できなくなる**。`scripts/convert_hubert.py` が fairseq に対してやっているのと同じ手（`load_fairseq_model` 内で import）を踏襲する。これにより `layer_indices` / `HubertOnnxWrapper` がダミーの `nn.Module` で単体テストできる。

- [ ] **Step 1: 失敗するテストを書く（`tests/test_hubert_export.py` を新規作成）**

実資産の `layer_offset` は 0 なので、`L9 + layer_offset` から `+ layer_offset` を落とす退行は
golden ゲートを含めて全テストを通過してしまう。**ここだけが off-by-one を pin する。**

```python
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
        hidden = tuple(
            torch.full((1, 2, 4), float(i)) for i in range(HIDDEN_STATES)
        )
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
```

- [ ] **Step 2: 失敗を確認**

Run: `uv run --all-extras pytest tests/test_hubert_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_hubert_onnx'`

- [ ] **Step 3: `scripts/export_hubert_onnx.py` を書く**

```python
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
from vspeech.lib.rvc import FEATS_L12_RAW
from vspeech.lib.rvc import FEATS_L9_PROJ
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
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


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
    """ONNX を書き、使った exporter 名 ("dynamo" / "legacy") を返す。"""
    kwargs: dict[str, Any] = dict(
        input_names=["source"],
        output_names=[FEATS_L9_PROJ, FEATS_L12_RAW],
        dynamic_axes={
            "source": {1: "N"},
            FEATS_L9_PROJ: {1: "T"},
            FEATS_L12_RAW: {1: "T"},
        },
        opset_version=OPSET,
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
        print(f"{label} {golden_key}: cosine={cosine:.8f} max_abs={max_abs:.3e} [{verdict}]")
        ok = ok and verdict == "OK"
    return ok


def main() -> None:
    # torch.onnx の進捗表示は ✅ を含む。Windows の既定 stdout (cp1252) では
    # UnicodeEncodeError になり、export_graph の except がそれを「dynamo 失敗」と
    # 誤認して黙って legacy へ落ちる。ここで潰しておく。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    np.savez(golden_dir / "hubert_golden_fp16.npz", wav=wav, **reference)

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
```

- [ ] **Step 4: 単体テストが通ることを確認**

Run: `uv run --all-extras pytest tests/test_hubert_export.py -v`
Expected: PASS 4 件

Run: `uv run --all-extras python -c "import scripts.export_hubert_onnx"`
Expected: 例外なし（transformers を module 直下で import していないことの確認。Task 8 後もこれが通る必要がある）

- [ ] **Step 5: `--measure-only` で確認する**

Run:
```bash
uv run --with onnx --with onnxscript python -m scripts.export_hubert_onnx \
  --asset ./hubert_contentvec --golden ./hubert_golden --measure-only
```

Expected（**2026-07-10 にこの機械で実測済み。この値を再現すること**）:

```
exported fp32 with dynamo exporter        <- legacy へ落ちたら失敗。原因を調べること
fp32 l9_proj: cosine=1.00000000 max_abs=1.010e-05 [OK]
fp32 l12_raw: cosine=1.00000000 max_abs=9.030e-06 [OK]
fp16 l9_proj: cosine=0.99999010 max_abs=1.379e-02 [OK]   <- 参照は torch fp16
fp16 l12_raw: cosine=0.99997235 max_abs=1.074e-02 [OK]
```

fp32 が FAIL する場合は export が壊れている。`fold_weight_norm` と exporter を疑うこと。**しきい値を緩めて逃げない。**

- [ ] **Step 6: fp16 しきい値を確定させる**

`scripts/hubert_metrics.py` の定数とコメントを実測値に置き換える。`COSINE_MIN_FP16` は据え置き、`MAX_ABS_MAX_FP16` だけ `1e-2` → `5e-2` に直す（実測 1.379e-2 の約 3.6 倍。10 倍規則の範囲内で、より厳しい側に取る）:

```python
# fp16 ONNX グラフ vs **torch fp16 参照**（fp32 golden ではない）。
# hidden state は O(1)-O(2.5) あり、半精度の絶対誤差はもともと 1e-1 オーダー。fp32 golden に
# 対しては現行 runtime の HubertModel.half() 自身が cosine 0.987 / max_abs 0.435 を出すので、
# fp32 golden を fp16 の参照にすること自体が誤り。問うべきは「ONNX 化で fp16 の振る舞いが
# 変わっていないか」であり、参照は置き換え対象の torch fp16 である。
# 実測 (2026-07-10, RTX 4060, ONNX fp16 vs torch fp16):
#   l9_proj  cosine=0.99999010 max_abs=1.379e-02
#   l12_raw  cosine=0.99997235 max_abs=1.074e-02
COSINE_MIN_FP16 = 0.9999
MAX_ABS_MAX_FP16 = 5e-2
```

`tests/test_hubert_metrics.py::test_fp16_thresholds_are_looser_than_fp32_but_still_tight` の `MAX_ABS_MAX_FP16 <= 1e-1` と `COSINE_MIN_FP16 >= 0.999` は**動かさない硬い上限**である。上の値は両方を満たす。超えるなら export が壊れている。

Run: `uv run --all-extras pytest tests/test_hubert_metrics.py -v`
Expected: PASS

- [ ] **Step 7: 本番 export を実行する**

Run:
```bash
uv run --with onnx --with onnxscript python -m scripts.export_hubert_onnx \
  --asset ./hubert_contentvec --golden ./hubert_golden
```
Expected: 全て `[OK]`、`wrote onnx + mapping.json -> hubert_contentvec`

Run: `ls hubert_contentvec/ hubert_golden/`
Expected: `hubert_fp32.onnx`（約 380MB）、`hubert_fp16.onnx`（約 190MB）、更新された `mapping.json`、そして `hubert_golden/hubert_golden_fp16.npz`（torch fp16 参照）

- [ ] **Step 8: commit**

```bash
git add scripts/export_hubert_onnx.py scripts/hubert_metrics.py tests/test_hubert_export.py
git commit -m "feat(rvc): add offline HuBERT ONNX exporter with a self-verifying gate"
```

（`hubert_contentvec/` は gitignore 済みなので資産は commit されない。）

---

## Task 6: 等価ゲートを ONNX 経路へ

**Files:**
- Modify: `vspeech/lib/rvc.py`（`create_session` が device を尊重するようにする）
- Modify: `vspeech/worker/vc.py:186`, `scripts/capture_change_voice_golden.py:81`（呼び出し追随）
- Rewrite: `tests/test_hubert_equivalence.py`

**Interfaces:**
- Consumes: `load_hubert_model` / `extract_features`（Task 4）、fp16 しきい値（Task 5 で確定）
- Produces: `create_session(model_file: Path, device: torch.device) -> InferenceSession`（`gpu_id: int` から差し替え）

### Step 0（前提バグの修正）: `create_session` が device を無視している

`create_session` は `torch.cuda.is_available()` だけで CUDA EP を先頭に挿し、**呼び出し側の `device` を一切見ない**。実測（2026-07-10, RTX 4060）:

```
create_session(fp32.onnx, gpu_id=0)  providers=['CUDAExecutionProvider','CPUExecutionProvider']
  l9_proj: cosine=0.99999966 max_abs=2.625e-03     <- CUDA EP の TF32 行列積
  l12_raw: cosine=0.99999893 max_abs=2.164e-03
InferenceSession(fp32.onnx, providers=['CPUExecutionProvider'])
  l9_proj: cosine=1.00000000 max_abs=1.010e-05     <- golden と一致
  l12_raw: cosine=1.00000000 max_abs=9.030e-06
```

これは 2 つの問題を同時に起こす:

1. **production のバグ**: `get_device()` は gpu 未設定なら `torch.device("cpu")` を返すが、`create_session` は CUDA EP を使う。「config で CPU を指定したのに黙って GPU で走る」。しかも `gpu_id=device.index` は `None` が渡る。
2. **ゲートが成立しない**: fp32 等価ゲートは CPU device で走らせるのに、CUDA ボックスでは実際には CUDA EP で走り、TF32 由来の `2.6e-3` で `MAX_ABS_MAX = 1e-4` を 26 倍超過する。

`2.6e-3` は EP の数値特性であってグラフの欠陥ではない（cosine は 0.99999966）。**しきい値を緩めて逃げてはいけない。** 直すのは `create_session` のほう。

- [ ] **Step 0-1: `create_session` のシグネチャを `device` に変える**

```python
def create_session(model_file: Path, device: torch.device) -> InferenceSession:
    """`device` を尊重してセッションを開く。

    以前は `torch.cuda.is_available()` だけで CUDA EP を選んでいたため、呼び出し側が
    CPU device を渡しても GPU で走っていた（`gpu_id=device.index` は None が渡る）。
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if device.type == "cuda" and torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(
            0,
            {
                "device_id": device.index if device.index is not None else 0,
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        )
    return InferenceSession(
        str(model_file.expanduser()),
        sess_options=sess_options,
        providers=providers,
        provider_options=providers_options,
    )
```

- [ ] **Step 0-2: 呼び出し 3 箇所を追随させる**

  - `vspeech/lib/rvc.py` の `load_hubert_model`: `create_session(model_file, device)`
  - `vspeech/worker/vc.py:186`: `create_session(rvc_config.model_file, device)`
  - `scripts/capture_change_voice_golden.py:81`: `create_session(rvc_config.model_file, device)`

  `create_rmvpe_session`（`vspeech/lib/pitch_extract.py:25`）も同じバグを持つが、**この branch では触らない**（スコープ外。ledger に Minor として記録する）。

- [ ] **Step 0-3: EP 選択を pin するテストを `tests/test_rvc_helpers.py` に足す**

```python
def test_create_session_uses_cpu_ep_for_a_cpu_device(tmp_path, monkeypatch):
    """CUDA が使えても device が CPU なら CUDA EP を積まないこと。

    以前は torch.cuda.is_available() だけで判定しており、config で CPU を指定しても
    GPU で走っていた。fp32 等価ゲートはこの差 (TF32, max_abs 2.6e-3) で落ちる。
    """
    import torch

    import vspeech.lib.rvc as rvc

    captured: dict = {}

    def fake_session(path, sess_options, providers, provider_options):
        captured["providers"] = providers
        return object()

    monkeypatch.setattr(rvc, "InferenceSession", fake_session)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    rvc.create_session(tmp_path / "m.onnx", torch.device("cpu"))
    assert captured["providers"] == ["CPUExecutionProvider"]

    rvc.create_session(tmp_path / "m.onnx", torch.device("cuda", 0))
    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

- [ ] **Step 1: `tests/test_hubert_equivalence.py` を全面置換**

```python
"""ONNX 版 HuBERT の数値等価ゲート。

fp32 グラフ: fairseq 時代に scripts/convert_hubert.py が捕獲した特徴量（fp32）を正解とし、
(9, use_final_proj=True) と (12, use_final_proj=False) の両方を厳密に照合する。

fp16 グラフ: 参照は **fp32 golden ではなく torch fp16**（置き換え対象の実装）。半精度の
絶対誤差は hidden state のスケール (O(1)-O(2.5)) に対して 1e-1 オーダーで、現行 runtime の
HubertModel.half() 自身が fp32 golden 比 cosine 0.987 / max_abs 0.435 を出す。したがって
fp32 golden を fp16 の参照にすること自体が誤り。問うべきは「ONNX 化で fp16 の振る舞いが
変わっていないか」であり、参照は scripts/export_hubert_onnx.py が捕獲した
hubert_golden_fp16.npz。GPU 依存の参照なので CUDA gating 済みの開発機でのみ走る。

資産と golden は派生物なので gitignore してある。環境変数が未設定なら skip し、
CPU/CI のスイートを壊さない（tests/test_change_voice_golden.py と同じ流儀）。
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hubert_metrics import COSINE_MIN
from scripts.hubert_metrics import COSINE_MIN_FP16
from scripts.hubert_metrics import MAX_ABS_MAX
from scripts.hubert_metrics import MAX_ABS_MAX_FP16
from scripts.hubert_metrics import feature_cosine
from scripts.hubert_metrics import feature_max_abs_diff

_ASSET_ENV = "VSPEECH_HUBERT_ASSET_DIR"
_GOLDEN_ENV = "VSPEECH_HUBERT_GOLDEN_DIR"

_asset = os.environ.get(_ASSET_ENV)
_golden = os.environ.get(_GOLDEN_ENV)
ASSET_DIR = Path(_asset) if _asset else None
GOLDEN_NPZ = Path(_golden) / "hubert_golden.npz" if _golden else None
GOLDEN_FP16_NPZ = Path(_golden) / "hubert_golden_fp16.npz" if _golden else None

# しきい値 (COSINE_MIN / MAX_ABS_MAX / *_FP16) の単一情報源は scripts/hubert_metrics.py。
# 緩めるときはそこで変更し、実測値を根拠としてコメントに残すこと（実測の 10 倍まで）。
pytestmark = pytest.mark.skipif(
    ASSET_DIR is None
    or not ASSET_DIR.exists()
    or GOLDEN_NPZ is None
    or not GOLDEN_NPZ.exists(),
    reason=f"${_ASSET_ENV} / ${_GOLDEN_ENV} not available",
)

CASES = [(9, True, "l9_proj"), (12, False, "l12_raw")]


def _compare(device: torch.device, is_half: bool, case) -> tuple[float, float]:
    """`is_half` は判定に使う参照 npz も選ぶ。fp16 の参照は torch fp16。"""
    from vspeech.lib.rvc import extract_features
    from vspeech.lib.rvc import load_hubert_model

    emb_output_layer, use_final_proj, golden_key = case
    assert ASSET_DIR is not None and GOLDEN_NPZ is not None  # skipif guarantees
    assert GOLDEN_FP16_NPZ is not None

    data = np.load(GOLDEN_FP16_NPZ if is_half else GOLDEN_NPZ)
    wav = np.load(GOLDEN_NPZ)["wav"].astype(np.float32)
    reference = data[golden_key].astype(np.float32)

    model = load_hubert_model(ASSET_DIR, device, is_half=is_half)
    assert model.is_half == is_half, "期待した精度のグラフが選ばれていない"

    out = extract_features(
        model,
        torch.from_numpy(wav).unsqueeze(0),
        device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    candidate = out.squeeze(0).float().cpu().numpy()
    assert candidate.shape == reference.shape, f"{candidate.shape} vs {reference.shape}"
    return feature_cosine(candidate, reference), feature_max_abs_diff(
        candidate, reference
    )


@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp32_features_match_fairseq_golden(
    emb_output_layer, use_final_proj, golden_key
):
    cosine, max_abs = _compare(
        torch.device("cpu"), False, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN, f"cosine {cosine:.8f} < {COSINE_MIN}"
    assert max_abs <= MAX_ABS_MAX, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX:.1e}"


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or GOLDEN_FP16_NPZ is None
    or not GOLDEN_FP16_NPZ.exists(),
    reason="fp16 graph needs CUDA and hubert_golden_fp16.npz",
)
@pytest.mark.parametrize(("emb_output_layer", "use_final_proj", "golden_key"), CASES)
def test_fp16_features_match_the_torch_fp16_reference(
    emb_output_layer, use_final_proj, golden_key
):
    """ONNX 化で fp16 の振る舞いが変わっていないこと。fp32 golden とは比べない。"""
    cosine, max_abs = _compare(
        torch.device("cuda", 0), True, (emb_output_layer, use_final_proj, golden_key)
    )
    assert cosine >= COSINE_MIN_FP16, f"cosine {cosine:.8f} < {COSINE_MIN_FP16}"
    assert max_abs <= MAX_ABS_MAX_FP16, f"max-abs {max_abs:.3e} > {MAX_ABS_MAX_FP16:.1e}"
```

- [ ] **Step 2: 資産ありで実行**

Run:
```bash
VSPEECH_HUBERT_ASSET_DIR=./hubert_contentvec \
VSPEECH_HUBERT_GOLDEN_DIR=./hubert_golden \
uv run --all-extras pytest tests/test_hubert_equivalence.py -v
```
Expected: PASS 4 件（fp32 2 件 + fp16 2 件）

`assert model.is_half == is_half` が落ちる場合、`hubert_fp16.onnx` が無いか `_select_onnx_file` が誤っている。

- [ ] **Step 3: 資産なしで skip されることを確認**

Run: `uv run --all-extras pytest tests/test_hubert_equivalence.py -v`
Expected: 4 skipped

- [ ] **Step 4: ゲートが空虚でないことを確かめる**

fp16 テストの参照を一時的に `hubert_golden.npz`（fp32 golden）に差し替えて実行し、`max_abs` が
`5e-2` を大きく超えて FAIL することを確認する。正しいグラフでも**間違った参照に対しては落ちる**
はずで、落ちないならゲートが何も見ていない。確認後、必ず元に戻して再実行する。

- [ ] **Step 5: commit**

Step 0 と Step 1 は別 commit にする（前者は既存バグの修正、後者はゲートの書き換え）。

```bash
git add vspeech/lib/rvc.py vspeech/worker/vc.py scripts/capture_change_voice_golden.py tests/test_rvc_helpers.py
git commit -m "fix(rvc): make create_session honour the caller's device"

git add tests/test_hubert_equivalence.py
git commit -m "test(rvc): gate the ONNX HuBERT (fp32 vs fairseq golden, fp16 vs torch fp16)"
```

---

## Task 7: 構造ゲートを `transformers` へ拡張

**Files:**
- Rename + Modify: `tests/test_no_fairseq_import.py` → `tests/test_forbidden_imports.py`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: ファイルを rename**

```bash
git mv tests/test_no_fairseq_import.py tests/test_forbidden_imports.py
```

- [ ] **Step 2: 内容を置換**

```python
"""runtime に重い ML フレームワークを二度と入れないための構造ゲート。

- fairseq: requires-python 引き上げの唯一の障害（上流は 0.12.2 で凍結、リポジトリは
  2026-03-20 に archived）。spec ① で撤去。
- transformers: uv.lock に載るだけで `uv audit` に 3 件の advisory を持ち込む。
  spec ② で content encoder を ONNX 化して撤去。

どちらも offline ツール (scripts/convert_hubert.py, scripts/export_hubert_onnx.py) では
使ってよい。禁じるのは `vspeech/` 配下、すなわち runtime だけ。
"""

import ast
from pathlib import Path

import pytest

VSPEECH_DIR = Path(__file__).resolve().parents[1] / "vspeech"

FORBIDDEN = ("fairseq", "transformers")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_vspeech_never_imports(forbidden: str):
    offenders = []
    for py_file in sorted(VSPEECH_DIR.rglob("*.py")):
        for module in _imported_modules(py_file):
            if module == forbidden or module.startswith(f"{forbidden}."):
                offenders.append(f"{py_file.relative_to(VSPEECH_DIR.parent)}: {module}")
    assert not offenders, (
        f"{forbidden} import leaked back into the runtime:\n" + "\n".join(offenders)
    )


def test_the_gate_would_catch_a_regression(tmp_path):
    """ゲートが空虚でないこと: 実際に禁止 import を含むファイルを検出できる。"""
    leaked = tmp_path / "leak.py"
    leaked.write_text("from transformers import HubertModel\n", encoding="utf-8")
    assert "transformers" in list(_imported_modules(leaked))
```

- [ ] **Step 3: 実行**

Run: `uv run --all-extras pytest tests/test_forbidden_imports.py -v`
Expected: PASS 3 件

- [ ] **Step 4: ゲートが本当に効くことを手で確かめる**

`vspeech/lib/rvc.py` の先頭に一時的に `from transformers import HubertModel` を足す。

Run: `uv run --all-extras pytest tests/test_forbidden_imports.py -v`
Expected: FAIL — `transformers import leaked back into the runtime: vspeech/lib/rvc.py: transformers`

足した行を消す。

Run: `uv run --all-extras pytest tests/test_forbidden_imports.py -v`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add tests/test_forbidden_imports.py
git commit -m "test: generalize the import gate to forbid transformers in the runtime"
```

---

## Task 8: 依存手術

`transformers` / `fairseq` を `pyproject.toml` と `uv.lock` から完全に取り除き、オフライン工程を poe task にする。

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `scripts/convert_hubert.py`（既存）、`scripts/export_hubert_onnx.py`（Task 5）
- Produces: `poe convert-hubert`、`poe export-hubert-onnx`

- [ ] **Step 1: `rvc` extra から `transformers` を撤去**

`pyproject.toml`:

```toml
rvc = [
    "torch ; sys_platform == 'win32'",
    "pyworld>=0.3.3,<0.4",
    "faiss-cpu>=1.7.2,<2 ; sys_platform == 'win32'",
    "onnxruntime-gpu>=1.24.4,<2",
    "torchaudio ; sys_platform == 'win32'",
    "numpy>=1.23,<2",
    "scipy>=1.10.1,<2",
]
```

- [ ] **Step 2: `convert` extra を削除**

`pyproject.toml` から次のブロックを丸ごと消す（先行するコメント 3 行も含む）:

```toml
# HuBERT 変換ツール専用（scripts/convert_hubert.py）。runtime には含めない。
# fairseq 0.12.2 は numpy>=1.24 で撤去された旧エイリアスを使うため <1.24 を保持する。
# uv は 1 環境に numpy を 1 版しか解決しないので、rvc とは別の使い捨て 3.11 環境で使うこと。
convert = [
    "fairseq ; sys_platform == 'win32'",
    "torch ; sys_platform == 'win32'",
    "torchaudio ; sys_platform == 'win32'",
    "transformers>=4.44,<5",
    "numpy>=1.18,<1.24",
]
```

- [ ] **Step 3: `[tool.uv.sources]` から fairseq を削除**

次の 1 行を消す:

```toml
fairseq = { url = "https://github.com/sondeko143/fairseq-311/releases/download/v0.12.2.post1/fairseq-0.12.2.post1-cp311-cp311-win_amd64.whl" }
```

- [ ] **Step 4: オフライン工程を poe task にする**

`[tool.poe.tasks]` の `deadcode` の直後に追記:

```toml
# 一度きりのオフライン工程。依存 (fairseq / transformers / onnx / onnxscript) は
# pyproject にも uv.lock にも載せず、`uv run --with` の一時環境で供給する。
# `poe security` / `poe deadcode` と同じ手口。
#
# convert-hubert: hubert_base.pt -> hubert_contentvec/ (transformers 資産)。
#   `--python 3.11` は必須。省くとプロジェクトの処理系に落ち、cp311 の fairseq
#   wheel が入らない（spec ③ で 3.12 化した後に効いてくる）。`--no-project` なので
#   [tool.uv.sources] は読まれず、torch は PyPI 版 (Windows は CPU) になる。変換は
#   CPU で足りるのでこれでよい。
#   例: uv run poe convert-hubert --input ~/.config/vstreamer/hubert_base.pt \
#           --output ./hubert_contentvec --golden ./hubert_golden
convert-hubert = { cmd = "uv run --isolated --no-project --python 3.11 --with 'fairseq @ https://github.com/sondeko143/fairseq-311/releases/download/v0.12.2.post1/fairseq-0.12.2.post1-cp311-cp311-win_amd64.whl' --with 'numpy<1.24' --with torch --with transformers python -m scripts.convert_hubert", help = "hubert_base.pt -> transformers 資産 (offline, 3.11 使い捨て環境)" }
# export-hubert-onnx: hubert_contentvec/ -> hubert_fp32.onnx + hubert_fp16.onnx。
#   `--no-project` は付けない。fp16 を CUDA 上で export するためプロジェクト環境の
#   cu128 torch が要る。
#   例: uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden
export-hubert-onnx = { cmd = "uv run --with transformers --with onnx --with onnxscript python -m scripts.export_hubert_onnx", help = "transformers 資産 -> ONNX (offline, プロジェクト環境 + overlay)" }
```

- [ ] **Step 5: ty override を足す**

`[tool.ty.environment]` の直後に追記:

```toml
# scripts/convert_hubert.py と scripts/export_hubert_onnx.py はオフライン専用ツール。
# その依存 (fairseq / transformers / onnx / safetensors) は poe task の `uv run --with`
# が一時環境で供給するので、プロジェクト依存ではない。未解決 import はここでは正常な
# 状態であって、`python-health` が禁じる「extras 未同期による偽陽性の抑制」ではない。
[[tool.ty.overrides]]
include = ["scripts/convert_hubert.py", "scripts/export_hubert_onnx.py"]

[tool.ty.overrides.rules]
unresolved-import = "ignore"
```

- [ ] **Step 6: lock を更新する**

Run: `uv lock`
Expected: 成功

Run: `uv sync --all-extras`
Expected: 成功。`transformers` / `fairseq` がアンインストールされる

- [ ] **Step 7: lock から両者が消えたことを確認**

Run: `grep -c "^name = \"fairseq\"" uv.lock || true; grep -c "^name = \"transformers\"" uv.lock || true`
Expected: `0` と `0`（`grep -c` は 0 件のとき exit 1 を返すので `|| true` を付ける）

Run: `grep -A 2 "^name = \"numpy\"" uv.lock`
Expected: `version = "1.26.x"` 系（1.23.5 ではない）

- [ ] **Step 8: `uv audit` が torch 1 件だけになったことを確認**

Run: `uv audit`
Expected: `torch 2.10.0+cu128 has 1 known vulnerability` のみ（`GHSA-rrmf-rvhw-rf47`）。`transformers` の 3 件は消えている

- [ ] **Step 9: ty が通ることを確認**

Run: `uv run --all-extras ty check`
Expected: PASS。もし `[[tool.ty.overrides]]` のスキーマが違うと言われたら `uv run ty check --help` と ty のバージョンを確認し、同等の設定に読み替えること（override は 2 ファイル・`unresolved-import` 1 ルールに限定する。範囲を広げない）

- [ ] **Step 10: `CLAUDE.md` を更新**

`## Commands` の `uv sync --extra rvc` の行のコメントを直し、オフライン工程の節を足す。`### Architecture` の直前に:

```markdown
### HuBERT assets (RVC only, offline)

The RVC content encoder runs as ONNX. Its assets are derived, gitignored, and built by two
one-shot offline steps whose dependencies live **only** in the poe task's `uv run --with`
overlay — never in `pyproject.toml` or `uv.lock`:

```sh
# hubert_base.pt -> hubert_contentvec/ (transformers asset). Needs fairseq, so it runs in a
# throwaway 3.11 environment. Keep ~/.config/vstreamer/hubert_base.pt: it is the input.
uv run poe convert-hubert --input ~/.config/vstreamer/hubert_base.pt \
    --output ./hubert_contentvec --golden ./hubert_golden

# hubert_contentvec/ -> hubert_fp32.onnx + hubert_fp16.onnx. Runs on the project env (cu128
# torch) because the fp16 graph is exported on CUDA. Self-verifies against the golden.
uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden
```

`rvc.hubert_model_file` points at the **asset directory**, not a file. The runtime opens only
`hubert_*.onnx` + `mapping.json`; `vspeech/` never imports `fairseq` or `transformers`
(enforced by `tests/test_forbidden_imports.py`). Never run `uv sync --extra rvc` — it
uninstalls the other extras. Use `uv sync --all-extras`.
```

`## Conventions & gotchas` の platform constraints 段落から `fairseq` への言及を消し、代わりに「fairseq / transformers は runtime にも lock にも無い」と書く。

- [ ] **Step 11: 全ゲートを回す**

Run: `uv run --all-extras poe check`
Expected: PASS（fmt-check / lint / type / test / lock-check / audit / security / deadcode）

`deadcode`（vulture）が `half_precision_available` などを未使用と誤検知した場合は、実際の呼び出し元（`vspeech/worker/vc.py:180`）を確認してから判断すること。

- [ ] **Step 12: commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "build!: drop fairseq and transformers from pyproject and uv.lock

オフライン工程 (convert / export) の依存は poe task の `uv run --with` が
一時環境で供給する。uv.lock から fairseq と transformers が消え、uv audit は
受容済みの torch 1 件のみになり、numpy の 1.23.5 ピンも外れる。"
```

---

## Task 9: 実機での回帰検証

自動ゲートでは捕まえられない部分を実機で確認する。**spec ① で 2 つの罠が実機でしか見つからなかった**（`torch.compile` の `TritonMissing`、stale な golden）ことを踏まえ、必ず通すこと。

**Files:** なし（検証のみ）

**Interfaces:**
- Consumes: 全タスクの成果
- Produces: なし

- [ ] **Step 1: change_voice の音声回帰ゲート**

Run:
```bash
VSPEECH_RVC_GOLDEN_CONFIG=<rvc worker の toml> uv run --all-extras \
  pytest tests/test_change_voice_golden.py -v
```
Expected: PASS。しきい値は `corr ≥ 0.999` / `SNR ≥ 40 dB`（据え置き）

**落ちた場合、golden を作り直してはいけない。** spec ① の実測は `corr 0.99998675` / `SNR 44.59 dB` で 4.6 dB の余裕があった。ここを割るということは export か runtime が壊れている。まず `tests/test_hubert_equivalence.py` の fp16 側が通っているかを見る。通っているなら `_extract_hubert_feats` の dtype / `interpolate` / `infer` への受け渡しを疑う。

注: この config の `[rvc] hubert_model_file` は ONNX を含む**資産ディレクトリ**を指していること。

- [ ] **Step 2: 等価ゲート（両精度）**

Run:
```bash
VSPEECH_HUBERT_ASSET_DIR=./hubert_contentvec \
VSPEECH_HUBERT_GOLDEN_DIR=./hubert_golden \
VSPEECH_RVC_GOLDEN_CONFIG=<rvc worker の toml> \
uv run --all-extras pytest tests/test_hubert_equivalence.py tests/test_change_voice_golden.py -v
```
Expected: PASS 5 件

- [ ] **Step 3: vc worker を実機で起動して耳で確認**

Run: `uv run python -m vspeech --config <rvc を有効にした config.toml>`

確認すること:
- warmup が例外を出さずに `vc worker warmed up` を出す（`vspeech/worker/vc.py:220`）
- 実際に喋って、声質・レイテンシに知覚できる劣化がないこと
- ログに `TritonMissing` / `CUDAExecutionProvider` 不在の警告が出ないこと

これは自動ゲートに含めない（VAD v6 移行・spec ① と同じ運用）。

- [ ] **Step 4: 起動レイテンシの目視確認（advisory）**

vc worker の warmup が数分かかるようなら、ORT が fp16 グラフのカーネルを autotune している。初回だけなら想定内。毎回なら `create_session` の `cudnn_conv_algo_search` を疑う。

- [ ] **Step 5: 最終 commit（あれば）**

実機検証で修正が入った場合のみ commit する。無ければ何もしない。

---

## 完了条件

これを全て満たした時点で spec ② は完了し、spec ③（`requires-python` 引き上げ）が着手可能になる。

- [ ] `uv.lock` に `fairseq` と `transformers` のエントリが無い
- [ ] `uv audit` が torch の 1 件（`GHSA-rrmf-rvhw-rf47`、受容済み）だけを報告する
- [ ] `uv.lock` の numpy が 1.24 以上（cp312 wheel が存在する版）
- [ ] `uv run --all-extras poe check` が green
- [ ] `tests/test_hubert_equivalence.py` が fp32 2 件 + fp16 2 件で PASS
- [ ] `tests/test_change_voice_golden.py` が**再ベースライン無し**で PASS
- [ ] `tests/test_forbidden_imports.py` が PASS
- [ ] 実機の耳チェックで劣化なし
