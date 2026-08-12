"""`change_voice` end to end on a CPU device, on synthetic ONNX graphs.

The CUDA path is covered on real hardware by scripts/stream_vc_baseline.py and
tests/test_change_voice_golden.py, both of which skip without a GPU and a machine-local
config. Nothing covered the other branch of `_run_on_device` end to end, and ADR-0081
made that branch the shape the CUDA one was rewritten towards: the same numpy inputs,
`session.run` instead of an io_binding. So it is worth an actual conversion rather than
an assertion about the code.

The graphs are built here with onnx's helper API and are deliberately trivial -- the
subject is the plumbing (dtypes, shapes, the pitch/feature alignment, the int16
conversion), not the audio.
"""

import json
from typing import Any
from typing import cast

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx import helper
from onnxruntime import InferenceSession

from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig
from vspeech.lib.cuda_util import Device

HUBERT_DIM = 256
# 320 input samples per HuBERT frame at 16kHz, and the decoder emits `_UPSAMPLE` output
# samples per (2x upsampled) frame -- the same relation the real 40kHz model has.
_UPSAMPLE = 200


def _hubert_asset(path):
    """source (1, N) float32 -> feats_l9_proj (1, N//320, 256), plus the mapping.json."""
    source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, "N"])
    out9 = helper.make_tensor_value_info(
        "feats_l9_proj", TensorProto.FLOAT, [1, "T", HUBERT_DIM]
    )
    # Average-pool the waveform into frames of 320, then broadcast to 256 channels.
    pool = helper.make_node(
        "AveragePool",
        ["src3"],
        ["frames"],
        kernel_shape=[320],
        strides=[320],
        auto_pad="VALID",
    )
    nodes = [
        helper.make_node("Unsqueeze", ["source", "axis1"], ["src3"]),
        pool,
        helper.make_node("Transpose", ["frames"], ["framesT"], perm=[0, 2, 1]),
        helper.make_node("Concat", ["framesT"] * HUBERT_DIM, ["feats_l9_proj"], axis=2),
    ]
    axis1 = helper.make_tensor("axis1", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(nodes, "tiny_hubert", [source], [out9], [axis1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path / "hubert_fp32.onnx"))
    with open(path / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "outputs": [
                    {"name": "feats_l9_proj", "layer": 9, "use_final_proj": True}
                ]
            },
            f,
        )
    return path


def _decoder(path):
    """feats/p_len/pitch/pitchf/sid -> audio (T*_UPSAMPLE,) float32.

    `feats` is declared first so `_is_model_half` reads it, and fp32 so the CPU path is
    the fp32 one (an fp16 graph is unusable on CPUExecutionProvider).
    """
    feats = helper.make_tensor_value_info(
        "feats", TensorProto.FLOAT, [1, "T", HUBERT_DIM]
    )
    p_len = helper.make_tensor_value_info("p_len", TensorProto.INT64, [1])
    pitch = helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "T"])
    pitchf = helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "T"])
    sid = helper.make_tensor_value_info("sid", TensorProto.INT64, [1])
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, ["A"])
    # audio = repeat(mean(feats, axis=-1) + pitchf/10000, _UPSAMPLE), flattened.
    nodes = [
        helper.make_node("ReduceMean", ["feats", "axes2"], ["fmean"], keepdims=0),
        helper.make_node("Div", ["pitchf", "tenk"], ["pf"]),
        helper.make_node("Add", ["fmean", "pf"], ["frame"]),
        helper.make_node("Unsqueeze", ["frame", "axes2"], ["frame3"]),
        helper.make_node("Tile", ["frame3", "reps"], ["tiled"]),
        helper.make_node("Reshape", ["tiled", "flat"], ["audio"]),
    ]
    initializers = [
        helper.make_tensor("axes2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("tenk", TensorProto.FLOAT, [1], [10000.0]),
        helper.make_tensor("reps", TensorProto.INT64, [3], [1, 1, _UPSAMPLE]),
        helper.make_tensor("flat", TensorProto.INT64, [1], [-1]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_decoder",
        [feats, p_len, pitch, pitchf, sid],
        [audio],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class _FakeF0Session:
    """A waveform-input rmvpe.onnx stand-in returning a constant voiced f0."""

    def __init__(self, hz: float = 220.0):
        self.hz = hz
        self.calls: list[dict[str, Any]] = []

    def run(self, output_names, input_feed):
        self.calls.append(dict(input_feed))
        n = input_feed["waveform"].shape[-1] // 160
        return [np.full((1, n), self.hz, dtype=np.float32)]


@pytest.fixture
def cpu_models(tmp_path):
    from vspeech.lib.rvc import load_hubert_model

    asset = _hubert_asset(tmp_path)
    hubert = load_hubert_model(asset, Device("cpu"), is_half=True)
    session = InferenceSession(
        str(_decoder(tmp_path / "decoder.onnx")), providers=["CPUExecutionProvider"]
    )
    return hubert, session


def _pcm(n: int = 16000) -> bytes:
    t = np.arange(n, dtype=np.float32) / 16000.0
    return (np.sin(2 * np.pi * 220.0 * t) * 12000).astype(np.int16).tobytes()


def test_change_voice_converts_on_a_cpu_device(cpu_models):
    """The whole utterance path runs on `Device("cpu")` and produces int16 audio.

    `is_half=True` is requested on purpose: CPU must fall back to the fp32 HuBERT graph
    (an fp16 graph is unusable on CPUExecutionProvider), and the fp32 features must then
    reach an fp32 decoder without a dtype mismatch -- ORT rejects the feed outright if
    `infer` gets that wrong, which is the failure this guards.
    """
    from vspeech.lib.rvc import change_voice

    hubert, session = cpu_models
    assert hubert.is_half is False  # CPU downgraded it, as _select_onnx_file promises

    f0_session = _FakeF0Session()
    out = change_voice(
        voice_frames=_pcm(),
        rvc_config=RvcConfig(f0_extractor_type=F0ExtractorType.rmvpe),
        voice_sample_rate=16000,
        target_sample_rate=40000,
        device=Device("cpu"),
        emb_output_layer=9,
        use_final_proj=True,
        hubert_model=hubert,
        session=session,
        f0_enabled=True,
        f0_session=cast(InferenceSession, f0_session),
    )
    assert out.dtype == np.int16
    assert out.ndim == 1
    # 16000 samples -> 50 HuBERT frames -> 100 upsampled frames -> 100*_UPSAMPLE samples
    assert out.shape[0] == 100 * _UPSAMPLE
    assert np.abs(out).max() > 0  # the graph's output actually reached the caller
    assert len(f0_session.calls) == 1


def test_change_voice_on_cpu_without_f0(cpu_models):
    """f0_enabled=False leaves pitch/pitchf unbound, which the CPU feed must honour.

    The decoder graph still declares them, so ORT would raise on a missing input if
    `infer` fed them anyway -- but a real f0-less RVC model does not declare them, and
    this pins that the two are dropped together rather than one of them.
    """
    from vspeech.lib.rvc import _select_pitch
    from vspeech.lib.rvc import infer

    hubert, session = cpu_models
    assert _select_pitch(
        audio_pad=np.zeros(16000, dtype=np.float32),
        rvc_config=RvcConfig(),
        f0_enabled=False,
        p_len=10,
        f0_session=None,
    ) == (None, None)

    feats = np.zeros((1, 4, HUBERT_DIM), dtype=np.float32)
    # The exact class onnxruntime raises for a required input that is absent from the
    # feed: `_validate_input` checks it in Python before the graph runs, so it is a
    # plain ValueError rather than one of ORT's native error types. Pinned rather than
    # left as `Exception`, which `match="pitch"` alone would let a TypeError from a
    # mistyped keyword satisfy.
    with pytest.raises(ValueError, match=r"Required inputs.*pitch"):
        infer(
            is_half=False,
            session=session,
            device=Device("cpu"),
            feats=feats,
            pitch_length=np.array([4], dtype=np.int64),
            pitch=None,
            pitchf=None,
            sid=np.zeros(1, dtype=np.int64),
        )


def test_cpu_and_cuda_paths_feed_the_same_arrays(cpu_models):
    """`infer` builds one input feed and only the transport differs (ADR-0081).

    Pinned by feeding the CPU branch and checking the dtypes ORT accepted: int64 for
    p_len/pitch/sid, float32 for pitchf, and the decoder's own precision for feats. A
    dtype regression here is invisible on CUDA (the io_binding would just bind the wrong
    element type) until inference returns nonsense.
    """
    from vspeech.lib.rvc import infer

    _hubert, session = cpu_models
    types = {i.name: i.type for i in session.get_inputs()}
    assert types == {
        "feats": "tensor(float)",
        "p_len": "tensor(int64)",
        "pitch": "tensor(int64)",
        "pitchf": "tensor(float)",
        "sid": "tensor(int64)",
    }
    t = 6
    out = infer(
        is_half=False,
        session=session,
        device=Device("cpu"),
        feats=np.full((1, t, HUBERT_DIM), 0.25, dtype=np.float64),  # widened on purpose
        pitch_length=np.array([t], dtype=np.int32),  # widened on purpose
        pitch=np.full((1, t), 100, dtype=np.int32),  # widened on purpose
        pitchf=np.full((1, t), 220.0, dtype=np.float64),  # widened on purpose
        sid=np.zeros(1, dtype=np.int64),
    )
    assert out.shape == (1, t * _UPSAMPLE)
    np.testing.assert_allclose(out[0, 0], 0.25 + 220.0 / 10000.0, rtol=1e-5)
