import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from torch.nn import functional

from vspeech.config import RvcConfig
from vspeech.lib.cuda_util import Device
from vspeech.lib.onnx_session import create_session
from vspeech.lib.pitch_extract import pitch_extract
from vspeech.lib.resample import make_resampler
from vspeech.logger import logger

# RVC runs HuBERT and pitch extraction on a fixed 16kHz mono signal; the input
# is resampled to this rate before feature extraction, so any pad math must use
# it -- not the remote's original capture rate.
HUBERT_SAMPLE_RATE = 16000

# Output names of the ONNX graph. scripts/export_hubert_onnx.py exports under these
# names and mapping.json records how they correspond to (emb_output_layer,
# use_final_proj). Real RVC models only come in two flavours: v1 = (9, True) and
# v2 = (12, False).
FEATS_L9_PROJ = "feats_l9_proj"
FEATS_L12_RAW = "feats_l12_raw"


_REEXPORT_HINT = "scripts/export_hubert_onnx.py で再 export してください"


def parse_output_names(mapping: dict[str, Any]) -> dict[tuple[int, bool], str]:
    """Expand mapping.json's `outputs` into (emb_output_layer, use_final_proj) ->
    output name.

    The runtime never guesses the layer index; it trusts only the table read here.
    Silently accepting a corrupt or stale mapping.json would route voice conversion to
    the output of the wrong layer (the failure mode this module exists to forbid), so
    any deviation from the expected shape stops with a ValueError.
    """
    outputs = mapping.get("outputs")
    if not outputs:
        raise ValueError(f"mapping.json に 'outputs' がありません。{_REEXPORT_HINT}")
    if not isinstance(outputs, list):
        raise ValueError(
            "mapping.json の 'outputs' は list である必要があります"
            f"（実際: {type(outputs).__name__}）。{_REEXPORT_HINT}"
        )
    result: dict[tuple[int, bool], str] = {}
    for entry in outputs:
        try:
            layer = entry["layer"]
            use_final_proj = entry["use_final_proj"]
            name = entry["name"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"mapping.json の 'outputs' の要素が壊れています: {entry!r}。"
                f"{_REEXPORT_HINT}"
            ) from e
        # bool is a subclass of int, so isinstance(True, int) is True. Reject bool
        # first so JSON's true/false is never accepted as a layer number.
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError(
                "mapping.json の 'layer' は int である必要があります"
                f"（実際: {layer!r}）。{_REEXPORT_HINT}"
            )
        if not isinstance(use_final_proj, bool):
            raise ValueError(
                "mapping.json の 'use_final_proj' は bool である必要があります"
                f"（実際: {use_final_proj!r}）。{_REEXPORT_HINT}"
            )
        if not isinstance(name, str):
            raise ValueError(
                f"mapping.json の 'name' は str である必要があります"
                f"（実際: {name!r}）。{_REEXPORT_HINT}"
            )
        key = (layer, use_final_proj)
        if key in result:
            raise ValueError(
                f"mapping.json の 'outputs' にキー {key} の重複があります。"
                f"{_REEXPORT_HINT}"
            )
        result[key] = name
    return result


@dataclass
class HubertSession:
    """The runtime representation of the ONNX-ified ContentVec.

    `final_proj` is baked into the graph, so the runtime does not carry it. mapping.json
    is the single source of truth for which output corresponds to which
    (emb_output_layer, use_final_proj).
    """

    session: InferenceSession
    output_names: dict[tuple[int, bool], str]
    is_half: bool


def _torch_device(device: Device) -> torch.device:
    """The one place the torch-free `Device` becomes a `torch.device` (ADR-0078).

    torch lives inside the conversion path only; everything upstream of here -- device
    resolution, session creation, the whisper worker -- speaks `Device` and never
    imports torch. `Device("cuda")` has an index of None, which torch would keep as an
    unindexed device; pin it to 0 so the io_binding path always has a real ordinal.
    """
    if device.type == "cuda":
        return torch.device("cuda", device.index if device.index is not None else 0)
    return torch.device("cpu")


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
    """Bind a torch CUDA buffer to an ORT input with zero copy.

    The caller **must keep a reference** to the return value: making the tensor
    contiguous may produce a new tensor, and the lifetime of the bound pointer depends
    on it.
    """
    tensor = tensor.contiguous()
    device = tensor.device
    # `device.index` is None for a bare torch.device("cuda"), so `else 0` is a
    # real branch, though ty narrows index to non-optional and marks it
    # unreachable (ty check still exits 0). Keep it. (Same in extract_features
    # / infer.)
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
    except Exception as e:  # noqa: BLE001 - any failure must still return output
        # dlpack zero-copy is the fast path; on any failure fall back to a numpy
        # copy so inference still returns. Warn, so a broad except here can't turn
        # a real dlpack bug into a silent slow path.
        logger.warning("dlpack transfer failed; using numpy fallback: %s", e)
        return torch.tensor(ort_output.numpy(), device=device)


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
        # `bound` owns the lifetime of the bound pointer. Do not drop it before run
        # returns.
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

    if is_half:
        input_feed = {
            "feats": feats.cpu().numpy().astype(np.float16),
            "p_len": pitch_length.cpu().numpy().astype(np.int64),
            "sid": sid.cpu().numpy().astype(np.int64),
        }
    else:
        input_feed = {
            "feats": feats.cpu().numpy().astype(np.float32),
            "p_len": pitch_length.cpu().numpy().astype(np.int64),
            "sid": sid.cpu().numpy().astype(np.int64),
        }
    if pitch is not None and pitchf is not None:
        input_feed.update(
            {
                "pitch": pitch.cpu().numpy().astype(np.int64),
                "pitchf": pitchf.cpu().numpy().astype(np.float32),
            }
        )
    audio1 = cast(
        list,
        session.run(
            output_names=["audio"],
            input_feed=input_feed,
        ),
    )
    return torch.tensor(np.array(audio1), device=device)


def _select_onnx_file(
    asset_dir: Path, device: Device, is_half: bool
) -> tuple[Path, bool]:
    """Return the ONNX file to use and whether it is fp16.

    An fp16 graph is effectively unusable on CPUExecutionProvider, so CPU always gets
    fp32.
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


def load_hubert_model(file_name: Path, device: Device, is_half: bool) -> HubertSession:
    """Load the ONNX-ified ContentVec asset directory (the output of
    scripts/export_hubert_onnx.py)."""
    asset_dir = file_name.expanduser()
    model_file, half = _select_onnx_file(asset_dir, device, is_half)
    session = create_session(model_file, device)
    with open(asset_dir / "mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    return HubertSession(
        session=session,
        output_names=parse_output_names(mapping),
        is_half=half,
    )


def _pad_input_to_block(voice_frames: bytes) -> np.ndarray:
    input_sound = np.frombuffer(voice_frames, dtype="int16")
    input_size = input_sound.shape[0]
    if input_size % 128 != 0:
        input_size = input_size + (128 - (input_size % 128))
    audio = input_sound.astype(np.float32) / 32768.0
    if audio.shape[0] < input_size:
        audio = np.concatenate([np.zeros([input_size - audio.shape[0]]), audio])
    return audio


def _to_hubert_rate(audio: NDArray[np.floating], src_rate: int) -> NDArray[np.float32]:
    """Bring one utterance to HUBERT_SAMPLE_RATE through the in-house polyphase FIR.

    The same filter the device boundaries use (ADR-0073), reused here so the repo holds
    one resampler instead of a second, weaker one from torchaudio (ADR-0082).

    `resample_full`, not `process`: an utterance is a self-contained buffer, so the
    filter tail is flushed and the group delay removed -- `process` would leave the last
    few milliseconds inside the filter. `make_resampler` returns None when the rates
    already match, and the buffer then passes through untouched, as it did before.

    Takes any float width because `_pad_input_to_block` returns float64 whenever it
    prepends zeros; the resampler is a float32 pipeline, so the cast happens here.
    """
    audio32 = np.ascontiguousarray(audio, dtype=np.float32)
    resampler = make_resampler(src_rate, HUBERT_SAMPLE_RATE)
    if resampler is None:
        return audio32
    return resampler.resample_full(audio32)


def _quality_padding(
    audio: torch.Tensor,
    rvc_config: RvcConfig,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    # `audio` is already at HUBERT_SAMPLE_RATE; pad each side by `repeat` whole
    # reflections for extra model context and report the matching output-side
    # pad (at target_sample_rate) for _postprocess to trim.
    repeat = rvc_config.quality.value
    t_pad = repeat * (audio.shape[1] - 1)
    t_pad_tgt = round(t_pad * target_sample_rate / HUBERT_SAMPLE_RATE)
    audio_pad = functional.pad(audio, (t_pad, t_pad), mode="reflect").squeeze(0)
    return audio_pad, t_pad_tgt


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


def _select_pitch(
    audio_pad: torch.Tensor,
    rvc_config: RvcConfig,
    f0_enabled: bool,
    p_len: int,
    device: torch.device,
    f0_session: InferenceSession | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not f0_enabled:
        return None, None
    pitch, pitchf = pitch_extract(
        audio_pad,
        rvc_config.f0_up_key,
        16000,
        rvc_config.window,
        f0_extractor=rvc_config.f0_extractor_type,
        f0_filter_radius=rvc_config.f0_filter_radius,
        f0_session=f0_session,
        silence_front=0,
    )
    pitch = pitch[:p_len]
    pitchf = pitchf[:p_len]
    pitch_t = torch.tensor(pitch, device=device).unsqueeze(0).long()
    pitchf_t = torch.tensor(pitchf, device=device, dtype=torch.float).unsqueeze(0)
    return pitch_t, pitchf_t


def _is_model_half(session: InferenceSession) -> bool:
    return session.get_inputs()[0].type != "tensor(float)"


def _align_pitch_to_feats(
    pitch: torch.Tensor | None,
    pitchf: torch.Tensor | None,
    feats_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if pitch is not None and pitchf is not None:
        return pitch[:, -feats_len:], pitchf[:, -feats_len:]
    return pitch, pitchf


def _to_int16(audio: torch.Tensor) -> torch.Tensor:
    """Scale a decoder waveform (~[-1, 1]) to int16, saturating out of range.

    RVC/vocoder output is not guaranteed to stay within [-1, 1] (pitch-shifted
    or loud segments overshoot). Clamp BEFORE the int16 cast: an unclamped cast
    wraps modulo 2**16, turning a >+1.0 peak into a large negative sample -- a
    loud click. Clamping saturates to the rail instead.
    """
    return torch.clamp(audio * 32767.5, -32768.0, 32767.0).to(dtype=torch.int16)


def _postprocess(audio1: torch.Tensor, t_pad_tgt: int) -> NDArray[np.int16]:
    if t_pad_tgt != 0:
        audio1 = audio1[t_pad_tgt : -1 * t_pad_tgt]
    return audio1.detach().cpu().numpy()


def change_voice(
    voice_frames: bytes,
    rvc_config: RvcConfig,
    voice_sample_rate: int,
    target_sample_rate: int,
    device: Device,
    emb_output_layer: int,
    use_final_proj: bool,
    hubert_model: HubertSession,
    session: InferenceSession,
    f0_enabled: bool,
    f0_session: InferenceSession | None,
) -> NDArray[np.int16]:
    vc_start_time = time.time()
    torch_dev = _torch_device(device)
    audio_np = _to_hubert_rate(_pad_input_to_block(voice_frames), voice_sample_rate)
    audio = (
        torch.from_numpy(audio_np)
        .to(device=torch_dev, dtype=torch.float32)
        .unsqueeze(0)
    )

    audio_pad, t_pad_tgt = _quality_padding(audio, rvc_config, target_sample_rate)
    sid = torch.tensor(0, device=torch_dev).unsqueeze(0).long()

    feats = _extract_hubert_feats(
        hubert_model=hubert_model,
        audio_pad=audio_pad,
        device=torch_dev,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )

    p_len = audio_pad.shape[0] // rvc_config.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
    pitch, pitchf = _select_pitch(
        audio_pad=audio_pad,
        rvc_config=rvc_config,
        f0_enabled=f0_enabled,
        p_len=p_len,
        device=torch_dev,
        f0_session=f0_session,
    )

    vc_end_time = time.time()
    logger.info(
        "rvc: pitch size adjusted: elapsed time: %s", vc_end_time - vc_start_time
    )

    is_model_half = _is_model_half(session)
    feats_len = feats.shape[1]
    pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
    p_len_tensor = torch.tensor([feats_len], device=torch_dev).long()

    with torch.inference_mode():
        audio1 = _to_int16(
            infer(
                session=session,
                is_half=is_model_half,
                feats=feats,
                pitch_length=p_len_tensor,
                pitch=pitch,
                pitchf=pitchf,
                sid=sid,
            )[0]
        )

    del feats, p_len_tensor

    vc_end_time = time.time()
    logger.info("rvc: inferred: elapsed time: %s", vc_end_time - vc_start_time)

    result = _postprocess(audio1, t_pad_tgt)
    del pitch, pitchf, sid
    return result
