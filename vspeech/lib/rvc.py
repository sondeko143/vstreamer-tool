import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from onnxruntime import OrtValue

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


def _ort_device_id(device: Device) -> int:
    """The ordinal onnxruntime wants for `device`.

    `Device("cuda")` has an `index` of None, and ORT needs a real ordinal; 0 is the same
    default `create_session` applies, so the values land on the card the session runs on.
    """
    return device.index if device.index is not None else 0


def _run_on_device(
    session: InferenceSession,
    device: Device,
    input_feed: dict[str, NDArray[Any]],
    output_name: str,
) -> NDArray[Any]:
    """Run `session` for a single output, binding values with onnxruntime's own OrtValue.

    On CUDA each input is copied into device memory as an `OrtValue` and attached with
    `bind_ortvalue_input`, and the output is allocated on the device and read back once
    (ADR-0081). `bound` keeps every input alive until the run returns: onnxruntime does
    not document whether the binding owns the value it is handed, and the cost of one
    list is nothing next to a use-after-free on device memory. Do not "simplify" it into
    a temporary inside the `bind_ortvalue_input` call.

    [Open, deferred 2026-08-12 -- outside ADR-0080's scope] That last sentence is stated
    more strongly than the evidence supports. What is established is only that the
    ownership is undocumented; "a use-after-free on device memory" is the consequence that
    *would* follow if the binding held no reference, not something anyone observed, and
    the C++ IOBinding is likely to hold one. The list stays either way (it costs nothing
    and is correct under both readings), but the hazard should not be repeated elsewhere
    as an established fact until someone has read onnxruntime's C++ side. Deferred because
    settling it changes no code here.

    Everything that is not CUDA takes the plain numpy `session.run` the CPU path always
    took.
    """
    if device.type != "cuda":
        result = cast(
            list, session.run(output_names=[output_name], input_feed=input_feed)
        )
        return np.asarray(result[0])
    io_binding = session.io_binding()
    device_id = _ort_device_id(device)
    bound: list[OrtValue] = []
    for name, value in input_feed.items():
        ort_value = OrtValue.ortvalue_from_numpy(value, "cuda", device_id)
        io_binding.bind_ortvalue_input(name, ort_value)
        bound.append(ort_value)
    io_binding.bind_output(output_name, "cuda", device_id=device_id)
    session.run_with_iobinding(io_binding)
    return io_binding.get_outputs()[0].numpy()


def extract_features(
    model: HubertSession,
    feats: NDArray[np.floating[Any]],
    device: Device,
    emb_output_layer: int = 9,
    use_final_proj: bool = True,
) -> NDArray[np.floating[Any]]:
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

    source = np.ascontiguousarray(
        feats, dtype=np.float16 if model.is_half else np.float32
    )
    return _run_on_device(model.session, device, {"source": source}, output_name)


def infer(
    is_half: bool,
    session: InferenceSession,
    device: Device,
    feats: NDArray[np.floating[Any]],
    pitch_length: NDArray[np.int64],
    pitch: NDArray[np.int64] | None,
    pitchf: NDArray[np.float32] | None,
    sid: NDArray[np.int64],
) -> NDArray[np.floating[Any]]:
    """Run the RVC decoder once and return its waveform batched as `(1, N)`."""
    input_feed: dict[str, NDArray[Any]] = {
        "feats": np.ascontiguousarray(
            feats, dtype=np.float16 if is_half else np.float32
        ),
        "p_len": np.ascontiguousarray(pitch_length, dtype=np.int64),
        "sid": np.ascontiguousarray(sid, dtype=np.int64),
    }
    if pitch is not None and pitchf is not None:
        input_feed["pitch"] = np.ascontiguousarray(pitch, dtype=np.int64)
        input_feed["pitchf"] = np.ascontiguousarray(pitchf, dtype=np.float32)
    audio1 = _run_on_device(session, device, input_feed, "audio")
    return audio1[np.newaxis, ...]


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

    Any float width is accepted because `_pad_input_to_block` returns float64 whenever
    it prepends zeros. The cast is here rather than left to the resampler because the
    passthrough branch never reaches one: `process` coerces to float32 itself, so only
    a rate that already matches 16kHz would otherwise escape with the caller's dtype
    and break the float32 return type.
    """
    audio32 = np.ascontiguousarray(audio, dtype=np.float32)
    resampler = make_resampler(src_rate, HUBERT_SAMPLE_RATE)
    if resampler is None:
        return audio32
    return resampler.resample_full(audio32)


def _quality_padding(
    audio: NDArray[np.float32],
    rvc_config: RvcConfig,
    target_sample_rate: int,
) -> tuple[NDArray[np.float32], int]:
    """Reflect-pad one utterance and report the matching output-side pad.

    `audio` is already at HUBERT_SAMPLE_RATE; pad each side by `repeat` whole
    reflections for extra model context and report the matching output-side pad (at
    target_sample_rate) for _postprocess to trim.

    numpy's `reflect` is the same reflection torch's `functional.pad(mode="reflect")`
    applies -- neither repeats the edge sample -- and `RvcQuality` only takes 0 and 1, so
    the pad width is either 0 or n-1 and never exceeds what either accepts (verified
    element-wise for both widths at several lengths).
    """
    repeat = rvc_config.quality.value
    t_pad = repeat * (audio.shape[0] - 1)
    t_pad_tgt = round(t_pad * target_sample_rate / HUBERT_SAMPLE_RATE)
    return np.pad(audio, (t_pad, t_pad), mode="reflect"), t_pad_tgt


def _extract_hubert_feats(
    hubert_model: HubertSession,
    audio_pad: NDArray[np.float32],
    device: Device,
    emb_output_layer: int,
    use_final_proj: bool,
) -> NDArray[np.floating[Any]]:
    """HuBERT features for one analysis window, upsampled 2x along time.

    The upsample used to be `functional.interpolate(..., scale_factor=2)` in nearest
    mode, which is elementwise duplication by definition and produces exactly what
    `np.repeat(..., 2)` does (verified element-wise on fp16 and fp32, CPU and CUDA).
    """
    feats = audio_pad
    if feats.ndim == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.ndim == 1, feats.ndim  # nosec B101 - internal shape invariant
    feats = feats.reshape(1, -1)
    features = extract_features(
        model=hubert_model,
        feats=feats,
        device=device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    return np.repeat(features, 2, axis=1)


def _select_pitch(
    audio_pad: NDArray[np.float32],
    rvc_config: RvcConfig,
    f0_enabled: bool,
    p_len: int,
    f0_session: InferenceSession | None,
) -> tuple[NDArray[np.int64] | None, NDArray[np.float32] | None]:
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
    return (
        pitch[:p_len].astype(np.int64)[np.newaxis, :],
        pitchf[:p_len].astype(np.float32)[np.newaxis, :],
    )


def _is_model_half(session: InferenceSession) -> bool:
    return session.get_inputs()[0].type != "tensor(float)"


def _align_pitch_to_feats(
    pitch: NDArray[np.int64] | None,
    pitchf: NDArray[np.float32] | None,
    feats_len: int,
) -> tuple[NDArray[np.int64] | None, NDArray[np.float32] | None]:
    if pitch is not None and pitchf is not None:
        return pitch[:, -feats_len:], pitchf[:, -feats_len:]
    return pitch, pitchf


def _to_int16(audio: NDArray[np.floating[Any]]) -> NDArray[np.int16]:
    """Scale a decoder waveform (~[-1, 1]) to int16, clipping out of range.

    RVC/vocoder output is not guaranteed to stay within [-1, 1] (pitch-shifted or loud
    segments overshoot). Clip BEFORE the int16 cast: an unclipped cast wraps modulo
    2**16, turning a >+1.0 peak into a large negative sample -- a loud click.

    The arithmetic stays in the decoder output's own dtype, which is what the torch
    version did (`torch.clamp(x * 32767.5, ...).to(torch.int16)` on a half tensor
    computes and clamps in half). Both were measured to agree bit for bit on 2M samples,
    on CPU and CUDA, which is what keeps the ADR-0080 bit-exactness gate meaningful.

    [Open, deferred 2026-08-12 -- pre-existing, NOT introduced by ADR-0081] The clip does
    not actually saturate when the decoder emits float16, which every fp16 RVC model
    does. 32767.0 is not representable in float16 and rounds to 32768.0, so the upper
    bound is 32768.0 and a sample at or above +1.0 casts to **-32768** -- the very
    sign-flipped click this function exists to prevent. Measured: torch and numpy do this
    identically, so it is not a regression, and neither the streaming baseline (peak
    24288) nor the utterance golden (peak 18128) reaches the rail. Fixing it means
    computing in float32, which shifts in-range samples by up to 1 LSB and therefore
    cannot ride along with a change whose acceptance criterion is bit equality; it needs
    its own baseline re-capture and ear check.
    """
    return np.clip(audio * 32767.5, -32768.0, 32767.0).astype(np.int16)


def _postprocess(audio1: NDArray[np.int16], t_pad_tgt: int) -> NDArray[np.int16]:
    if t_pad_tgt != 0:
        audio1 = audio1[t_pad_tgt : -1 * t_pad_tgt]
    return audio1


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
    audio = _to_hubert_rate(_pad_input_to_block(voice_frames), voice_sample_rate)

    audio_pad, t_pad_tgt = _quality_padding(audio, rvc_config, target_sample_rate)
    sid = np.zeros(1, dtype=np.int64)

    feats = _extract_hubert_feats(
        hubert_model=hubert_model,
        audio_pad=audio_pad,
        device=device,
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
        f0_session=f0_session,
    )

    vc_end_time = time.time()
    logger.info(
        "rvc: pitch size adjusted: elapsed time: %s", vc_end_time - vc_start_time
    )

    is_model_half = _is_model_half(session)
    feats_len = feats.shape[1]
    pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
    p_len_array = np.array([feats_len], dtype=np.int64)

    audio1 = _to_int16(
        infer(
            session=session,
            is_half=is_model_half,
            device=device,
            feats=feats,
            pitch_length=p_len_array,
            pitch=pitch,
            pitchf=pitchf,
            sid=sid,
        )[0]
    )

    vc_end_time = time.time()
    logger.info("rvc: inferred: elapsed time: %s", vc_end_time - vc_start_time)

    return _postprocess(audio1, t_pad_tgt)
