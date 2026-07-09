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
from safetensors.torch import load_file
from torch.nn import functional
from transformers import HubertModel

from vspeech.config import RvcConfig
from vspeech.lib.pitch_extract import pitch_extract
from vspeech.logger import logger

# RVC runs HuBERT and pitch extraction on a fixed 16kHz mono signal; the input
# is resampled to this rate before feature extraction, so any pad math must use
# it -- not the remote's original capture rate.
HUBERT_SAMPLE_RATE = 16000


@dataclass
class HubertBundle:
    """変換済み ContentVec 資産の runtime 表現。

    `final_proj` は資産に含まれていれば読む。RVC モデルのメタデータ `useFinalProj`
    が真のときだけ使われるため、ロード時点では必須にできない（不在は正当な構成）。
    """

    model: HubertModel
    final_proj: torch.nn.Linear | None
    layer_offset: int


def create_session(model_file: Path, gpu_id: int) -> InferenceSession:
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(
            0,
            {
                "device_id": gpu_id,
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


def half_precision_available(id: int):
    try:
        gpuName = torch.cuda.get_device_name(id).upper()
        if (
            ("16" in gpuName and "V100" not in gpuName)
            or "P40" in gpuName.upper()
            or "1070" in gpuName
            or "1080" in gpuName
        ):
            return False
    except Exception as e:
        logger.warning("%s", e)
        return False

    return True


@lru_cache(maxsize=4)
def get_resampler(orig_freq: int, new_freq: int, device: torch.device):
    return T.Resample(orig_freq, new_freq, rolloff=0.99).to(device)


def extract_features(
    model: HubertBundle,
    feats: torch.Tensor,
    dev: torch.device,
    emb_output_layer: int = 9,
    use_final_proj: bool = True,
) -> torch.Tensor:
    with torch.inference_mode():
        # fairseq の padding_mask (True=パディング) と transformers の attention_mask
        # (1=有効) は意味が反転している。ここは常にパディング無しなので何も渡さない
        # (= 全フレーム有効) のが等価。
        outputs = model.model(input_values=feats.to(dev), output_hidden_states=True)
        hidden = outputs.hidden_states[emb_output_layer + model.layer_offset]
        if use_final_proj:
            if model.final_proj is None:
                raise RuntimeError(
                    "RVC モデルが useFinalProj=True を要求していますが、変換済み資産に "
                    "final_proj.safetensors がありません"
                )
            hidden = model.final_proj(hidden)
    return hidden


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

        def bind(name: str, tensor: torch.Tensor):
            tensor = tensor.contiguous()
            if tensor.dtype == torch.float16:
                element_type = np.float16
            elif tensor.dtype == torch.float32:
                element_type = np.float32
            elif tensor.dtype == torch.int64:
                element_type = np.int64
            else:
                raise ValueError(f"Unsupported dtype: {tensor.dtype}")

            io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=device.index if device.index is not None else 0,
                element_type=element_type,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
            return tensor

        tensors = []
        tensors.append(bind("feats", feats.half() if is_half else feats.float()))
        tensors.append(bind("p_len", pitch_length))
        tensors.append(bind("sid", sid))

        if pitch is not None and pitchf is not None:
            tensors.append(bind("pitch", pitch))
            tensors.append(bind("pitchf", pitchf))

        io_binding.bind_output(
            "audio", "cuda", device_id=device.index if device.index is not None else 0
        )
        session.run_with_iobinding(io_binding)
        ort_output = io_binding.get_outputs()[0]

        try:
            from torch.utils import dlpack

            try:
                dlp = ort_output._ortvalue.to_dlpack()
            except AttributeError:
                dlp = ort_output.to_dlpack()
            audio1 = dlpack.from_dlpack(dlp).clone()
        except Exception:
            audio1 = torch.tensor(ort_output.numpy(), device=device)

        return audio1.unsqueeze(0)

    # Fallback for CPU
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


def load_hubert_model(
    file_name: Path, device: torch.device, is_half: bool
) -> HubertBundle:
    """変換済み ContentVec 資産ディレクトリを読む（scripts/convert_hubert.py の出力）。"""
    asset_dir = file_name.expanduser()

    # torch.compile は使わない。旧 fairseq 実装では OptimizedModule が forward/__call__ しか
    # 包まないため `model.extract_features(...)` が素通しされ、コンパイルは一度も走っていなかった。
    # transformers 版は __call__ を呼ぶので本当にコンパイルされ、Triton の無い Windows/CUDA では
    # TritonMissing で落ちる。旧実装の実効挙動（eager）に合わせる。
    # 資産は scripts/convert_hubert.py が出力したローカルディレクトリで、Hub からは一切取得しない。
    # local_files_only=True でそれを強制する（B615 の revision ピンは Hub 取得時のみ意味を持つ）。
    model = HubertModel.from_pretrained(  # nosec B615 - local dir only, no Hub download
        asset_dir, local_files_only=True
    )
    model.eval()
    model = model.to(device)
    if is_half:
        model = model.half()

    with open(asset_dir / "mapping.json", encoding="utf-8") as f:
        layer_offset = int(json.load(f)["layer_offset"])

    final_proj: torch.nn.Linear | None = None
    final_proj_path = asset_dir / "final_proj.safetensors"
    if final_proj_path.exists():
        tensors = load_file(str(final_proj_path))
        weight = tensors["weight"]
        bias = tensors["bias"]
        final_proj = torch.nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            final_proj.weight.copy_(weight)
            final_proj.bias.copy_(bias)
        final_proj.eval()
        final_proj = final_proj.to(device)
        if is_half:
            final_proj = final_proj.half()

    return HubertBundle(
        model=cast(HubertModel, model),
        final_proj=final_proj,
        layer_offset=layer_offset,
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
    hubert_model: HubertBundle,
    audio_pad: torch.Tensor,
    device: torch.device,
    half_available: bool,
    emb_output_layer: int,
    use_final_proj: bool,
) -> torch.Tensor:
    feats = audio_pad
    if half_available:
        feats = feats.half()
    else:
        feats = feats.float()
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
    rmvpe_session: InferenceSession | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not f0_enabled:
        return None, None
    pitch, pitchf = pitch_extract(
        audio_pad,
        rvc_config.f0_up_key,
        16000,
        rvc_config.window,
        f0_extractor=rvc_config.f0_extractor_type,
        rmvpe_session=rmvpe_session,
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
    half_available: bool,
    rvc_config: RvcConfig,
    voice_sample_rate: int,
    target_sample_rate: int,
    device: torch.device,
    emb_output_layer: int,
    use_final_proj: bool,
    hubert_model: HubertBundle,
    session: InferenceSession,
    f0_enabled: bool,
    rmvpe_session: InferenceSession | None,
) -> NDArray[np.int16]:
    vc_start_time = time.time()
    audio_np = _pad_input_to_block(voice_frames)
    audio = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)

    resampler = get_resampler(voice_sample_rate, HUBERT_SAMPLE_RATE, device)
    audio = resampler(audio).unsqueeze(0)

    audio_pad, t_pad_tgt = _quality_padding(audio, rvc_config, target_sample_rate)
    sid = torch.tensor(0, device=device).unsqueeze(0).long()

    feats = _extract_hubert_feats(
        hubert_model=hubert_model,
        audio_pad=audio_pad,
        device=device,
        half_available=half_available,
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
        device=device,
        rmvpe_session=rmvpe_session,
    )

    vc_end_time = time.time()
    logger.info(
        "rvc: pitch size adjusted: elapsed time: %s", vc_end_time - vc_start_time
    )

    is_model_half = _is_model_half(session)
    feats_len = feats.shape[1]
    pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
    p_len_tensor = torch.tensor([feats_len], device=device).long()

    # 推論実行
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
    # torch.cuda.empty_cache()

    vc_end_time = time.time()
    logger.info("rvc: inferred: elapsed time: %s", vc_end_time - vc_start_time)

    result = _postprocess(audio1, t_pad_tgt)
    del pitch, pitchf, sid
    # torch.cuda.empty_cache()
    return result
