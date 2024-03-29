from pathlib import Path
from typing import Any
from typing import Optional
from typing import cast

import numpy as np
import torch
from fairseq import checkpoint_utils
from fairseq.models.hubert import HubertModel
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from torch.nn import functional
from torchaudio.functional import resample

from vspeech.config import RvcConfig
from vspeech.lib.pitch_extract import pitch_extract
from vspeech.logger import logger


def create_session(model_file: Path, gpu_id: int):
    providers = ["CPUExecutionProvider"]
    providers_options: list[dict[str, Any]] = [{}]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        providers_options.insert(0, {"device_id": gpu_id})
    return InferenceSession(
        str(model_file.expanduser()),
        providers=providers,
        provider_options=providers_options,
    )


def get_device(gpu_id: int):
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device("cpu")


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


def extract_features(
    model: HubertModel,
    feats: torch.Tensor,
    dev: torch.device,
    emb_output_layer: int = 9,
    use_final_proj: bool = True,
) -> torch.Tensor:
    padding_mask = torch.BoolTensor(feats.shape).to(dev).fill_(False)
    inputs = {
        "source": feats.to(dev),
        "padding_mask": padding_mask,
        "output_layer": emb_output_layer,
    }

    with torch.no_grad():
        logits = model.extract_features(**inputs)  # type: ignore
        if use_final_proj:
            feats = model.final_proj(logits[0])  # type: ignore
        else:
            feats = logits[0]
    return feats


def infer(
    is_half: bool,
    session: InferenceSession,
    feats: torch.Tensor,
    pitch_length: torch.Tensor,
    pitch: torch.Tensor | None,
    pitchf: torch.Tensor | None,
    sid: torch.Tensor,
):
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
        NDArray[np.float32],
        session.run(
            output_names=["audio"],
            input_feed=input_feed,
        ),
    )
    return torch.tensor(np.array(audio1))


def load_hubert_model(
    file_name: Path, device: torch.device, is_half: bool
) -> HubertModel:
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(file_name.expanduser())],
        suffix="",
    )
    model = cast(HubertModel, models[0])
    model.eval()

    model = model.to(device)
    if is_half:
        model = model.half()
    return model


def change_voice(
    voice_frames: bytes,
    half_available: bool,
    rvc_config: RvcConfig,
    voice_sample_rate: int,
    target_sample_rate: int,
    device: torch.device,
    emb_output_layer: int,
    use_final_proj: bool,
    hubert_model: HubertModel,
    session: InferenceSession,
    f0_enabled: bool,
    crepe_session: Optional[InferenceSession],
) -> NDArray[np.int16]:
    input_sound = np.frombuffer(voice_frames, dtype="int16")
    input_size = input_sound.shape[0]
    if input_size % 128 != 0:
        input_size = input_size + (128 - (input_size % 128))

    audio = input_sound.astype(np.float32) / 32768.0
    if audio.shape[0] < input_size:
        audio = np.concatenate([np.zeros([input_size]), audio])
    audio = torch.from_numpy(audio).to(device=device, dtype=torch.float32)

    audio = resample(audio, voice_sample_rate, 16000, rolloff=0.99)
    audio = audio.unsqueeze(0)

    repeat = rvc_config.quality.value
    quality_padding_sec = (repeat * (audio.shape[1] - 1)) / voice_sample_rate
    t_pad = round(voice_sample_rate * quality_padding_sec)
    t_pad_tgt = round(target_sample_rate * quality_padding_sec)
    audio_pad = functional.pad(audio, (t_pad, t_pad), mode="reflect").squeeze(0)
    sid = 0
    sid = torch.tensor(sid, device=device).unsqueeze(0).long()

    f0_up_key = rvc_config.f0_up_key

    feats = audio_pad
    if half_available:
        feats = feats.half()
    else:
        feats = feats.float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)

    padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)
    feats = extract_features(
        model=hubert_model,
        feats=feats,
        dev=device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )

    feats = cast(
        torch.Tensor,
        functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1),
    )

    # ピッチサイズ調整
    p_len = audio_pad.shape[0] // rvc_config.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
    if f0_enabled:
        pitch, pitchf = pitch_extract(
            audio_pad,
            f0_up_key,
            voice_sample_rate,
            rvc_config.window,
            f0_extractor=rvc_config.f0_extractor_type,
            crepe_session=crepe_session,
            silence_front=0,
        )
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        pitch = torch.tensor(pitch, device=device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=device, dtype=torch.float).unsqueeze(0)
    else:
        pitch, pitchf = None, None
    p_len_tensor = torch.tensor([p_len], device=device).long()

    # check half-precision
    first_input_type = session.get_inputs()[0].type
    if first_input_type == "tensor(float)":
        is_model_half = False
    else:
        is_model_half = True

    feats_len = feats.shape[1]
    if pitch is not None and pitchf is not None:
        pitch = pitch[:, -feats_len:]
        pitchf = pitchf[:, -feats_len:]
    p_len_tensor = torch.tensor([feats_len], device=device).long()

    # 推論実行
    with torch.no_grad():
        audio1 = (
            infer(
                session=session,
                is_half=is_model_half,
                feats=feats,
                pitch_length=p_len_tensor,
                pitch=pitch,
                pitchf=pitchf,
                sid=sid,
            )[0]
            * 32767.5
        ).data.to(dtype=torch.int16)

    del feats, p_len_tensor, padding_mask
    torch.cuda.empty_cache()

    if t_pad_tgt != 0:
        offset = t_pad_tgt
        end = -1 * t_pad_tgt
        audio1 = audio1[offset:end]

    del pitch, pitchf, sid
    torch.cuda.empty_cache()
    return audio1.detach().cpu().numpy()
