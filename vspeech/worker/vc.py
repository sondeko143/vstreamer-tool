import json
from asyncio import AbstractEventLoop
from asyncio import CancelledError
from asyncio import Queue
from asyncio import to_thread
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import cast

import numpy as np
import pyworld
import torch
from fairseq import checkpoint_utils
from fairseq.models.hubert import HubertModel
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from scipy import signal
from torch.nn import functional

from vspeech.config import EventType
from vspeech.config import F0ExtractorType
from vspeech.config import RvcConfig
from vspeech.config import SampleFormat
from vspeech.logger import logger
from vspeech.shared_context import SharedContext
from vspeech.shared_context import SoundOutput
from vspeech.shared_context import WorkerInput
from vspeech.shared_context import WorkerOutput


def create_session(model_file: Path):
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
    return InferenceSession(str(model_file.expanduser()), providers=providers)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
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
        print(e)
        return False

    return True


def pitch_extract_harvest(
    audio: NDArray[np.float32],
    f0_max: int,
    sr: int,
) -> NDArray[np.double]:
    f0_, t = pyworld.harvest(  # type: ignore
        audio.astype(np.double),
        fs=sr,
        f0_ceil=f0_max,
        frame_period=10,
    )
    f0 = cast(
        NDArray[np.double], pyworld.stonemask(audio.astype(np.double), f0_, t, sr)  # type: ignore
    )
    return signal.medfilt(f0, 3)


def pitch_extract_dio(
    audio: NDArray[np.float32],
    f0_max: int,
    f0_min: int,
    sr: int,
):
    f0_, t = pyworld.dio(  # type: ignore
        audio.astype(np.double),
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        channels_in_octave=2,
        frame_period=10,
    )
    return cast(NDArray[np.double], pyworld.stonemask(audio.astype(np.double), f0_, t, sr))  # type: ignore


def pitch_extract(
    audio: NDArray[np.float32],
    f0_up_key: int,
    sr: int,
    window: int,
    f0_extractor: F0ExtractorType,
    silence_front: int = 0,
) -> Tuple[NDArray[Any], NDArray[np.floating[Any]]]:
    n_frames = int(len(audio) // window) + 1
    start_frame = int(silence_front * sr / window)
    real_silence_front = start_frame * window / sr

    silence_front_offset = int(np.round(real_silence_front * sr))
    audio = audio[silence_front_offset:]

    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    if f0_extractor == F0ExtractorType.dio:
        f0 = pitch_extract_dio(audio=audio, f0_max=f0_max, f0_min=f0_min, sr=sr)
    elif f0_extractor == F0ExtractorType.harvest:
        f0 = pitch_extract_harvest(audio=audio, f0_max=f0_max, sr=sr)
    else:
        raise ValueError("unknown f0 extractor type")

    f0 = np.pad(f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame))

    f0 *= pow(2, f0_up_key / 12)
    f0bak = f0.copy()
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)  # type: ignore

    return f0_coarse, f0bak


def extract_features(
    model: HubertModel,
    feats: torch.Tensor,
    dev: torch.device,
    emb_channels: int = 256,
) -> torch.Tensor:
    padding_mask = torch.BoolTensor(feats.shape).to(dev).fill_(False)
    if emb_channels == 256:
        inputs = {
            "source": feats.to(dev),
            "padding_mask": padding_mask,
            "output_layer": 9,  # layer 9
        }
    else:
        inputs = {
            "source": feats.to(dev),
            "padding_mask": padding_mask,
        }

    with torch.no_grad():
        logits = model.extract_features(**inputs)
        if emb_channels == 256:
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
    emb_channels: int,
    hubert_model: HubertModel,
    session: InferenceSession,
    f0_enabled: bool,
) -> NDArray[np.int16]:
    sound_data = np.frombuffer(voice_frames, dtype="int16")
    audio = sound_data.astype(np.float32) / 32768.0
    # audio = resampy.resample(audio, sampling_rate, 16000)

    repeat = 3 if half_available else 1
    repeat *= rvc_config.quality.value
    t_pad = voice_sample_rate * repeat
    t_pad_tgt = target_sample_rate * repeat

    audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
    sid = 0
    sid = torch.tensor(sid, device=device).unsqueeze(0).long()

    f0_up_key = rvc_config.f0_up_key

    feats: torch.Tensor = torch.from_numpy(audio_pad)
    if half_available:
        feats = feats.half()
    else:
        feats = feats.float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)

    padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)
    try:
        feats = extract_features(
            model=hubert_model, feats=feats, emb_channels=emb_channels, dev=device
        )
    except RuntimeError as e:
        raise e

    feats = cast(
        torch.Tensor,
        functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1),
    )

    # ピッチサイズ調整
    p_len = audio_pad.shape[0] // rvc_config.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
    if f0_enabled:
        try:
            pitch, pitchf = pitch_extract(
                audio_pad,
                f0_up_key,
                voice_sample_rate,
                rvc_config.window,
                f0_extractor=rvc_config.f0_extractor_type,
                silence_front=0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=device, dtype=torch.float).unsqueeze(0)
        except IndexError as e:
            print(e)
            raise
    else:
        pitch, pitchf = None, None
    p_len_tensor = torch.tensor([p_len], device=device).long()

    # check half-precision
    first_input_type = session.get_inputs()[0].type
    if first_input_type == "tensor(float)":
        is_model_half = False
    else:
        is_model_half = True

    # 推論実行
    try:
        with torch.no_grad():
            audio1 = (
                (
                    infer(
                        session=session,
                        is_half=is_model_half,
                        feats=feats,
                        pitch_length=p_len_tensor,
                        pitch=pitch,
                        pitchf=pitchf,
                        sid=sid,
                    )[0][0, 0]
                    * 32768
                )
                .data.cpu()
                .float()
                .numpy()
                .astype(np.int16)
            )
    except RuntimeError as e:
        if "HALF" in e.__str__().upper():
            raise e
        else:
            raise e

    del feats, p_len_tensor, padding_mask
    torch.cuda.empty_cache()

    if t_pad_tgt != 0:
        offset = t_pad_tgt
        end = -1 * t_pad_tgt
        audio1 = audio1[offset:end]

    del pitch, pitchf, sid
    torch.cuda.empty_cache()
    return audio1


async def rvc_worker(
    context: SharedContext,
    in_queue: Queue[WorkerInput],
):
    rvc_config = context.config.rvc
    device = get_device()
    half_available = half_precision_available(rvc_config.gpu_id)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file,
        device=device,
        is_half=half_available,
    )
    session = create_session(rvc_config.model_file)
    modelmeta: Any = session.get_modelmeta()
    metadata = json.loads(modelmeta.custom_metadata_map["metadata"])
    target_sample_rate = metadata["samplingRate"]
    f0_enabled = metadata["f0"]
    logger.info("vc worker started")
    while True:
        speech = await in_queue.get()
        rvc_config = context.config.rvc
        try:
            logger.info("voice changing...")
            audio = await to_thread(
                change_voice,
                voice_frames=speech.sound.data,
                voice_sample_rate=speech.sound.rate,
                rvc_config=rvc_config,
                half_available=half_available,
                target_sample_rate=target_sample_rate,
                device=device,
                emb_channels=metadata["embChannels"],
                hubert_model=hubert_model,
                session=session,
                f0_enabled=f0_enabled,
            )
            logger.info("voice changed")
            yield WorkerOutput(
                followings=speech.following_events,
                sound=SoundOutput(
                    data=audio.tobytes(),
                    rate=target_sample_rate,
                    channels=1,
                    format=SampleFormat.INT16,
                ),
            )
        except Exception as e:
            logger.exception(e)


async def vc_worker(
    context: SharedContext, in_queue: Queue[WorkerInput], out_queue: Queue[WorkerOutput]
):
    while True:
        context.reset_need_reload()
        try:
            async for output in rvc_worker(context=context, in_queue=in_queue):
                out_queue.put_nowait(output)
                if context.need_reload:
                    break
        except CancelledError:
            logger.debug("vc worker cancelled")
            raise


def create_vc_task(
    loop: AbstractEventLoop,
    context: SharedContext,
):
    in_queue = Queue[WorkerInput]()
    event = EventType.vc
    context.input_queues[event] = in_queue
    task = loop.create_task(
        vc_worker(context, in_queue=in_queue, out_queue=context.sender_queue),
        name=event.name,
    )
    context.worker_need_reload[task.get_name()] = False
    return task
