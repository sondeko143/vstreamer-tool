from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union

import numpy as np
import torch
from whisper.model import Whisper

class Segment(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

class TranscribeResult(TypedDict):
    text: str
    segments: List[Segment]
    language: str


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = ...,
    download_root: str = ...,
    in_memory: bool = ...,
) -> Whisper: ...
def transcribe(
    model: Whisper,
    audio: Union[str, np.ndarray[Any, Any], torch.Tensor],
    *,
    verbose: Optional[bool] = ...,
    temperature: Union[float, Tuple[float, ...]] = ...,
    compression_ratio_threshold: Optional[float] = ...,
    logprob_threshold: Optional[float] = ...,
    no_speech_threshold: Optional[float] = ...,
    condition_on_previous_text: bool = ...,
    **decode_options: Any,
) -> TranscribeResult: ...
