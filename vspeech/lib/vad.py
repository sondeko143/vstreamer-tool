"""Silero VAD gate shared by the vc and transcription workers (ADR-0037).

Pure-numpy gate math plus a thin onnxruntime wrapper. This module must stay
importable without onnxruntime/torch installed (base extras): onnxruntime is
imported lazily inside create_vad_session and only type-checked here.
"""

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

# Silero VAD v5/v6 share this contract: 16kHz mono, 512-sample (32ms) windows,
# with a 64-sample context carried between windows and a (2, 1, 128) recurrent
# state. This module pins the v6.2.1 model.
VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_WINDOW_MS = VAD_WINDOW_SAMPLES * 1000.0 / VAD_SAMPLE_RATE


def should_skip_vc(
    probs: NDArray[np.float64], threshold: float, min_speech_ratio: float
) -> tuple[bool, float]:
    """Decide whether a chunk has too little speech to be worth RVC inference.

    Returns (skip, speech_ratio) where speech_ratio is the fraction of windows
    whose speech probability reaches threshold. An empty chunk is skipped.

    Deferred limitation: as a *ratio*, recording's lookback + gradual-stop
    silence padding inflates the denominator for short utterances, dropping a
    ~0.4s aizuchi below vad_min_speech_ratio -- against the filler-preservation
    goal (ADR-0037). Workaround: a low ratio (0.03-0.05, see config.toml.example).
    A real fix (floor the *absolute* speech-window count) needs a new knob and
    also touches the shared vc gate, so it's left as a separate design call.
    """
    if probs.shape[0] == 0:
        return True, 0.0
    ratio = float(np.mean(probs >= threshold))
    return ratio < min_speech_ratio, ratio


def speech_gate_mask(
    probs: NDArray[np.float64], threshold: float, pad_ms: float, min_gain: float
) -> NDArray[np.float64]:
    """Per-window gains: 1.0 on speech windows, min_gain elsewhere.

    The binary speech mask is dilated by pad_ms on both sides before gains are
    assigned, so consonant onsets and word tails just outside the VAD's speech
    region are not clipped.
    """
    speech = (probs >= threshold).astype(np.float64)
    pad_windows = round(pad_ms / _WINDOW_MS)
    if pad_windows > 0 and speech.shape[0]:
        kernel = np.ones(2 * pad_windows + 1)
        speech = (np.convolve(speech, kernel, mode="same") > 0).astype(np.float64)
    return np.where(speech > 0, 1.0, min_gain)


def apply_vad_gate(
    output_i16: NDArray[np.int16], window_gains: NDArray[np.float64]
) -> NDArray[np.int16]:
    """Multiply the RVC output by the window-resolution gain mask.

    Window centers are mapped onto the output sample grid through a normalized
    0..1 time axis (input and output cover the same duration at different
    sample rates) and linearly interpolated, so gain transitions ramp over a
    ~32ms window instead of stepping (no clicks).
    """
    out_len = int(output_i16.shape[0])
    n_windows = int(window_gains.shape[0])
    if out_len == 0 or n_windows == 0:
        return output_i16
    src_x = (np.arange(n_windows) + 0.5) / n_windows
    dst_x = (np.arange(out_len) + 0.5) / out_len
    gain = np.interp(dst_x, src_x, window_gains)
    out_f = output_i16.astype(np.float32) * gain
    return np.clip(out_f, -32768.0, 32767.0).astype(np.int16)


def create_vad_session(model_file: Path) -> InferenceSession:
    """Build a CPU onnxruntime session for the Silero VAD model (v6.2.1).

    Fails loudly on a missing file or a model lacking the shared v5/v6
    state-input contract: silently passing audio through would mean the noise
    the gate exists to stop comes back unnoticed. CPU is deliberate -- the
    model is ~2MB and must not contend with RVC for the GPU.
    """
    from onnxruntime import GraphOptimizationLevel
    from onnxruntime import InferenceSession
    from onnxruntime import SessionOptions

    path = model_file.expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Silero VAD model not found: {path}. Download silero_vad.onnx"
            " (v6.2.1) from the snakers4/silero-vad repository and set"
            " vc.vad_model_file."
        )
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in session.get_inputs()}
    if "state" not in input_names:
        raise ValueError(
            f"{path} does not look like a Silero VAD v5/v6 model (inputs:"
            f" {sorted(input_names)}); v4 models (h/c inputs) are unsupported."
        )
    return session


class VadCarry:
    """Silero's recurrent state and 64-sample context, carried between calls.

    Silero VAD is an RNN: its per-window probability depends on the state built
    up from preceding audio. A caller that hands it one whole utterance at a
    time (the vc and transcription workers) is right to start cold, because the
    chunk boundary IS the start of the audio it cares about. A caller that feeds
    a continuous stream in small fixed blocks is not: restarting cold every
    block makes the model re-decide "is this speech?" from scratch several times
    per second. Measured on a real 12s recording at 160ms blocks, that cold
    start dropped 15 of 34 unambiguously voiced windows (input RMS > 0.05)
    below 0.3, where a state-carrying run scores them 0.969-1.000 (mean 0.996).

    Pass one of these to `speech_probs` to thread the state; carrying it makes
    the per-block calls bit-exact with a single whole-signal call **when the
    block length is a multiple of VAD_WINDOW_SAMPLES** (the streaming default,
    160ms at 16kHz = 5 windows, is). Otherwise each block's tail window is
    zero-padded and that padding is both fed to the RNN and written into the
    carried context, so the split result drifts from the whole-signal one. A
    proper fix would buffer the sub-window remainder here instead of padding it;
    left undone because the validated block sizes are multiples of 32ms.

    Drop it (or make a fresh one) whenever real time jumps -- pause/resume,
    device re-open.
    """

    __slots__ = ("state", "context")

    def __init__(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)


def speech_probs(
    session: Any, audio_16k: NDArray[np.float32], carry: VadCarry | None = None
) -> NDArray[np.float64]:
    """Per-window speech probabilities for a 16kHz float32 chunk.

    Replicates the silero-vad v5/v6 wrapper: 512-sample windows, each prefixed
    with the previous window's last 64 samples, with the recurrent state
    threaded through. The tail window is zero-padded. `session` is an
    onnxruntime InferenceSession (typed Any so tests can substitute a stub).

    `carry` is None for chunk-at-a-time callers: the state and context start at
    zero and are discarded, which is the historical behaviour. Streaming callers
    pass a `VadCarry` so the state survives across blocks -- see `VadCarry` for
    why that matters.
    """
    n = int(audio_16k.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    n_windows = ceil(n / VAD_WINDOW_SAMPLES)
    padded = np.zeros(n_windows * VAD_WINDOW_SAMPLES, dtype=np.float32)
    padded[:n] = audio_16k
    if carry is None:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
    else:
        state, context = carry.state, carry.context
    sr = np.array(VAD_SAMPLE_RATE, dtype=np.int64)
    probs = np.zeros(n_windows, dtype=np.float64)
    for i in range(n_windows):
        window = padded[i * VAD_WINDOW_SAMPLES : (i + 1) * VAD_WINDOW_SAMPLES]
        feed = {
            "input": np.concatenate([context, window]).reshape(1, -1),
            "state": state,
            "sr": sr,
        }
        out, state = session.run(None, feed)
        probs[i] = float(out[0, 0])
        context = window[-_CONTEXT_SAMPLES:]
    if carry is not None:
        carry.state, carry.context = state, context
    return probs
