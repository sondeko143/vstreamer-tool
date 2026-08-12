"""PCM builders shared by the audio-boundary tests.

Not fixtures: these are called from module-level helpers (`_peak_frequency` and friends)
as well as from tests, which a fixture cannot serve. `tests/conftest.py` holds the things
that genuinely need fixture semantics.
"""

import numpy as np
from numpy.typing import NDArray


def i16(pcm: bytes) -> NDArray[np.int16]:
    """Read raw bytes as the int16 mono samples every device boundary speaks."""
    return np.frombuffer(pcm, dtype=np.int16)


def sine(rate: int, samples: int, freq: float = 440.0) -> bytes:
    """int16 mono PCM of a sine -- the shape an utterance or a StreamPacket carries."""
    t = np.arange(samples, dtype=np.float64) / rate
    return np.rint(np.sin(2 * np.pi * freq * t) * 20000.0).astype(np.int16).tobytes()


def peak_frequency(pcm: bytes, rate: int) -> float:
    """The dominant frequency of `pcm` **read at `rate`**.

    Reading converted bytes at the device rate is what makes a resampling test a test of
    the conversion: audio left at the source rate peaks at the wrong multiple here.
    """
    samples = i16(pcm).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))])
