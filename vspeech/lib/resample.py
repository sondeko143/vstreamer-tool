"""Rational-ratio polyphase FIR resampler for the device boundaries (ADR-0070).

numpy only -- no torch, no scipy, no sounddevice. This module is reachable from the
streaming consumer role, which must not pull torch in (ADR-0055), and numpy is
guaranteed present because the base dependency ctranslate2 requires it.

The filter is a Kaiser-windowed sinc whose length is *derived* from the requested
transition width and stopband attenuation rather than fixed at some tap count. A
fixed tap count silently produces a useless filter as the ratio changes: at 16
taps per phase the 48k->16k transition band spans 6.6-9.4 kHz, leaving 8.4 kHz
only 14 dB down (measured).
"""

from math import ceil
from math import gcd
from math import pi

import numpy as np
from numpy.typing import NDArray


def _kaiser_beta(stopband_db: float) -> float:
    """Kaiser's empirical beta for a target stopband attenuation."""
    if stopband_db > 50.0:
        return 0.1102 * (stopband_db - 8.7)
    if stopband_db >= 21.0:
        excess = stopband_db - 21.0
        return 0.5842 * excess**0.4 + 0.07886 * excess
    return 0.0


class PolyphaseResampler:
    """Stateful L/M polyphase resampler over float32.

    `process` is the streaming entry point: it keeps the filter tail across calls, so
    feeding a signal in blocks gives the same result as feeding it whole. Use it for
    a continuous stream (mic capture, continuous playback). `resample_full` is the
    one-shot entry point for a self-contained buffer (one utterance): it flushes the
    tail and removes the group delay, then resets.
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        transition_width: float = 0.10,
        stopband_db: float = 80.0,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"rates must be positive: {src_rate} -> {dst_rate}")
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        divisor = gcd(self.src_rate, self.dst_rate)
        self.up = self.dst_rate // divisor
        self.down = self.src_rate // divisor
        # The prototype runs at the interpolated rate up*src. Its stopband must start
        # at the narrower of the two Nyquist limits: below that is signal we keep,
        # above it is what would alias (downsampling) or show up as an image
        # (upsampling).
        interpolated = self.up * self.src_rate
        nyquist = min(self.src_rate, self.dst_rate) / 2.0
        passband = nyquist * (1.0 - transition_width)
        cutoff = 0.5 * (passband + nyquist)
        width = 2 * pi * (nyquist - passband) / interpolated
        n_taps = int(ceil((stopband_db - 8.0) / (2.285 * width))) + 1
        # Round the half length up to a multiple of `down` so the group delay is a
        # whole number of OUTPUT samples. A fractional delay cannot be trimmed exactly
        # in resample_full and shows up as a phase error (measured -25 dB against the
        # reference, versus -74 dB once the delay is integral).
        self._half_len = ceil(((n_taps - 1) // 2) / self.down) * self.down
        self.delay_samples = self._half_len // self.down
        n_taps = 2 * self._half_len + 1
        index = np.arange(-self._half_len, self._half_len + 1, dtype=np.float64)
        normalised_cutoff = 2.0 * cutoff / interpolated
        taps = (
            normalised_cutoff
            * np.sinc(normalised_cutoff * index)
            * np.kaiser(n_taps, _kaiser_beta(stopband_db))
        )
        # Unity gain at DC after the up-fold zero stuffing.
        taps *= self.up / taps.sum()
        self.taps_per_phase = ceil(n_taps / self.up)
        padded = np.concatenate(
            [taps, np.zeros(self.taps_per_phase * self.up - n_taps)]
        )
        # phase p holds taps[p::up], reversed so each output is a forward dot product
        # against a forward window of the input.
        self._phases = np.ascontiguousarray(
            padded.reshape(self.taps_per_phase, self.up).T[:, ::-1].astype(np.float32)
        )
        self._tail: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._fed = 0
        self._emitted = 0
        self.reset()

    def reset(self) -> None:
        """Drop the filter state. Call this whenever the stream is discontinuous
        (device reopen, pause/resume, a new sender session)."""
        self._tail = np.zeros(self.taps_per_phase - 1, dtype=np.float32)
        self._fed = 0
        self._emitted = 0

    def out_len(self, n_in: int) -> int:
        """How many output samples `process` will return for `n_in` more inputs."""
        total = self._fed + n_in
        return -((-self.up * total) // self.down) - self._emitted

    def process(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample a block, carrying the filter state across calls.

        Accepts `(n,)` or `(n, channels)`. Output sample k is the dot product of
        phase `k*down % up` with the input window ending at `k*down // up`, so the
        output is delayed by `delay_samples` and no samples are held back.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[0] == 0:
            return np.zeros_like(x, shape=(0, *x.shape[1:]))
        if self._tail.ndim != x.ndim:
            # First block decides the channel layout.
            self._tail = np.zeros(
                (self.taps_per_phase - 1, *x.shape[1:]), dtype=np.float32
            )
        full = np.concatenate([self._tail, x])
        total = self._fed + x.shape[0]
        end = -((-self.up * total) // self.down)
        n_out = end - self._emitted
        out = np.zeros((n_out, *x.shape[1:]), dtype=np.float32)
        if n_out:
            window = np.lib.stride_tricks.sliding_window_view(
                full, self.taps_per_phase, axis=0
            )
            if self.up == 1:
                # Integer decimation: one phase, and the window start advances by a
                # constant `down`, so this is a single strided matvec.
                start = self._emitted * self.down - self._fed
                out[:] = window[start :: self.down][:n_out] @ self._phases[0]
            else:
                # Within one phase the window start also advances by exactly `down`
                # (k -> k+up maps to m -> m+down), so each phase is a strided view and
                # no gather copy is needed.
                for offset in range(min(self.up, n_out)):
                    k = self._emitted + offset
                    phase = (k * self.down) % self.up
                    start = (k * self.down) // self.up - self._fed
                    count = (n_out - offset + self.up - 1) // self.up
                    rows = window[start :: self.down][:count]
                    out[offset :: self.up][: len(rows)] = rows @ self._phases[phase]
        self._emitted = end
        self._fed = total
        keep = self.taps_per_phase - 1
        self._tail = full[-keep:] if keep else full[:0]
        return out

    def resample_full(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample one self-contained buffer: flush the tail, remove the group delay.

        Streaming `process` would leave the last `delay_samples` worth of audio inside
        the filter, so an utterance played through it would lose its tail.
        """
        self.reset()
        flush = np.zeros(
            (self._half_len // self.up + self.down, *x.shape[1:]), dtype=np.float32
        )
        out = np.concatenate([self.process(x), self.process(flush)])
        self.reset()
        want = -((-self.up * x.shape[0]) // self.down)
        return out[self.delay_samples : self.delay_samples + want]


def make_resampler(src_rate: int, dst_rate: int) -> PolyphaseResampler | None:
    """A resampler, or None when the rates already match.

    None means "pass the bytes through untouched" -- the callers rely on that to stay
    bit-identical to the pre-ADR-0070 behaviour when the device already runs at the
    pipeline's rate.
    """
    if src_rate == dst_rate:
        return None
    return PolyphaseResampler(src_rate, dst_rate)
