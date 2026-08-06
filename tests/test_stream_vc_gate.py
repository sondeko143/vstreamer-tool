"""The VAD noise gate of streaming VC (vspeech/stream_vc/gate.py).

The per-window mask and the emit-delay correction are model-independent pure logic, so
they run on CPU with no onnxruntime. The vc_loop wiring tests at the end also run on CPU,
with the real models substituted.
"""

import numpy as np
import pytest

from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import VAD_WINDOW_SAMPLES
from vspeech.stream_vc.gate import StreamingVadGate
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport

_RATE = 48000
# One window (32ms) in output samples. The grid spacing of the mask interpolation.
_STEP = round(VAD_WINDOW_SAMPLES * _RATE / VAD_SAMPLE_RATE)


def _gate(**kw) -> StreamingVadGate:
    params: dict = {"threshold": 0.5, "hangover_ms": 300.0, "min_gain": 0.0}
    params.update(kw)
    return StreamingVadGate(**params)


# --- the per-window mask ----------------------------------------------------


def test_window_gains_duck_non_speech_windows_inside_a_speech_block():
    """Even in a block containing speech, the non-speech windows inside it are ducked
    individually.

    The probability series measured on a real recording's onset block (t=3.52s). Deciding
    at block granularity (the max window probability) let a single window (0.868) pass the
    whole block at 1.0, and the tiny pre-phonation input -- amplified depending on the
    contents of the analysis window (+43dB over the batch path; the mechanism is broken
    down in ADR-0059) -- came out as a breath.
    """
    g = _gate(threshold=0.3, hangover_ms=0.0, min_gain=0.0)
    probs = np.array([0.030, 0.222, 0.140, 0.082, 0.868])
    assert list(g.window_gains(probs)) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_gate_starts_closed_so_silence_never_leaks():
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    assert list(g.window_gains(np.zeros(3))) == [0.0, 0.0, 0.0]


def test_hangover_holds_the_gate_open_after_speech():
    g = _gate(hangover_ms=2 * 32.0, min_gain=0.0)  # two windows' worth
    gains = g.window_gains(np.array([0.9, 0.1, 0.1, 0.1]))
    assert list(gains) == [1.0, 1.0, 1.0, 0.0]


def test_hangover_does_not_open_windows_before_speech():
    """Never dilate forward.

    The batch-side speech_gate_mask dilates symmetrically, but dilating forward in
    streaming opens the breath right before an onset (measured: 32ms forward makes it
    worse, +15dB -> +32dB).
    """
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    assert list(g.window_gains(np.array([0.1, 0.1, 0.9]))) == [0.0, 0.0, 1.0]


def test_hangover_budget_carries_across_blocks():
    """The hangover carries across block boundaries (decisions arrive per block)."""
    g = _gate(hangover_ms=2 * 32.0, min_gain=0.0)
    g.window_gains(np.array([0.9]))
    assert list(g.window_gains(np.array([0.1, 0.1, 0.1]))) == [1.0, 1.0, 0.0]


def test_min_gain_is_the_closed_gain():
    g = _gate(hangover_ms=0.0, min_gain=0.25)
    assert list(g.window_gains(np.zeros(2))) == [0.25, 0.25]


def test_probability_equal_to_the_threshold_counts_as_speech():
    """The boundary is inclusive (>=). Pins whether exactly the threshold opens or
    closes."""
    g = _gate(threshold=0.5, hangover_ms=0.0, min_gain=0.0)
    assert list(g.window_gains(np.array([0.5]))) == [1.0]
    g.reset()
    assert list(g.window_gains(np.array([0.499999]))) == [0.0]


def test_reset_gives_a_fresh_vad_carry():
    """The VAD's recurrent state is reset too (never carried across a real-time jump).

    Keeping it lets "pause mid-speech -> resume into silence" misjudge the first window as
    speech from the stale state. A single misjudged window resets _since_speech to 0 and
    rearms the full hangover, so the leak does not stop at one window (measured: 8 of 104
    cases, up to 320ms).
    """
    g = _gate()
    g.vad_carry.state += 1.0
    old = g.vad_carry
    g.reset()
    assert g.vad_carry is not old
    assert not g.vad_carry.state.any()


# --- application to the emit ------------------------------------------------


def test_apply_is_bit_identical_when_every_window_is_open():
    """With continuous speech (and with the feature off by default), the output is
    bit-identical to the ungated one.

    The single block right after startup or a reset is the exception: the head of the emit
    is audio from before the real-time jump, or rendered from a zeros context, so it opens
    from the closed state.
    """
    g = _gate()
    block = np.full(5 * _STEP, 1000, dtype=np.int16)
    g.apply(block, np.ones(5), 0, _RATE)  # the first block (closed -> open)
    out = np.array([100, -200, 300], dtype=np.int16)
    assert g.apply(out, np.ones(5), 0, _RATE) is out


def test_apply_keeps_the_head_closed_on_the_first_block_after_reset():
    """Right after a reset, the head of the emit (the delay portion) does not open.

    What sits there is audio from before the pause, or rendered from a zeros context --
    exactly the span ADR-0059 wants kept closed. It matches `_since_speech`'s initial
    (closed) value.
    """
    amp = 10000
    block = np.full(5 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    out = g.apply(block, np.ones(5), _STEP, _RATE).astype(np.float64)
    assert out[0] == pytest.approx(0.0, abs=1.0)  # the head really is closed
    assert out[-1] == pytest.approx(amp, abs=1.0)  # the body passes through


def test_apply_shifts_the_mask_by_the_emit_delay():
    """The mask is overlaid shifted by the emit delay.

    The emit's content starts earlier than the input block (about 52ms measured, from the
    crossfade plus SOLA plus HuBERT's receptive field). Without the correction the gate
    decision lands on audio 52ms out of alignment and breath suppression drops from -26dB
    to -8dB.
    """
    amp = 10000
    block = np.full(5 * _STEP, amp, dtype=np.int16)
    gains = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    g = _gate(min_gain=0.0)
    without = g.apply(block, gains, 0, _RATE)
    g.reset()
    shifted = g.apply(block, gains, _STEP, _RATE)

    opened_without = int(np.argmax(without > amp // 2))
    opened_shifted = int(np.argmax(shifted > amp // 2))
    assert opened_shifted - opened_without == pytest.approx(_STEP, abs=2)


def test_apply_anchors_the_mask_at_window_centers():
    """Pin the mask's absolute position (window centre = (i+0.5)*window length).

    A test that only looks at the delay would cancel out a uniform offset error, so this
    directly checks that the transition falls exactly midway between the centre of window
    0 and the centre of window 1.
    """
    amp = 10000
    block = np.full(3 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    g._prev_gains = np.zeros(3)  # start with the head closed too
    out = g.apply(block, np.array([0.0, 1.0, 1.0]), 0, _RATE).astype(np.float64)
    crossed = int(np.argmax(out > amp * 0.5))
    assert crossed == pytest.approx(
        _STEP, abs=2
    )  # midway between 0.5*step and 1.5*step


def test_apply_places_the_previous_block_one_hop_back_not_one_window_grid_back():
    """The previous block's origin is one emit length (hop) earlier -- not window count x
    window length.

    speech_probs zero-pads to ceil(block_len/512) windows, so when block_len is not a
    multiple of 512 (such as the block_ms=80 the config mentions) the windows total more
    than the block length. Shifting on the window grid moves the whole mask earlier by
    that difference (16ms at the 80ms setting).
    """
    # An 80ms block: 1280 input samples = 2.5 windows -> speech_probs returns 3 windows.
    hop = round(2.5 * _STEP)
    amp = 10000
    block = np.full(hop, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    g.apply(block, np.zeros(3), 0, _RATE)  # previous block = fully closed
    out = g.apply(block, np.ones(3), 0, _RATE).astype(np.float64)
    # The centre of the previous block's last window is -hop + 2.5*step = 0, so the head
    # of the emit already sits exactly on "the centre of a closed window". On the window
    # grid (-3*step) it would shift 0.5*step earlier and the head would rise above 0.
    assert out[0] == pytest.approx(0.0, abs=1.0)


def test_apply_ramps_across_a_window_without_a_step():
    """The gain transition ramps linearly across the window interval (32ms) = it creates
    no step (no click)."""
    amp = 10000
    block = np.full(4 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    out = g.apply(block, np.array([0.0, 0.0, 1.0, 1.0]), 0, _RATE).astype(np.float64)
    assert np.max(np.abs(np.diff(out))) <= 2.0 * amp / _STEP


def test_apply_carries_the_previous_block_mask_across_the_boundary():
    """Continue from the previous block's trailing gain (no step at the block
    boundary)."""
    amp = 10000
    block = np.full(4 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    g.apply(block, np.zeros(4), 0, _RATE)  # fully closed
    reopened = g.apply(block, np.ones(4), 0, _RATE).astype(np.float64)
    # The previous block was closed, so the head rises from a mid value (it never jumps
    # straight to full).
    assert reopened[0] < amp * 0.6
    assert np.max(np.abs(np.diff(reopened))) <= 2.0 * amp / _STEP


def test_apply_preserves_dtype_and_clips_to_int16_range():
    g = _gate()
    block = np.full(64, 32767, dtype=np.int16)
    out = g.apply(block, np.array([1.2, 1.2]), 0, _RATE)
    assert out.dtype == np.int16
    assert out.max() <= 32767
    assert out.min() >= -32768


def test_apply_handles_empty_block():
    g = _gate()
    out = g.apply(np.zeros(0, dtype=np.int16), np.ones(2), 0, _RATE)
    assert out.dtype == np.int16
    assert out.shape[0] == 0


def test_reset_closes_the_gate_and_drops_the_previous_mask_but_keeps_warned():
    """reset() returns to the closed state but leaves warned (the fail-open fault flag)
    alone."""
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    g.window_gains(np.array([0.9]))
    g.apply(np.full(_STEP, 100, dtype=np.int16), np.ones(1), 0, _RATE)
    g.warned = True
    g.reset()
    assert g._prev_gains is None
    assert list(g.window_gains(np.zeros(2))) == [
        0.0,
        0.0,
    ]  # the hangover is not carried
    assert g.warned is True


# --- vc_loop wiring ---------------------------------------------------------
#
# To run vc_loop without loading the real models or a GPU, only
# build_stream_vc_runtime and make_streaming_vc are substituted. The branch that
# decides whether there is a gate at all (no gate when `vad_session is None`) runs
# through the real code.

# Make it "one hop's worth of emit" as on real hardware (40kHz x 160ms = 6400). The mask
# grid is based on the emit length, so a contrived emit shorter than one window would
# clamp everywhere to the previous block's value and verify nothing about the wiring.
_VC_OUT = np.tile(
    np.array([1000, -2000, 3000, -4000, 5000, -6000], dtype=np.int16), 1067
)[:6400]


class _FakeStreamingVc:
    """A stand-in for the real model whose process_block returns a fixed int16 block."""

    def __init__(self, emit_delay_samples: int = 0) -> None:
        self.warmed = 0
        self.resets = 0
        self.emit_delay_samples = emit_delay_samples

    def warmup(self, _n: int = 3) -> None:
        self.warmed += 1

    def process_block(self, _block):
        return _VC_OUT.copy()

    def _reset_context(self) -> None:
        self.resets += 1


class _FakeSession:
    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _CollectTransport(Transport):
    def __init__(self) -> None:
        self.packets: list[StreamPacket] = []

    async def send(self, packet: StreamPacket) -> bool:
        self.packets.append(packet)
        return True

    async def recv(self) -> StreamPacket:  # pragma: no cover - vc_loop never uses it
        raise NotImplementedError


def _context():
    """A minimal SharedContext with running set (i.e. not paused)."""
    from vspeech.config import Config
    from vspeech.shared_context import SharedContext

    return SharedContext(config=Config())


async def _run_vc_loop(
    monkeypatch, sv_config, vad_session, n_blocks: int, emit_delay_samples: int = 0
):
    """Run vc_loop for exactly n_blocks and return (transport, the recorded gate.apply
    arguments)."""
    import asyncio
    from asyncio import Event
    from asyncio import Queue

    from vspeech.stream_vc import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build_stream_vc_runtime",
        lambda cfg: {
            "rvc_config": cfg.rvc,
            "device": None,
            "hubert_model": None,
            "session": _FakeSession(),
            "f0_session": None,
            "vad_session": vad_session,
            "target_sample_rate": 40000,
            "f0_enabled": True,
            "emb_output_layer": 9,
            "use_final_proj": True,
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "make_streaming_vc",
        lambda _rt, _cfg: _FakeStreamingVc(emit_delay_samples),
    )

    applied: list[tuple[list[float], int, int]] = []
    real_apply = StreamingVadGate.apply

    def spy_apply(self, out_i16, gains, delay_samples, sample_rate):
        # Record **the delay and the output rate** as well as gains. Without watching
        # them, the tests stay green even if the runner starts passing 0 instead of
        # sv.emit_delay_samples, or starts passing 16kHz.
        applied.append(([float(x) for x in gains], delay_samples, sample_rate))
        return real_apply(self, out_i16, gains, delay_samples, sample_rate)

    monkeypatch.setattr(StreamingVadGate, "apply", spy_apply)

    in_queue: Queue = Queue()
    for _ in range(n_blocks):
        in_queue.put_nowait(np.zeros(2560, dtype=np.float32))
    transport = _CollectTransport()
    task = asyncio.create_task(
        runner_mod.vc_loop(_context(), sv_config, in_queue, transport, "sess", Event())
    )
    for _ in range(2000):
        await asyncio.sleep(0)
        if len(transport.packets) >= n_blocks or task.done():
            break
    if task.done():
        task.result()  # let a startup exception surface as-is
    task.cancel()
    try:
        await task
    except BaseException:
        pass
    assert len(transport.packets) == n_blocks
    return transport, applied


async def test_default_off_never_applies_the_gate(monkeypatch):
    """With vad_gate=False no gate is built and apply is never reached = bit-identical."""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig()
    assert sv.vad_gate is False  # off by default
    transport, applied = await _run_vc_loop(monkeypatch, sv, None, 2)
    assert applied == []  # apply is never called
    assert transport.packets[0].pcm == _VC_OUT.tobytes()  # exactly the ungated output
    assert transport.packets[1].pcm == _VC_OUT.tobytes()


async def test_gate_enabled_attenuates_silent_blocks(monkeypatch):
    """With vad_gate=True and a silence verdict, the gate closes and the output is
    attenuated."""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(
        vad_gate=True, vad_hangover_ms=0.0, vad_min_gain=0.0, block_ms=160.0
    )
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs",
        lambda _session, _audio, _carry=None: np.zeros(5),
    )
    transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert [a[0] for a in applied] == [[0.0] * 5, [0.0] * 5]  # min_gain in every window
    for packet in transport.packets:
        assert not np.frombuffer(packet.pcm, dtype=np.int16).any()


async def test_vc_loop_forwards_the_emit_delay_and_the_output_sample_rate(monkeypatch):
    """The runner passes StreamingVc's emit delay and the **output** rate to the gate.

    These two arguments are the substance of the correction ADR-0059 estimated at 18dB.
    Watching gains alone would detect neither dropping the delay to 0 nor passing the
    input rate (16k).
    """
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs",
        lambda _session, _audio, _carry=None: np.zeros(5),
    )
    _, applied = await _run_vc_loop(
        monkeypatch, sv, object(), 2, emit_delay_samples=1234
    )
    # target_sample_rate=40000 comes from _run_vc_loop's runtime stub.
    assert [(a[1], a[2]) for a in applied] == [(1234, 40000), (1234, 40000)]


async def test_vc_loop_forwards_the_emit_delay_to_the_envelope_too(monkeypatch):
    """The envelope overlays the same emit in time alignment as the gate, so the runner
    has to hand it the same emit delay (ADR-0065).

    The harness lives in this module, so the envelope's vc_loop wiring is checked here
    rather than in test_stream_vc_envelope.py (which covers the pure logic).
    """
    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc.envelope import StreamingEnvelope

    sv = StreamVcConfig(envelope_follow=True)
    seen: list[int] = []
    real_apply = StreamingEnvelope.apply

    def spy_apply(self, out_i16, in_block, delay_samples):
        seen.append(delay_samples)
        return real_apply(self, out_i16, in_block, delay_samples)

    monkeypatch.setattr(StreamingEnvelope, "apply", spy_apply)
    await _run_vc_loop(monkeypatch, sv, None, 2, emit_delay_samples=1234)
    assert seen == [1234, 1234]


async def test_vc_loop_threads_the_vad_carry_into_speech_probs(monkeypatch):
    """The runner passes gate.vad_carry into speech_probs (the same object every block).

    Dropping the third argument returns Silero to a cold start on every block and the
    speech windows detectable on a real recording fall from 56 to 23. Without checking
    that it is passed, that regression lands with the tests still green.
    """
    from vspeech.config import StreamVcConfig
    from vspeech.lib.vad import VadCarry

    sv = StreamVcConfig(vad_gate=True)
    seen: list[object] = []

    def spy_probs(_session, _audio, carry=None):
        seen.append(carry)
        return np.zeros(5)

    monkeypatch.setattr("vspeech.lib.vad.speech_probs", spy_probs)
    await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert len(seen) == 2
    assert isinstance(seen[0], VadCarry)  # not None = the argument is passed
    assert seen[0] is seen[1]  # the same carry every block = the state is carried over


async def test_gate_open_on_speech_is_bit_identical(monkeypatch):
    """While the speech verdict holds, the identity path matches the ungated output.

    The first block is the exception (it opens from the closed state, so the head of the
    emit ramps up).
    """
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs",
        lambda _session, _audio, _carry=None: np.full(5, 0.99),
    )
    transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 3)
    assert [a[0] for a in applied] == [[1.0] * 5] * 3
    for p in transport.packets[1:]:
        assert p.pcm == _VC_OUT.tobytes()


async def test_gate_failure_is_fail_open_and_warns_once(monkeypatch, caplog):
    """Even when the VAD fails the audio passes through, and the warning is emitted once
    rather than per block."""
    import logging

    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)

    def boom(_session, _audio, _carry=None):
        # Match the real arity (session, audio, carry). Otherwise a TypeError fires first
        # and the test observes an arity mistake rather than the fail-open behaviour.
        raise RuntimeError("vad exploded")

    monkeypatch.setattr("vspeech.lib.vad.speech_probs", boom)
    with caplog.at_level(logging.WARNING):
        transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 3)
    # fail-open returns a fully open mask with **the real window count** (2560 samples =
    # 5 windows). With a single element the next block would place it one hop earlier and
    # a step would appear at the seam.
    assert [a[0] for a in applied] == [[1.0] * 5] * 3
    for p in transport.packets[1:]:
        assert p.pcm == _VC_OUT.tobytes()  # passed through (the identity fast path)
    warnings = [r for r in caplog.records if "vad gate failed" in r.getMessage()]
    assert len(warnings) == 1
    # Confirm we are observing the intended exception (with only the assert above, a
    # TypeError from a stub arity mismatch firing first would still pass green).
    assert "vad exploded" in warnings[0].getMessage()


# --- the pause/resume gate --------------------------------------------------
#
# vc_loop lives outside Command routing but still respects context.running. With the
# real models substituted, this verifies on CPU that consumption/conversion stops
# while paused and that _reset_context is called on resume.


def _patch_runtime(monkeypatch, fake, vad_session=None):
    from vspeech.stream_vc import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build_stream_vc_runtime",
        lambda cfg: {
            "rvc_config": cfg.rvc,
            "device": None,
            "hubert_model": None,
            "session": _FakeSession(),
            "f0_session": None,
            "vad_session": vad_session,
            "target_sample_rate": 40000,
            "f0_enabled": True,
            "emb_output_layer": 9,
            "use_final_proj": True,
        },
    )
    monkeypatch.setattr(runner_mod, "make_streaming_vc", lambda _rt, _cfg: fake)
    return runner_mod


async def test_pause_stops_consuming_and_resets_on_resume(monkeypatch):
    """While paused vc_loop converts no blocks, and on resume it calls _reset_context
    once."""
    import asyncio
    from asyncio import Event
    from asyncio import Queue

    from vspeech.config import Config
    from vspeech.config import StreamVcConfig
    from vspeech.shared_context import SharedContext

    fake = _FakeStreamingVc()
    # Run with the gate enabled so we also see the resume transition calling gate.reset
    # (a different call site from the REOPEN sentinel path).
    runner_mod = _patch_runtime(monkeypatch, fake, vad_session=object())
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs",
        lambda _session, _audio, _carry=None: np.full(5, 0.99),
    )
    gate_resets: list[int] = []
    real_reset = StreamingVadGate.reset

    def spy_reset(self):
        gate_resets.append(1)
        return real_reset(self)

    monkeypatch.setattr(StreamingVadGate, "reset", spy_reset)

    context = SharedContext(config=Config())  # running.set() by default
    sv = StreamVcConfig(vad_gate=True)
    in_queue: Queue = Queue()
    transport = _CollectTransport()
    ready = Event()
    task = asyncio.create_task(
        runner_mod.vc_loop(context, sv, in_queue, transport, "sess", ready)
    )
    # Wait for startup (warmup, to_thread) to finish.
    await asyncio.wait_for(ready.wait(), timeout=5)

    # Pause and push in 3 blocks. The loop parks in running.wait() right after getting
    # block0 (it does not process_block or send).
    context.running.clear()
    for _ in range(3):
        in_queue.put_nowait(np.zeros(2560, dtype=np.float32))
    for _ in range(50):
        await asyncio.sleep(0)
    assert transport.packets == []  # paused: nothing was converted at all
    # consumed only block0 and parked (the stale one is discarded on resume)
    assert in_queue.qsize() == 2
    assert fake.resets == 0

    # resume: the parked wait() returns, _reset_context runs, continue discards block0,
    # and block1/block2 are converted.
    context.running.set()
    for _ in range(2000):
        await asyncio.sleep(0)
        if len(transport.packets) >= 2:
            break
    assert fake.resets == 1  # exactly once on the resume transition
    # the VAD gate (hangover/mask/recurrent state) is reset exactly once too
    assert gate_resets == [1]
    assert len(transport.packets) == 2  # stale block0 dropped; only block1/block2

    task.cancel()
    try:
        await task
    except BaseException:
        pass


# --- the capture-reopen sentinel --------------------------------------------
#
# When capture reopens the device, a CaptureSignal.REOPEN sentinel enters capture_queue.
# On seeing it the runner resets the context and the VAD gate and never converts the
# sentinel itself (treated like a pause). Verified on CPU with the real models
# substituted.


async def test_capture_reopen_sentinel_resets_context_and_gate(monkeypatch):
    """The reopen sentinel is not converted; it calls _reset_context + gate.reset."""
    import asyncio
    from asyncio import Event
    from asyncio import Queue

    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc import runner as runner_mod
    from vspeech.stream_vc.capture import CaptureSignal

    fake = _FakeStreamingVc()
    # Enable the gate (vad_session != None) so gate.reset is verified too.
    sv = StreamVcConfig(vad_gate=True)
    monkeypatch.setattr(
        runner_mod,
        "build_stream_vc_runtime",
        lambda cfg: {
            "rvc_config": cfg.rvc,
            "device": None,
            "hubert_model": None,
            "session": _FakeSession(),
            "f0_session": None,
            "vad_session": object(),
            "target_sample_rate": 40000,
            "f0_enabled": True,
            "emb_output_layer": 9,
            "use_final_proj": True,
        },
    )
    monkeypatch.setattr(runner_mod, "make_streaming_vc", lambda _rt, _cfg: fake)
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs",
        lambda _session, _audio, _carry=None: np.full(5, 0.99),
    )
    reset_calls: list[int] = []
    real_reset = StreamingVadGate.reset

    def spy_reset(self):
        reset_calls.append(1)
        return real_reset(self)

    monkeypatch.setattr(StreamingVadGate, "reset", spy_reset)

    in_queue: Queue = Queue()
    in_queue.put_nowait(CaptureSignal.REOPEN)  # the reopen sentinel (first)
    in_queue.put_nowait(np.zeros(2560, dtype=np.float32))  # the fresh block after it
    transport = _CollectTransport()
    task = asyncio.create_task(
        runner_mod.vc_loop(_context(), sv, in_queue, transport, "sess", Event())
    )
    for _ in range(2000):
        await asyncio.sleep(0)
        if len(transport.packets) >= 1 or task.done():
            break
    if task.done():
        task.result()  # let a startup exception surface as-is
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    assert fake.resets == 1  # the sentinel triggers _reset_context once
    assert reset_calls == [1]  # gate.reset once as well
    assert (
        len(transport.packets) == 1
    )  # the sentinel produces no packet (only the block)
    # Right after the reset, only the head of the emit ramps up from closed (what sits
    # there is audio from before the reopen, or rendered from a zeros context). The body
    # passes through.
    got = np.frombuffer(transport.packets[0].pcm, dtype=np.int16)
    assert abs(int(got[0])) < abs(int(_VC_OUT[0]))
    assert got[-1] == _VC_OUT[-1]
