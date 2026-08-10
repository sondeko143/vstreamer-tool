"""The conversion loop of streaming VC (ADR-0053).

Converts capture's float32 blocks through StreamingVc (fixed block + left context +
crossfade), wraps them in a StreamPacket and sends them to the transport. Models are
built by the same procedure as the utterance-path rvc_worker (vspeech/worker/vc.py), but
from [stream_vc.rvc] (the utterance path is untouched). Heavy imports live inside the
functions.
"""

from __future__ import annotations

from asyncio import CancelledError
from asyncio import Event
from asyncio import Queue
from asyncio import to_thread
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Any

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.log_throttle import LogThrottle
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.capture import CaptureSignal
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vspeech.lib.stream_vc import StreamingVc
    from vspeech.shared_context import SharedContext
    from vspeech.stream_vc.capture import CaptureItem
    from vspeech.stream_vc.envelope import StreamingEnvelope
    from vspeech.stream_vc.gate import StreamingVadGate

# ORT log threshold: 0=VERBOSE / 1=INFO / 2=WARNING / 3=ERROR / 4=FATAL.
# The default of SessionOptions().log_severity_level is -1 = inherit the Env level
# (onnx_session.py).
_ORT_LOG_ERROR = 3

# Allow this many consecutive transient GPU errors from process_block and drop the
# blocks. Past that, fail instead of spinning silently (see vc_loop's error handling
# below).
_MAX_CONSECUTIVE_VC_ERRORS = 10


def make_stream_packet(
    session_id: str, seq: int, hop_seconds: float, pcm: bytes, sample_rate: int
) -> StreamPacket:
    """A StreamPacket carrying seq/pts (pts = seq * hop_seconds)."""
    return StreamPacket(
        session_id=session_id,
        seq=seq,
        pts=seq * hop_seconds,
        pcm=pcm,
        sample_rate=sample_rate,
    )


def apply_input_boost(block, boost):
    """Apply the input_boost gain to the input block (clipping to [-1,1], equivalent to
    the int16 `mul` saturation in the utterance path's vc.py). The utterance path applies
    the gain outside `change_voice` (in the worker), so streaming applies it outside
    `StreamingVc` (in this runner) for symmetry. boost==1.0 is the identity (and is the
    default, so the default config behaves unchanged)."""
    import numpy as np

    if boost == 1.0:
        return block
    return np.clip(block * boost, -1.0, 1.0).astype(np.float32)


def make_stream_envelope(sv_config: StreamVcConfig) -> StreamingEnvelope | None:
    """Build a StreamingEnvelope when envelope_follow is on (None when off). Pure."""
    if not sv_config.envelope_follow:
        return None
    from vspeech.stream_vc.envelope import StreamingEnvelope

    return StreamingEnvelope(
        strength=sv_config.envelope_strength,
        min_gain=sv_config.envelope_min_gain,
        max_gain=sv_config.envelope_max_gain,
        window_ms=sv_config.envelope_window_ms,
        ema_ms=sv_config.envelope_ema_ms,
        block_ms=sv_config.block_ms,
    )


async def gate_window_gains(
    gate: StreamingVadGate,
    vad_session: Any,
    block: NDArray[np.float32],
    error_throttle: LogThrottle,
) -> NDArray[np.float64]:
    """Return the 32ms per-window gains from the VAD decision on the **input** block
    (ADR-0059).

    The decision is made on the input side (the bare mic level, before input_boost, i.e.
    at the actual S/N) and applied on the output side. Inference itself is never skipped:
    `StreamingVc` carries a rolling left context and a crossfade tail, so skipping a block
    punches a hole in the context and breaks the seam when speech resumes (the GPU has
    plenty of headroom -- measured RTF 0.24).

    Silero is an RNN, so its recurrent state is carried across blocks
    (`gate.vad_carry`). Rebuilding it per block cold-starts it every time and returns low
    probabilities even for clearly voiced windows (measured: 15 of 34 windows below 0.3).
    A per-window gate uses those probabilities directly, one window at a time, so
    carrying the state over is mandatory (see VadCarry in lib/vad.py).

    ONNX inference blocks, so it is offloaded to `to_thread`, as in the utterance path's
    worker/vc.py. On failure the audio passes through (fail-open).

    Failing open means the gate silently stops doing its job, and what that sounds like is
    precisely the amplified room noise ADR-0059 exists to remove -- so the failure must
    stay observable however long it lasts. Telemetry is therefore recorded on **every**
    occurrence and only the log line is thinned, by time and per episode (ADR-0062), the
    same discipline as every other fault path in this subsystem. A boolean warn-once would
    go permanently quiet after the first line and record nothing at all.
    """
    import numpy as np

    from vspeech.lib.vad import speech_probs

    try:
        probs = await to_thread(speech_probs, vad_session, block, gate.vad_carry)
        return gate.window_gains(probs)
    except Exception as e:
        telemetry.record("stream_vc_vad_error", 1.0)
        if (n := error_throttle.hit()) is not None:
            logger.warning(
                "stream_vc vad gate failed; passing audio ungated (total %d): %s", n, e
            )
        # window_gains is not run, so the hangover budget (`_since_speech`) is left as is.
        # While failing open everything is wide open anyway, so it does no harm, and on
        # recovery it continues from the previous budget. The length must match **the real
        # window count**: returning a single element makes the next successful block place
        # it as "the previous block's mask" one hop earlier, shifting the window centre to
        # -144ms and producing a 0.59 gain step (a click) at the seam.
        from math import ceil

        from vspeech.lib.vad import VAD_WINDOW_SAMPLES

        n_windows = max(1, ceil(int(block.shape[0]) / VAD_WINDOW_SAMPLES))
        return np.ones(n_windows, dtype=np.float64)


def build_stream_vc_runtime(sv_config: StreamVcConfig) -> dict[str, Any]:
    """Build the device, models and metadata from [stream_vc.rvc]."""
    import json

    from vspeech.config import F0ExtractorType
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.onnx_session import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    rvc = sv_config.rvc
    device, device_name = get_device(rvc.gpu_id, rvc.gpu_name)
    logger.info("stream_vc device: %s, %s", device, device_name)
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc.hubert_model_file, device=device, is_half=half_available
    )
    session = create_session(rvc.model_file, device)
    # Silence ORT's warnings for the f0 session only. Because fcpe.onnx
    # (poe export-fcpe-onnx, ADR-0049) traces torchfcpe with dynamic_axes, the inferred
    # rank of the intermediate node /bundled/Squeeze_1 disagrees with the real one and ORT
    # prints a VerifyOutputSizes warning to stdout on **every inference**. It is benign
    # (the real shape is allocated and f0 is correct) but in streaming it becomes about 6
    # lines a second and buries the log. Fixing the graph would need graph surgery or an
    # upstream patch since it comes from torchfcpe's tracing, which is not worth it (a
    # past attempt at ONNX graph surgery went nowhere).
    # The cost: ORT warnings specific to this f0 session (provider fallback, etc.) become
    # invisible too. So the one silenced warning that does real harm -- "CUDA was
    # requested but it fell back to CPU" -- is caught programmatically by
    # check_cuda_provider(f0_session) in vc_loop (WorkerStartupError = fail-loud). That is
    # the counterpart that keeps the log thinning from costing us the diagnosis.
    if rvc.f0_extractor_type == F0ExtractorType.rmvpe:
        f0_session = create_session(
            rvc.rmvpe_model_file, device, log_severity=_ORT_LOG_ERROR
        )
    elif rvc.f0_extractor_type == F0ExtractorType.fcpe:
        f0_session = create_session(
            rvc.fcpe_model_file, device, log_severity=_ORT_LOG_ERROR
        )
    else:
        f0_session = None
    # The VAD noise gate (off by default). Opens the same silero_vad.onnx as the utterance
    # path's [vc] on CPU (reusing vspeech/lib/vad.py read-only). It is called inside the
    # worker_startup scope, so a missing or corrupt model fails loud at startup
    # (ADR-0038).
    if sv_config.vad_gate:
        from vspeech.lib.vad import create_vad_session

        vad_session = create_vad_session(sv_config.vad_model_file)
        logger.info("stream_vc vad gate enabled: %s", sv_config.vad_model_file)
    else:
        vad_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc,
        "device": device,
        "hubert_model": hubert_model,
        "session": session,
        "f0_session": f0_session,
        "vad_session": vad_session,
        "target_sample_rate": metadata["samplingRate"],
        "f0_enabled": metadata["f0"],
        "emb_output_layer": metadata.get("embOutputLayer", 9),
        "use_final_proj": metadata.get("useFinalProj", True),
    }


def make_streaming_vc(rt: dict[str, Any], sv_config: StreamVcConfig) -> StreamingVc:
    from vspeech.lib.stream_vc import StreamingVc

    # rvc.quality (the utterance path's reflect-pad amount) is deliberately not applied:
    # it is fundamentally inapplicable to the fixed-block streaming core, because
    # streaming uses a real rolling left context (context_len) rather than reflect-pad, so
    # a pad amount is meaningless. input_boost, on the other hand, is honoured
    # symmetrically with the utterance path (applied to the block in vc_loop).
    # The analysis window is extended by lookahead_ms on top of context_ms, so that
    # buying right context does not eat into the left context the emit start sees
    # (ADR-0070). context_ms therefore keeps exactly the meaning it had before.
    return StreamingVc(
        rvc_config=rt["rvc_config"],
        device=rt["device"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_session=rt["f0_session"],
        target_sample_rate=rt["target_sample_rate"],
        f0_enabled=rt["f0_enabled"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        block_len=ms_to_samples(sv_config.block_ms),
        context_len=ms_to_samples(sv_config.context_ms + sv_config.lookahead_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
        sola_search_len=ms_to_samples(sv_config.sola_search_ms),
        lookahead_len=ms_to_samples(sv_config.lookahead_ms),
    )


def geometry_summary(
    sv_config: StreamVcConfig, emit_delay_samples: int, target_sample_rate: int
) -> str:
    """A one-line startup summary of the analysis window and the delays it implies.

    Pure, so the wording can be pinned without standing up a worker. Japanese because the
    reader is an operator reading the log (ADR-0064).
    """
    window_ms = sv_config.context_ms + sv_config.lookahead_ms + sv_config.block_ms
    delay_ms = emit_delay_samples * 1000.0 / target_sample_rate
    return (
        f"stream_vc geometry: 解析窓 {window_ms:.0f}ms "
        f"(context {sv_config.context_ms:.0f} + lookahead {sv_config.lookahead_ms:.0f}"
        f" + block {sv_config.block_ms:.0f}), emit 遅延 {delay_ms:.1f}ms "
        f"(うち lookahead 由来 {sv_config.lookahead_ms:.0f}ms)"
    )


async def vc_loop(
    context: SharedContext,
    sv_config: StreamVcConfig,
    in_queue: Queue[CaptureItem],
    transport: Transport,
    session_id: str,
    ready: Event,
) -> None:
    """Convert capture blocks and send them to the transport as StreamPackets.

    The subsystem lives outside Command routing but still respects the same global stop
    gate as the utterance path, `context.running`: while paused it stops consuming and
    converting, and capture's drop_oldest_put discards the backlog so paused audio never
    accumulates (ADR-0050).
    """
    with worker_startup("stream_vc"):
        # check_cuda_provider is reused by importing the pure helper from worker/vc.py
        # (relocating it would mean editing vc.py, which violates a non-goal; this follows
        # ADR-0050/0053's practice of reusing internals by import).
        from vspeech.worker.vc import check_cuda_provider

        rt = await to_thread(build_stream_vc_runtime, sv_config)
        check_cuda_provider(rt["session"].get_providers())
        # The f0 session has ORT's warnings silenced (see build_stream_vc_runtime), so a
        # provider fallback cannot be detected from the log. Check it explicitly here.
        # There are no false positives: create_session only requests CUDA when
        # device.type == "cuda", and a deliberate CPU run would already have failed on the
        # decoder check one line above.
        if rt["f0_session"] is not None:
            check_cuda_provider(rt["f0_session"].get_providers())
        sv = make_streaming_vc(rt, sv_config)
        # A warmup failure (failing to build the fixed-shape graph) is a startup failure
        # and is made fail-loud (ADR-0038). process_block inside the loop is not guarded,
        # so failing here as a WorkerStartupError is the right thing.
        await to_thread(sv.warmup)
    logger.info(
        "%s",
        geometry_summary(sv_config, sv.emit_delay_samples, rt["target_sample_rate"]),
    )
    logger.info("stream vc worker started")
    # only now does capture open the mic (preventing the startup drop storm)
    ready.set()
    hop_seconds = sv_config.block_ms / 1000.0
    sample_rate = rt["target_sample_rate"]
    vad_session = rt["vad_session"]
    gate: StreamingVadGate | None = None
    if vad_session is not None:
        from vspeech.stream_vc.gate import StreamingVadGate

        gate = StreamingVadGate(
            threshold=sv_config.vad_threshold,
            hangover_ms=sv_config.vad_hangover_ms,
            min_gain=sv_config.vad_min_gain,
        )
    envelope = make_stream_envelope(sv_config)
    seq = 0
    consecutive_errors = 0
    # Throttle warnings about transient process_block drops by time (ADR-0062). This is a
    # different thing from the consecutive-failure tear-down decision
    # (consecutive_errors / _MAX_CONSECUTIVE_VC_ERRORS), so do not mix them. Telemetry
    # (stream_vc_process_error) is recorded on every drop.
    vc_error_throttle = LogThrottle()
    # Separate episode bookkeeping for the VAD gate's fail-open (a different condition
    # from a process_block drop, so it must not share an episode with it).
    vad_error_throttle = LogThrottle()
    try:
        while True:
            block = await in_queue.get()
            # The sentinel marking the boundary where capture reopened the device (capture
            # and the runner are separate tasks and capture's on_reopen cannot touch sv
            # directly, hence the in-band signal). Up to this point sv is still holding a
            # rolling context and crossfade tail from seconds ago, and crossfading the
            # fresh post-reopen block against those clicks at the seam. As with
            # pause/resume, discard the context and the VAD gate so the next fresh block
            # starts from silence. The sentinel is not audio, so skip conversion and
            # continue.
            if block is CaptureSignal.REOPEN:
                sv._reset_context()
                if gate is not None:
                    gate.reset()
                if envelope is not None:
                    envelope.reset()
                telemetry.record("stream_vc_capture_reopen_reset", 1.0)
                continue
            # The global stop gate (the same idiom as the utterance path's
            # worker/playback.py). While paused, stop consuming and converting -- capture
            # keeps running and drop_oldest_put discards the backlog, so paused audio does
            # not accumulate. The block already taken by get() straddles the pause and is
            # stale, so after resuming it is discarded and we start from the next fresh
            # block.
            if not context.running.is_set():
                await context.running.wait()
                # The not-set -> set transition (= resume). Real time has jumped, so
                # discard the rolling context and crossfade tail (_reset_context) and the
                # VAD gate, and let the first post-resume block fade in from silence
                # rather than from the pre-pause tail.
                sv._reset_context()
                if gate is not None:
                    gate.reset()
                if envelope is not None:
                    envelope.reset()
                continue
            # The gate/envelope decisions are made on the raw block, **before**
            # input_boost (deciding and shaping at the actual mic level rather than the
            # apparent post-boost level). Keep the raw block, then boost (safe even on
            # the boost==1.0 identity fast path).
            raw_block = block
            gains = None
            if gate is not None:
                gains = await gate_window_gains(
                    gate, vad_session, raw_block, vad_error_throttle
                )
            block = apply_input_boost(raw_block, sv_config.rvc.input_boost)
            t0 = perf_counter()
            # Transient GPU errors (CUDA errors, OOM, ...) surface as a RuntimeError from
            # torch/CUDA (torch.cuda.OutOfMemoryError also derives from RuntimeError).
            # **Do not tear down: drop one block and carry on**:
            #   - Retrying CUDA OOM in a tight loop thrashes.
            #   - A one-off is recoverable, so dropping one block is enough. Tearing down
            #     here (as the raise does once consecutive failures reach _MAX_) would
            #     take the whole process down through the inner TaskGroup and main's outer
            #     TaskGroup, dragging the utterance path with it. That is the **intended**
            #     fail-loud when an opt-in feature hits an unrecoverable fault (a daemon
            #     restarts it; ADR-0050), but it is excessive for a one-off transient.
            #   - _reset_context is unnecessary either -- when process_block raises inside
            #     infer it does not update self._context, so the next successful block
            #     continues from the last good context. The audio is one block short, and
            #     the next tick's SOLA crossfades two spans that are genuinely
            #     discontinuous, so a single faint click can occasionally remain (the
            #     crossfade does not hide this transparently). But OOM is rare and
            #     resetting here would not improve it, so it is accepted. seq is not
            #     advanced either (never disguise a loss to playback).
            #   - **The VAD gate's `_prev_gains` is not updated either** (automatically so,
            #     since apply is never reached). That is intended: because process_block
            #     leaves self._context untouched when infer raises, the head of the next
            #     successful emit (the delay portion) is a re-render of **the tail of the
            #     last successful block**, not of the dropped one. Measured (dropping
            #     block 4 with a synthetic input carrying f0 markers): the f0 over the
            #     first 45ms of the next emit is 522Hz, matching block 3's expected 530Hz,
            #     not the dropped block 4's 699Hz. So the retained `_prev_gains` (block 3's
            #     mask) lands on the right audio. Advancing the mask alone here would apply
            #     the dropped block's mask to different audio.
            # If failures do keep coming, fail rather than spin silently (see _MAX_ below).
            # Native ORT exceptions (onnxruntime's Fail/RuntimeException) do not derive
            # from RuntimeError and are therefore **not caught** -- those are usually
            # permanent graph/model defects where fail-loud is right. No broad
            # except Exception here.
            try:
                out_i16 = await to_thread(sv.process_block, block)
            except RuntimeError as e:
                consecutive_errors += 1
                telemetry.record("stream_vc_process_error", 1.0)
                if (n := vc_error_throttle.hit()) is not None:
                    logger.warning(
                        "stream_vc process_block failed; dropping block (total %d): %r",
                        n,
                        e,
                    )
                if consecutive_errors >= _MAX_CONSECUTIVE_VC_ERRORS:
                    logger.error(
                        "stream_vc: process_block failed %d times consecutively — "
                        "treating this as an unrecoverable fault in an explicitly-"
                        "enabled feature and failing the whole process on purpose "
                        "(fail-loud; a supervisor/daemon is expected to restart it)",
                        consecutive_errors,
                    )
                    raise
                continue
            # Only the consecutive-failure counter is reset on recovery (it drives the
            # tear-down decision). The warning thinning is left alone: LogThrottle tracks
            # episodes itself, so alternating fail/success does not warn on every drop.
            consecutive_errors = 0
            telemetry.record("stream_vc", perf_counter() - t0)
            # Input envelope following (ADR-0057) first, then the VAD gate (the same order
            # as the batch apply_input_envelope). The envelope is cheap numpy work, so it
            # runs inline (no to_thread needed). Both take the same emit delay: they
            # overlay the same emit in time alignment, so neither may skip the correction
            # (ADR-0065).
            if envelope is not None:
                out_i16 = envelope.apply(out_i16, raw_block, sv.emit_delay_samples)
            if gate is not None and gains is not None:
                # The mask is overlaid with the emit delay corrected (ADR-0059). The delay
                # is derived from the nominal read position and is constant across ticks
                # (SOLA's lag is excluded -- including it would re-anchor the mask's time
                # axis every tick and make the gain jump at the seam).
                out_i16 = gate.apply(out_i16, gains, sv.emit_delay_samples, sample_rate)
                if float(gains.min()) < 1.0:
                    telemetry.record("stream_vc_vad_gated", 1.0)
            packet = make_stream_packet(
                session_id, seq, hop_seconds, out_i16.tobytes(), sample_rate
            )
            if not await transport.send(packet):
                telemetry.record("stream_vc_send_drop", 1.0)
            seq += 1
    except CancelledError as e:
        raise shutdown_worker(e)
