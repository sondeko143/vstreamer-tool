"""streaming VC の変換ループ(ADR-0053)。

capture の float32 ブロックを StreamingVc(固定ブロック+左文脈+クロスフェード)
で変換し、StreamPacket にして transport へ送る。モデルの構築は
発話系 rvc_worker(vspeech/worker/vc.py)と同じ手順を [stream_vc.rvc] から行う
(発話系は無改変)。重い import は関数内。
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
    from vspeech.stream_vc.gate import StreamingVadGate

# ORT のログ閾値: 0=VERBOSE / 1=INFO / 2=WARNING / 3=ERROR / 4=FATAL。
# SessionOptions().log_severity_level の既定は -1 = Env のレベルを継承 (onnx_session.py)。
_ORT_LOG_ERROR = 3

# process_block の transient GPU error をこの回数まで連続で許して drop する。
# 超えたら黙って spin せず落とす(下記 vc_loop の error handling 参照)。
_MAX_CONSECUTIVE_VC_ERRORS = 10

# transient な process_block drop の警告も underflow/drop 同様に間引く。fail/success が
# 交互だと reset-on-success 方式では毎 drop 警告が出て stdout(GUI の読むパイプ)を
# 埋める。連続失敗の tear-down 判定(_MAX_CONSECUTIVE_VC_ERRORS)とは別の通算カウンタで
# 絞る。telemetry(stream_vc_process_error)は毎 drop 記録する。
VC_ERROR_LOG_EVERY = 50


def should_log_vc_error(count: int) -> bool:
    """通算 count 回目の process_block drop をログに出すか(1 回目と以降 N 回ごと)。"""
    return count == 1 or count % VC_ERROR_LOG_EVERY == 0


def make_stream_packet(
    session_id: str, seq: int, hop_seconds: float, pcm: bytes, sample_rate: int
) -> StreamPacket:
    """seq/pts 付きの StreamPacket(pts = seq * hop_seconds)。"""
    return StreamPacket(
        session_id=session_id,
        seq=seq,
        pts=seq * hop_seconds,
        pcm=pcm,
        sample_rate=sample_rate,
    )


def apply_input_boost(block, boost):
    """入力ブロックに input_boost gain をかける([-1,1] へ clip = 発話系 vc.py の
    int16 `mul` 飽和相当)。発話系は `change_voice` の外(worker)で gain をかけるので、
    streaming も `StreamingVc` の外(この runner)でかけて対称にする。boost==1.0 は
    恒等(既定値なので、既定 config では挙動が変わらない)。"""
    import numpy as np

    if boost == 1.0:
        return block
    return np.clip(block * boost, -1.0, 1.0).astype(np.float32)


def make_stream_envelope(sv_config: StreamVcConfig):
    """envelope_follow のとき StreamingEnvelope を作る (off なら None)。純関数。"""
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


async def gate_target_gain(
    gate: StreamingVadGate, vad_session: Any, block: NDArray[np.float32]
) -> float:
    """**入力**ブロックの VAD 判定からこのブロックの目標ゲインを返す。

    判定は入力側(素のマイクレベル。input_boost をかける前 = 実際の S/N で
    判定する)、適用は出力側。推論そのものはスキップしない: `StreamingVc` は
    rolling 左文脈とクロスフェード tail を持つので、ブロックを飛ばすと文脈に
    穴が開き発話再開時の seam が壊れる(GPU 余力は実測 RTF 0.24 で十分)。

    ONNX 推論はブロッキングなので、発話系 worker/vc.py と同じく `to_thread`
    へ逃がす。失敗しても音は素通し(fail-open)で、警告は最初の 1 回だけ。
    """
    from vspeech.lib.vad import speech_probs

    try:
        probs = await to_thread(speech_probs, vad_session, block)
        return gate.update(gate.speech_from_probs(probs))
    except Exception as e:
        if not gate.warned:
            gate.warned = True
            logger.warning("stream_vc vad gate failed; passing audio ungated: %s", e)
        return 1.0


def build_stream_vc_runtime(sv_config: StreamVcConfig) -> dict[str, Any]:
    """[stream_vc.rvc] から device + モデル + metadata を構築する。"""
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
    # f0 セッションだけ ORT の警告を落とす。fcpe.onnx (poe export-fcpe-onnx, ADR-0049) は
    # torchfcpe を dynamic_axes 付きでトレースした都合で中間ノード /bundled/Squeeze_1 の
    # 推論 rank が実際と食い違い、ORT が VerifyOutputSizes 警告を **毎推論** stdout に出す。
    # 良性(実 shape で確保され f0 は正しい)だが streaming では ~6 行/秒になり、ログと
    # GUI が読む stdout パイプを埋める。グラフ側の修正は torchfcpe のトレース由来なので
    # graph surgery か上流パッチが要り、割に合わない(過去に ONNX graph surgery を
    # 試して徒労に終わっている)。
    # 代償: この f0 セッション固有の ORT 警告 (provider fallback 等) も見えなくなる。
    # そのため、握り潰したうちで実害のある「CUDA を要求したのに CPU へ落ちた」だけは
    # vc_loop 側の check_cuda_provider(f0_session) でプログラム的に捕まえる
    # (WorkerStartupError = fail-loud)。ログの間引きで診断を失わないための対。
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
    # VAD ノイズゲート(既定 off)。発話系 [vc] と同じ silero_vad.onnx を CPU で開く
    # (vspeech/lib/vad.py を読み取り専用で再利用)。worker_startup スコープ内で
    # 呼ばれるので、モデルが無い/壊れているときは起動時に fail-loud (ADR-0038)。
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

    # rvc.quality(発話系の reflect-pad 量)は固定ブロック streaming コアには本質的に
    # 非該当なので意図的に適用しない: streaming は reflect-pad ではなく実際のローリング
    # 左文脈(context_len)を使うため、pad 量というパラメタが意味を持たない。一方
    # input_boost は発話系と対称に honor する(vc_loop でブロックへ適用)。
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
        context_len=ms_to_samples(sv_config.context_ms),
        crossfade_len=ms_to_samples(sv_config.crossfade_ms),
        sola_search_len=ms_to_samples(sv_config.sola_search_ms),
    )


async def vc_loop(
    context: SharedContext,
    sv_config: StreamVcConfig,
    in_queue: Queue[CaptureItem],
    transport: Transport,
    session_id: str,
    ready: Event,
) -> None:
    """capture ブロックを変換し StreamPacket として transport へ送る。

    サブシステムは Command routing の外だが、発話系と同じ全体停止ゲート
    `context.running` は尊重する:pause 中は消費/変換を止め、capture の
    drop_oldest_put が backlog を捨てるので paused 音声は溜まらない(ADR-0050)。
    """
    with worker_startup("stream_vc"):
        # check_cuda_provider は worker/vc.py の pure helper を import 再利用
        # (relocate は vc.py 編集=非ゴール違反ゆえ不可; ADR-0050/0053 の内部部品
        # import 再利用に沿う)。
        from vspeech.worker.vc import check_cuda_provider

        rt = await to_thread(build_stream_vc_runtime, sv_config)
        check_cuda_provider(rt["session"].get_providers())
        # f0 セッションは ORT の警告を落としてある(build_stream_vc_runtime 参照)ので、
        # provider fallback をログでは検知できない。ここで明示的に見る。
        # 偽陽性は無い: create_session が CUDA を要求するのは device.type == "cuda"
        # のときだけで、意図的な CPU 実行なら 1 行上の decoder 検査で先に落ちる。
        if rt["f0_session"] is not None:
            check_cuda_provider(rt["f0_session"].get_providers())
        sv = make_streaming_vc(rt, sv_config)
        # warmup 失敗(固定 shape グラフ構築の失敗)は起動時失敗として fail-loud に
        # する(ADR-0038)。loop 内 process_block は guard していないので、ここで
        # 落として WorkerStartupError にするのが正しい。
        await to_thread(sv.warmup)
    logger.info("stream vc worker started")
    ready.set()  # ここで初めて capture がマイクを開く(起動時の drop 嵐を防ぐ)
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
            block_ms=sv_config.block_ms,
        )
    envelope = make_stream_envelope(sv_config)
    seq = 0
    consecutive_errors = 0
    vc_error_count = 0
    try:
        while True:
            block = await in_queue.get()
            # capture が device 再 open した境界の番兵(capture と runner は別タスクで
            # capture の on_reopen から sv へ直接触れないので帯域内で知らせる)。ここに
            # 至るまで sv は数秒前の rolling 文脈/クロスフェード tail を抱えたままで、
            # 再 open 直後の fresh block をそれと crossfade すると seam がプチる。
            # pause/resume と同じく文脈と VAD ゲートを捨て、次の fresh block を無音から
            # 始める。番兵は音声ではないので変換せず continue する。
            if block is CaptureSignal.REOPEN:
                sv._reset_context()
                if gate is not None:
                    gate.reset()
                if envelope is not None:
                    envelope.reset()
                telemetry.record("stream_vc_capture_reopen_reset", 1.0)
                continue
            # 全体停止ゲート(発話系 worker/playback.py と同じ idiom)。paused の間は
            # 消費/変換を止める — capture は回り続け drop_oldest_put が backlog を
            # 捨てるので paused 音声は溜まらない。get() 済みの block は pause 前後の
            # stale なものなので、resume 後は捨てて次の fresh block から始める。
            if not context.running.is_set():
                await context.running.wait()
                # not-set -> set の遷移(= resume)。実時間が飛んでいるので rolling
                # 文脈/クロスフェード tail(_reset_context)と VAD ゲートを捨て、
                # 最初の post-resume block を pre-pause の尾ではなく無音から fade-in
                # させる。
                sv._reset_context()
                if gate is not None:
                    gate.reset()
                if envelope is not None:
                    envelope.reset()
                continue
            # ゲート/エンベロープ判定は input_boost **前**の素のブロックで行う
            # (ブースト後の見かけのレベルではなく実際のマイクレベルで判定/整形する)。
            # raw を保持してから boost する (boost==1.0 の identity fast-path でも安全)。
            raw_block = block
            target_gain = 1.0
            if gate is not None:
                target_gain = await gate_target_gain(gate, vad_session, raw_block)
            block = apply_input_boost(raw_block, sv_config.rvc.input_boost)
            t0 = perf_counter()
            # transient GPU error(CUDA error / OOM 等)は torch/CUDA 由来の
            # RuntimeError(torch.cuda.OutOfMemoryError も RuntimeError 派生)で
            # 上がってくる。**tear down せず 1 ブロック drop して継続**する:
            #   - CUDA OOM を tight loop で retry すると thrash する。
            #   - 単発は回復可能なので 1 ブロック drop で十分。ここで tear down まで
            #     すると(= 連続失敗が _MAX_ に達したときの raise がそうなるように)
            #     内側 TaskGroup → main の外側 TaskGroup 経由でプロセスごと落ち、
            #     発話系も道連れになる。それは opt-in で有効化した機能が
            #     unrecoverable な障害を起こしたときに **意図した** fail-loud
            #     (daemon が再起動する; ADR-0050)だが、単発の transient には過剰。
            #   - _reset_context も不要 — process_block は infer で raise すると
            #     self._context を更新しないので、次の成功ブロックが直前の good な
            #     文脈から続く。drop したぶん音は 1 ブロック欠け、次tickの SOLA は
            #     本来非連続な 2 区間を crossfade するので、稀に 1 回のプチっとした
            #     不連続が残りうる(crossfade はこれを透明に隠すわけではない)。ただ
            #     OOM は稀で、ここで reset しても改善はしないので許容する。seq も
            #     進めない(欠落を playback に偽装しない)。
            # ただし連続失敗が続くなら黙って spin せず落とす(下 _MAX_...)。
            # ORT ネイティブ例外(onnxruntime の Fail/RuntimeException)は RuntimeError
            # 派生ではないので **捕えない** — あれは大抵グラフ/モデルの恒久的な不備で
            # fail-loud が正しい。broad な except Exception は使わない。
            try:
                out_i16 = await to_thread(sv.process_block, block)
            except RuntimeError as e:
                consecutive_errors += 1
                vc_error_count += 1
                telemetry.record("stream_vc_process_error", 1.0)
                if should_log_vc_error(vc_error_count):
                    logger.warning(
                        "stream_vc process_block failed; dropping block (total %d): %r",
                        vc_error_count,
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
            # 連続失敗カウンタだけ回復でリセットする(tear-down 判定用)。警告の間引きは
            # 通算カウンタ vc_error_count なので reset しない(fail/success 交互でも
            # 毎 drop 警告しないため)。
            consecutive_errors = 0
            telemetry.record("stream_vc", perf_counter() - t0)
            # 入力エンベロープ追従 (ADR-0057) → VAD ゲートの順 (バッチ apply_input_envelope
            # と同じ順)。envelope は安価な numpy 演算なので inline (to_thread 不要)。
            if envelope is not None:
                out_i16 = envelope.apply(out_i16, raw_block)
            if gate is not None:
                out_i16 = gate.ramp(out_i16, target_gain)
                if target_gain != 1.0:
                    telemetry.record("stream_vc_vad_gated", 1.0)
            packet = make_stream_packet(
                session_id, seq, hop_seconds, out_i16.tobytes(), sample_rate
            )
            if not await transport.send(packet):
                telemetry.record("stream_vc_send_drop", 1.0)
            seq += 1
    except CancelledError as e:
        raise shutdown_worker(e)
