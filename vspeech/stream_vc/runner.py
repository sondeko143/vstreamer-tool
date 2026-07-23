"""streaming VC の変換ループ(ADR-0053)。

capture の float32 ブロックを StreamingVc(固定ブロック+左文脈+等電力
クロスフェード)で変換し、StreamPacket にして transport へ送る。モデルの構築は
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
from vspeech.stream_vc.capture import ms_to_samples
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vspeech.lib.stream_vc import StreamingVc

# ORT のログ閾値: 0=VERBOSE / 1=INFO / 2=WARNING(既定) / 3=ERROR / 4=FATAL
_ORT_LOG_ERROR = 3


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
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc,
        "device": device,
        "hubert_model": hubert_model,
        "session": session,
        "f0_session": f0_session,
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
    sv_config: StreamVcConfig,
    in_queue: Queue[NDArray[np.float32]],
    transport: Transport,
    session_id: str,
    ready: Event,
) -> None:
    """capture ブロックを変換し StreamPacket として transport へ送る。"""
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
    seq = 0
    try:
        while True:
            block = await in_queue.get()
            block = apply_input_boost(block, sv_config.rvc.input_boost)
            t0 = perf_counter()
            out_i16 = await to_thread(sv.process_block, block)
            telemetry.record("stream_vc", perf_counter() - t0)
            packet = make_stream_packet(
                session_id, seq, hop_seconds, out_i16.tobytes(), sample_rate
            )
            if not await transport.send(packet):
                telemetry.record("stream_vc_send_drop", 1.0)
            seq += 1
    except CancelledError as e:
        raise shutdown_worker(e)
