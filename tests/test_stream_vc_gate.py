"""streaming VC の VAD ノイズゲート(vspeech/stream_vc/gate.py)。

ステートマシンと ramp はモデル非依存の純ロジックなので CPU・onnxruntime 無しで
走る。末尾の vc_loop 配線テストも実モデルを差し替えて CPU で回す。
"""

import numpy as np

from vspeech.stream_vc.gate import StreamingVadGate
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport


def _gate(**kw) -> StreamingVadGate:
    params: dict = {
        "threshold": 0.5,
        "hangover_ms": 300.0,
        "min_gain": 0.0,
        "block_ms": 100.0,
    }
    params.update(kw)
    return StreamingVadGate(**params)


def test_speech_opens_gate():
    g = _gate()
    assert g.update(True) == 1.0


def test_silence_within_hangover_stays_open():
    g = _gate(hangover_ms=300.0, block_ms=100.0)
    g.update(True)
    # 300ms の予算を 100ms ずつ食う: 200 / 100 が残るあいだは開いたまま。
    assert g.update(False) == 1.0
    assert g.update(False) == 1.0


def test_silence_beyond_hangover_closes_to_min_gain():
    g = _gate(hangover_ms=300.0, block_ms=100.0, min_gain=0.0)
    g.update(True)
    g.update(False)
    g.update(False)
    assert g.update(False) == 0.0  # 予算 0 -> 閉じる
    assert g.update(False) == 0.0  # 閉じたまま(予算は負に走らない)


def test_min_gain_is_the_closed_gain():
    g = _gate(hangover_ms=0.0, min_gain=0.25)
    assert g.update(False) == 0.25


def test_speech_burst_resets_hangover():
    g = _gate(hangover_ms=300.0, block_ms=100.0)
    g.update(True)
    for _ in range(5):
        g.update(False)
    assert g.update(False) == 0.0  # 閉じている
    assert g.update(True) == 1.0  # 発話が戻ると即開く
    # hangover 予算が満額に戻っているので、また 2 ブロック分は開いたまま。
    assert g.update(False) == 1.0
    assert g.update(False) == 1.0
    assert g.update(False) == 0.0


def test_speech_from_probs_uses_max_over_windows():
    g = _gate(threshold=0.5)
    # ブロック末尾で始まった発話頭(1 窓だけ)も speech と見なす。
    assert g.speech_from_probs(np.array([0.01, 0.02, 0.03, 0.9])) is True
    assert g.speech_from_probs(np.array([0.01, 0.02, 0.49])) is False
    assert g.speech_from_probs(np.zeros(0)) is False


def test_ramp_identity_fast_path_returns_input_unchanged():
    g = _gate()
    block = np.array([100, -200, 300], dtype=np.int16)
    out = g.ramp(block, 1.0)
    assert out is block  # 既定 off / 常時 speech ではビット単位で同一


def test_ramp_has_no_step_discontinuity():
    g = _gate(min_gain=0.0)
    n = 800
    amp = 10000
    block = np.full(n, amp, dtype=np.int16)  # 定数入力 -> 差分 = ゲイン差分のみ
    out = g.ramp(block, 0.0).astype(np.float64)
    # 先頭は入力そのまま、末尾はゼロ。
    assert abs(out[0] - amp) <= 1.0
    assert abs(out[-1]) <= 1.0
    # 段差なし: サンプル間の跳びは amp/n の定数倍で抑えられる。
    assert np.max(np.abs(np.diff(out))) <= 2.0 * amp / n


def test_ramp_resumes_from_previous_block_end_gain():
    g = _gate(min_gain=0.0)
    n = 400
    amp = 10000
    block = np.full(n, amp, dtype=np.int16)
    g.ramp(block, 0.0)  # 1.0 -> 0.0 で閉じ切る
    out = g.ramp(block, 1.0).astype(np.float64)  # 0.0 -> 1.0 で開く
    assert abs(out[0]) <= 1.0  # 前ブロック終端(0.0)から連続
    assert abs(out[-1] - amp) <= 1.0
    assert np.max(np.abs(np.diff(out))) <= 2.0 * amp / n


def test_ramp_preserves_dtype_and_clips_to_int16_range():
    g = _gate(min_gain=1.0)
    g._gain = 1.2  # 過大ゲインを強制して clip 経路を通す
    block = np.full(64, 32767, dtype=np.int16)
    out = g.ramp(block, 1.0)
    assert out.dtype == np.int16
    assert out.max() <= 32767
    assert out.min() >= -32768


def test_ramp_handles_empty_block():
    g = _gate()
    empty = np.zeros(0, dtype=np.int16)
    out = g.ramp(empty, 0.0)
    assert out.dtype == np.int16
    assert out.shape[0] == 0


def test_reset_restores_open_and_empty_hangover_but_keeps_warned():
    """reset() は構築直後(開いた状態・hangover 空)へ戻すが、warned は触らない。"""
    g = _gate(hangover_ms=300.0, block_ms=100.0, min_gain=0.0)
    g.update(True)
    g.update(False)
    g._gain = 0.3
    g.warned = True
    g.reset()
    assert g._gain == 1.0
    assert g._hangover_remaining_ms == 0.0
    assert g.warned is True  # fail-open の障害フラグは保持


# --- vc_loop の配線 ---------------------------------------------------------
#
# 実モデル/GPU を読まずに vc_loop を回すため build_stream_vc_runtime と
# make_streaming_vc だけ差し替える。ゲートの有無を決める分岐
# (`vad_session is None` なら gate を作らない)は本物のコードを通る。

_VC_OUT = np.array([1000, -2000, 3000, -4000, 5000, -6000], dtype=np.int16)


class _FakeStreamingVc:
    """process_block が決め打ちの int16 ブロックを返す実モデル代役。"""

    def __init__(self) -> None:
        self.warmed = 0
        self.resets = 0

    def warmup(self, n: int = 3) -> None:
        self.warmed += 1

    def process_block(self, block):
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

    async def recv(self) -> StreamPacket:  # pragma: no cover - vc_loop は使わない
        raise NotImplementedError


def _context():
    """running が set(= 非 pause)な最小 SharedContext。"""
    from vspeech.config import Config
    from vspeech.shared_context import SharedContext

    return SharedContext(config=Config())


async def _run_vc_loop(monkeypatch, sv_config, vad_session, n_blocks: int):
    """vc_loop を n_blocks 個だけ回し、(transport, ramp 目標ゲインの記録) を返す。"""
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
        runner_mod, "make_streaming_vc", lambda rt, cfg: _FakeStreamingVc()
    )

    ramp_calls: list[float] = []
    real_ramp = StreamingVadGate.ramp

    def spy_ramp(self, out_i16, target_gain):
        ramp_calls.append(target_gain)
        return real_ramp(self, out_i16, target_gain)

    monkeypatch.setattr(StreamingVadGate, "ramp", spy_ramp)

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
        task.result()  # 起動時例外はそのまま浮かせる
    task.cancel()
    try:
        await task
    except BaseException:
        pass
    assert len(transport.packets) == n_blocks
    return transport, ramp_calls


async def test_default_off_never_applies_the_gate(monkeypatch):
    """vad_gate=False では gate を作らず ramp も一切通らない = ビット単位で同一。"""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig()
    assert sv.vad_gate is False  # 既定 off
    transport, ramp_calls = await _run_vc_loop(monkeypatch, sv, None, 2)
    assert ramp_calls == []  # ramp は一度も呼ばれない
    assert transport.packets[0].pcm == _VC_OUT.tobytes()  # 無ゲート出力そのもの
    assert transport.packets[1].pcm == _VC_OUT.tobytes()


async def test_gate_enabled_attenuates_silent_blocks(monkeypatch):
    """vad_gate=True + 無音判定でゲートが閉じ、出力が減衰する。"""
    from vspeech.config import StreamVcConfig

    # hangover 0ms なので最初の無音ブロックで即閉じる。
    sv = StreamVcConfig(
        vad_gate=True, vad_hangover_ms=0.0, vad_min_gain=0.0, block_ms=160.0
    )
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs", lambda session, audio: np.zeros(5)
    )
    transport, ramp_calls = await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert ramp_calls == [0.0, 0.0]  # 目標ゲインは min_gain
    # 1 ブロック目は 1.0 -> 0.0 の ramp、2 ブロック目は完全に無音。
    first = np.frombuffer(transport.packets[0].pcm, dtype=np.int16)
    second = np.frombuffer(transport.packets[1].pcm, dtype=np.int16)
    assert np.abs(first).max() < np.abs(_VC_OUT).max()
    assert not second.any()


async def test_gate_open_on_speech_is_bit_identical(monkeypatch):
    """speech 判定が続くあいだは 1.0 -> 1.0 の恒等路で無ゲート出力と一致する。"""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs", lambda session, audio: np.full(5, 0.99)
    )
    transport, ramp_calls = await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert ramp_calls == [1.0, 1.0]
    for p in transport.packets:
        assert p.pcm == _VC_OUT.tobytes()


async def test_gate_failure_is_fail_open_and_warns_once(monkeypatch, caplog):
    """VAD が失敗しても音は素通し、警告はブロック毎ではなく 1 回だけ。"""
    import logging

    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)

    def boom(session, audio):
        raise RuntimeError("vad exploded")

    monkeypatch.setattr("vspeech.lib.vad.speech_probs", boom)
    with caplog.at_level(logging.WARNING):
        transport, ramp_calls = await _run_vc_loop(monkeypatch, sv, object(), 3)
    assert ramp_calls == [1.0, 1.0, 1.0]  # fail-open: 目標ゲインは常に 1.0
    for p in transport.packets:
        assert p.pcm == _VC_OUT.tobytes()  # 素通し(1.0 -> 1.0 は恒等の高速路)
    warnings = [r for r in caplog.records if "vad gate failed" in r.getMessage()]
    assert len(warnings) == 1


# --- pause/resume ゲート(COMMIT 2) -----------------------------------------
#
# vc_loop は Command routing の外だが context.running を尊重する。実モデルを
# 差し替えて、pause 中は消費/変換が止まり、resume で _reset_context が呼ばれる
# ことを CPU で検証する。


def _patch_runtime(monkeypatch, fake):
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
            "vad_session": None,
            "target_sample_rate": 40000,
            "f0_enabled": True,
            "emb_output_layer": 9,
            "use_final_proj": True,
        },
    )
    monkeypatch.setattr(runner_mod, "make_streaming_vc", lambda rt, cfg: fake)
    return runner_mod


async def test_pause_stops_consuming_and_resets_on_resume(monkeypatch):
    """pause 中は vc_loop がブロックを変換せず、resume で _reset_context を 1 回呼ぶ。"""
    import asyncio
    from asyncio import Event
    from asyncio import Queue

    from vspeech.config import Config
    from vspeech.config import StreamVcConfig
    from vspeech.shared_context import SharedContext

    fake = _FakeStreamingVc()
    runner_mod = _patch_runtime(monkeypatch, fake)

    context = SharedContext(config=Config())  # 既定で running.set()
    sv = StreamVcConfig()
    in_queue: Queue = Queue()
    transport = _CollectTransport()
    ready = Event()
    task = asyncio.create_task(
        runner_mod.vc_loop(context, sv, in_queue, transport, "sess", ready)
    )
    # 起動(warmup, to_thread)完了まで待つ。
    await asyncio.wait_for(ready.wait(), timeout=5)

    # pause して 3 ブロック投入。ループは block0 を get したところで running.wait()
    # に park する(process_block/send はしない)。
    context.running.clear()
    for _ in range(3):
        in_queue.put_nowait(np.zeros(2560, dtype=np.float32))
    for _ in range(50):
        await asyncio.sleep(0)
    assert transport.packets == []  # paused: 一切変換していない
    assert in_queue.qsize() == 2  # block0 だけ消費して park(stale は resume で捨てる)
    assert fake.resets == 0

    # resume: park していた wait() が返り _reset_context → continue で block0 は捨て、
    # block1/block2 を変換する。
    context.running.set()
    for _ in range(2000):
        await asyncio.sleep(0)
        if len(transport.packets) >= 2:
            break
    assert fake.resets == 1  # resume 遷移で 1 回だけ
    assert len(transport.packets) == 2  # stale block0 は drop、block1/block2 のみ

    task.cancel()
    try:
        await task
    except BaseException:
        pass
