"""streaming VC の VAD ノイズゲート(vspeech/stream_vc/gate.py)。

窓単位マスクと emit 遅延補正はモデル非依存の純ロジックなので CPU・onnxruntime
無しで走る。末尾の vc_loop 配線テストも実モデルを差し替えて CPU で回す。
"""

import numpy as np
import pytest

from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import VAD_WINDOW_SAMPLES
from vspeech.stream_vc.gate import StreamingVadGate
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport

_RATE = 48000
# 出力サンプルでの 1 窓ぶん(32ms)。マスク補間の格子間隔。
_STEP = round(VAD_WINDOW_SAMPLES * _RATE / VAD_SAMPLE_RATE)


def _gate(**kw) -> StreamingVadGate:
    params: dict = {"threshold": 0.5, "hangover_ms": 300.0, "min_gain": 0.0}
    params.update(kw)
    return StreamingVadGate(**params)


# --- 窓単位マスク -----------------------------------------------------------


def test_window_gains_duck_non_speech_windows_inside_a_speech_block():
    """発話を含むブロックでも、その中の非音声窓は個別に落とす。

    実録音の語頭ブロック(t=3.52s)で実測した確率列。ブロック粒度(窓確率の max)
    判定だと 1 窓 (0.868) のせいでブロック全体が 1.0 で通り、HuBERT の窓正規化が
    持ち上げた発声直前の微小入力がブレスとして出ていた(batch 経路比 +41dB)。
    """
    g = _gate(threshold=0.3, hangover_ms=0.0, min_gain=0.0)
    probs = np.array([0.030, 0.222, 0.140, 0.082, 0.868])
    assert list(g.window_gains(probs)) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_gate_starts_closed_so_silence_never_leaks():
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    assert list(g.window_gains(np.zeros(3))) == [0.0, 0.0, 0.0]


def test_hangover_holds_the_gate_open_after_speech():
    g = _gate(hangover_ms=2 * 32.0, min_gain=0.0)  # 2 窓ぶん
    gains = g.window_gains(np.array([0.9, 0.1, 0.1, 0.1]))
    assert list(gains) == [1.0, 1.0, 1.0, 0.0]


def test_hangover_does_not_open_windows_before_speech():
    """前方へは dilate しない。

    バッチ側 speech_gate_mask は前後対称に dilate するが、streaming で前方へ広げると
    語頭直前のブレスをそのまま開けてしまう(実測: 前方 32ms で +15dB -> +32dB へ悪化)。
    """
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    assert list(g.window_gains(np.array([0.1, 0.1, 0.9]))) == [0.0, 0.0, 1.0]


def test_hangover_budget_carries_across_blocks():
    """hangover はブロック境界をまたいで持ち越す(判定はブロックごとに来る)。"""
    g = _gate(hangover_ms=2 * 32.0, min_gain=0.0)
    g.window_gains(np.array([0.9]))
    assert list(g.window_gains(np.array([0.1, 0.1, 0.1]))) == [1.0, 1.0, 0.0]


def test_min_gain_is_the_closed_gain():
    g = _gate(hangover_ms=0.0, min_gain=0.25)
    assert list(g.window_gains(np.zeros(2))) == [0.25, 0.25]


# --- emit への適用 ----------------------------------------------------------


def test_apply_is_bit_identical_when_every_window_is_open():
    """常時 speech(と既定 off)では出力が無ゲート時とビット単位で一致する。"""
    g = _gate()
    out = np.array([100, -200, 300], dtype=np.int16)
    assert g.apply(out, np.ones(5), 0, _RATE) is out


def test_apply_shifts_the_mask_by_the_emit_delay():
    """マスクは emit 遅延ぶんずらして重ねる。

    emit の内容は入力ブロックより手前から始まる(crossfade + SOLA + HuBERT 受容野で
    実測 ~52ms)。補正しないとゲート判定が 52ms ずれた音声に当たり、ブレス抑圧が
    -26dB から -8dB まで落ちる。
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


def test_apply_ramps_across_a_window_without_a_step():
    """ゲイン遷移は窓間隔(32ms)で線形に渡る = 段差(クリック)を作らない。"""
    amp = 10000
    block = np.full(4 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    out = g.apply(block, np.array([0.0, 0.0, 1.0, 1.0]), 0, _RATE).astype(np.float64)
    assert np.max(np.abs(np.diff(out))) <= 2.0 * amp / _STEP


def test_apply_carries_the_previous_block_mask_across_the_boundary():
    """前ブロック末尾のゲインから連続させる(ブロック境界に段差を作らない)。"""
    amp = 10000
    block = np.full(4 * _STEP, amp, dtype=np.int16)
    g = _gate(min_gain=0.0)
    g.apply(block, np.zeros(4), 0, _RATE)  # 全閉
    reopened = g.apply(block, np.ones(4), 0, _RATE).astype(np.float64)
    # 直前が閉なので先頭は途中の値から立ち上がる(いきなり full にはならない)。
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
    """reset() は閉じた状態へ戻すが、warned(fail-open の障害フラグ)は触らない。"""
    g = _gate(hangover_ms=300.0, min_gain=0.0)
    g.window_gains(np.array([0.9]))
    g.apply(np.full(_STEP, 100, dtype=np.int16), np.ones(1), 0, _RATE)
    g.warned = True
    g.reset()
    assert g._prev_gains is None
    assert list(g.window_gains(np.zeros(2))) == [0.0, 0.0]  # hangover を持ち越さない
    assert g.warned is True


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
        self.emit_delay_samples = 0

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

    async def recv(self) -> StreamPacket:  # pragma: no cover - vc_loop は使わない
        raise NotImplementedError


def _context():
    """running が set(= 非 pause)な最小 SharedContext。"""
    from vspeech.config import Config
    from vspeech.shared_context import SharedContext

    return SharedContext(config=Config())


async def _run_vc_loop(monkeypatch, sv_config, vad_session, n_blocks: int):
    """vc_loop を n_blocks 個だけ回し、(transport, 適用された窓ゲイン列) を返す。"""
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
        runner_mod, "make_streaming_vc", lambda _rt, _cfg: _FakeStreamingVc()
    )

    applied: list[list[float]] = []
    real_apply = StreamingVadGate.apply

    def spy_apply(self, out_i16, gains, delay_samples, sample_rate):
        applied.append([float(x) for x in gains])
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
        task.result()  # 起動時例外はそのまま浮かせる
    task.cancel()
    try:
        await task
    except BaseException:
        pass
    assert len(transport.packets) == n_blocks
    return transport, applied


async def test_default_off_never_applies_the_gate(monkeypatch):
    """vad_gate=False では gate を作らず apply も一切通らない = ビット単位で同一。"""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig()
    assert sv.vad_gate is False  # 既定 off
    transport, applied = await _run_vc_loop(monkeypatch, sv, None, 2)
    assert applied == []  # apply は一度も呼ばれない
    assert transport.packets[0].pcm == _VC_OUT.tobytes()  # 無ゲート出力そのもの
    assert transport.packets[1].pcm == _VC_OUT.tobytes()


async def test_gate_enabled_attenuates_silent_blocks(monkeypatch):
    """vad_gate=True + 無音判定でゲートが閉じ、出力が減衰する。"""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(
        vad_gate=True, vad_hangover_ms=0.0, vad_min_gain=0.0, block_ms=160.0
    )
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs", lambda _session, _audio: np.zeros(5)
    )
    transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert applied == [[0.0] * 5, [0.0] * 5]  # 全窓 min_gain
    for packet in transport.packets:
        assert not np.frombuffer(packet.pcm, dtype=np.int16).any()


async def test_gate_open_on_speech_is_bit_identical(monkeypatch):
    """speech 判定が続くあいだは恒等路で無ゲート出力と一致する。"""
    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)
    monkeypatch.setattr(
        "vspeech.lib.vad.speech_probs", lambda _session, _audio: np.full(5, 0.99)
    )
    transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 2)
    assert applied == [[1.0] * 5, [1.0] * 5]
    for p in transport.packets:
        assert p.pcm == _VC_OUT.tobytes()


async def test_gate_failure_is_fail_open_and_warns_once(monkeypatch, caplog):
    """VAD が失敗しても音は素通し、警告はブロック毎ではなく 1 回だけ。"""
    import logging

    from vspeech.config import StreamVcConfig

    sv = StreamVcConfig(vad_gate=True)

    def boom(_session, _audio):
        raise RuntimeError("vad exploded")

    monkeypatch.setattr("vspeech.lib.vad.speech_probs", boom)
    with caplog.at_level(logging.WARNING):
        transport, applied = await _run_vc_loop(monkeypatch, sv, object(), 3)
    assert applied == [[1.0], [1.0], [1.0]]  # fail-open: 全開のマスク
    for p in transport.packets:
        assert p.pcm == _VC_OUT.tobytes()  # 素通し(恒等の高速路)
    warnings = [r for r in caplog.records if "vad gate failed" in r.getMessage()]
    assert len(warnings) == 1


# --- pause/resume ゲート ----------------------------------------------------
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
    monkeypatch.setattr(runner_mod, "make_streaming_vc", lambda _rt, _cfg: fake)
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


# --- capture 再 open の番兵 -------------------------------------------------
#
# capture が device 再 open すると capture_queue に CaptureSignal.REOPEN 番兵が入る。
# runner はそれを見たら文脈と VAD ゲートを reset し、番兵自体は変換しない(pause と
# 同じ扱い)。実モデルを差し替えて CPU で検証する。


async def test_capture_reopen_sentinel_resets_context_and_gate(monkeypatch):
    """再 open 番兵は変換されず、_reset_context + gate.reset を呼ぶ。"""
    import asyncio
    from asyncio import Event
    from asyncio import Queue

    from vspeech.config import StreamVcConfig
    from vspeech.stream_vc import runner as runner_mod
    from vspeech.stream_vc.capture import CaptureSignal

    fake = _FakeStreamingVc()
    # gate 有効(vad_session != None)にして gate.reset も検証する。
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
        "vspeech.lib.vad.speech_probs", lambda _session, _audio: np.full(5, 0.99)
    )
    reset_calls: list[int] = []
    real_reset = StreamingVadGate.reset

    def spy_reset(self):
        reset_calls.append(1)
        return real_reset(self)

    monkeypatch.setattr(StreamingVadGate, "reset", spy_reset)

    in_queue: Queue = Queue()
    in_queue.put_nowait(CaptureSignal.REOPEN)  # 再 open 番兵(先頭)
    in_queue.put_nowait(np.zeros(2560, dtype=np.float32))  # 続く fresh block
    transport = _CollectTransport()
    task = asyncio.create_task(
        runner_mod.vc_loop(_context(), sv, in_queue, transport, "sess", Event())
    )
    for _ in range(2000):
        await asyncio.sleep(0)
        if len(transport.packets) >= 1 or task.done():
            break
    if task.done():
        task.result()  # 起動時例外はそのまま浮かせる
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    assert fake.resets == 1  # 番兵で _reset_context を 1 回
    assert reset_calls == [1]  # gate.reset も 1 回
    assert len(transport.packets) == 1  # 番兵は packet を生まない(fresh block だけ)
    assert transport.packets[0].pcm == _VC_OUT.tobytes()
