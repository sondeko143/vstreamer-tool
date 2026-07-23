# RVC Streaming VC — M3 (2-Machine Split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the streaming VC subsystem across two machines — a producer (capture + VC + UDP send) on the GPU host and a consumer (UDP recv + jitter buffer + playback) on the playback host — so only converted audio crosses the wire, with the 発話系 running unmodified from the same mic.

**Architecture:** Add a raw-UDP transport behind the existing `Transport` ABC (ADR-0051 T3) and a `role` enum (`local | producer | consumer`, ADR-0055) that makes the one subsystem build a different loop set per machine. Reorder / loss / gap-fill / latency-bound move out of the transport into a transport-agnostic `JitterBuffer` on the consumer (ADR-0056). `role = local` (default) is byte-identical to M2.

**Tech Stack:** Python 3.14, asyncio (`create_datagram_endpoint`), `struct` for wire framing, numpy (concealment), sounddevice (output), pydantic v2 (config). No new third-party dependency.

## Global Constraints

- Python **3.14 only** (`>=3.14,<3.15`); package manager is **uv**; run everything via `uv run ...`.
- **Imports one-per-line** (ruff `force-single-line`); type-check with **ty**; format/lint with **ruff**. Gate: `uv run poe check` (exit 1 only on the pre-existing accepted findings — torch CVE + subtitle_tk/tts vr2_config deadcode; nothing new).
- **Pydantic v2 only** — `model_validator`, `Field`, `Enum` values; never v1 APIs.
- The **consumer role must import and run without torch / RVC / GPU**. Its modules (`udp.py`, `jitter.py`, `wire.py`, `consumer.py`, and `playback.py` helpers it reuses) must not import `vspeech.lib.stream_vc`, `vspeech.worker.vc`, `vspeech.stream_vc.runner`, or `vspeech.stream_vc.capture`. Verify with the forbidden-import test in Task 6.
- **`role = local` must stay byte-identical to M2** — the local path keeps `InProcessTransport` + the existing `playback_loop`, unchanged. No jitter buffer on the local path.
- Fail-loud rules unchanged (ADR-0050): first resource open is `worker_startup(...)` → `WorkerStartupError`; steady-state device faults self-heal via `vspeech/stream_vc/retry.py`; unrecoverable faults propagate → process exit. `CancelledError` is never swallowed (wrap with `shutdown_worker`).
- Never claim a cross-machine **one-way latency** number — the two machines have documented W32Time clock skew (ADR-0006 / topology memory). Measure only skew-immune quantities (inter-arrival jitter, seq-gap loss).
- ADRs of record: **0055** (role split), **0056** (jitter buffer + measurement) — both `Proposed`, promote to `Accepted` in Task 8 as the implementation backs them. **0051** flips `Proposed → Accepted` in Task 8 (a second transport now exists).

---

## File Structure

**New runtime modules (all torch-free):**
- `vspeech/stream_vc/wire.py` — UDP wire framing: `encode_packet` / `decode_packet` over a `struct` header. Pure, CPU-testable.
- `vspeech/stream_vc/jitter.py` — `JitterBuffer` (reorder / prebuffer / conceal / overflow-fast-forward / late-drop). Pure, CPU-testable.
- `vspeech/stream_vc/udp.py` — `UdpProducerTransport` / `UdpConsumerTransport` (`Transport` subclasses) + async factories over `create_datagram_endpoint`.
- `vspeech/stream_vc/consumer.py` — `network_playback_loop` (recv → jitter buffer → output write) reusing `playback.py` output helpers.

**Modified:**
- `vspeech/config.py` — `StreamVcRole` enum, `udp` in `TransportType`, fields `role` / `peer_host` / `peer_port` / `bind_host` / `bind_port` / `jitter_buffer_ms`.
- `vspeech/stream_vc/subsystem.py` — role-based transport + loop-set wiring.
- `vspeech/preflight.py` — role-aware `_check_stream_vc`.
- `config.toml.example` — document the new `[stream_vc]` keys.
- `docs/adr/0051-*.md`, `docs/adr/0055-*.md`, `docs/adr/0056-*.md`, `docs/adr/README.md` — status promotions.

**New tests:**
- `tests/test_stream_vc_wire.py`, `tests/test_stream_vc_jitter.py`, `tests/test_stream_vc_udp.py`, `tests/test_stream_vc_consumer.py`, and additions to `tests/test_stream_vc_subsystem.py` / `tests/test_preflight.py` / `tests/test_config*.py` (match the repo's existing test filenames — check with `ls tests | grep stream_vc` first and extend the matching file rather than duplicating).

---

### Task 1: Config — role, transport, addresses, jitter buffer

**Files:**
- Modify: `vspeech/config.py` (near `class TransportType` / `class StreamVcConfig`, lines ~386-492)
- Test: `tests/test_config_stream_vc.py` (create if absent; else extend the existing stream_vc config test)

**Interfaces:**
- Produces: `StreamVcRole` enum (`local="local"`, `producer="producer"`, `consumer="consumer"`); `TransportType.udp="udp"`; `StreamVcConfig.role: StreamVcRole = local`; `.peer_host: str | None`, `.peer_port: int | None`, `.bind_host: str`, `.bind_port: int | None`, `.jitter_buffer_ms: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_stream_vc.py
from vspeech.config import Config
from vspeech.config import StreamVcRole
from vspeech.config import TransportType


def test_stream_vc_defaults_are_local_in_process():
    sv = Config().stream_vc
    assert sv.role is StreamVcRole.local
    assert sv.transport_type is TransportType.in_process
    assert sv.jitter_buffer_ms == 0.0


def test_stream_vc_producer_parses_udp_and_peer():
    sv = Config.model_validate(
        {"stream_vc": {"role": "producer", "transport_type": "udp",
                       "peer_host": "playback-host", "peer_port": 9999}}
    ).stream_vc
    assert sv.role is StreamVcRole.producer
    assert sv.transport_type is TransportType.udp
    assert (sv.peer_host, sv.peer_port) == ("playback-host", 9999)


def test_stream_vc_consumer_parses_bind_and_jitter():
    sv = Config.model_validate(
        {"stream_vc": {"role": "consumer", "transport_type": "udp",
                       "bind_host": "0.0.0.0", "bind_port": 9999,
                       "jitter_buffer_ms": 60.0}}
    ).stream_vc
    assert sv.role is StreamVcRole.consumer
    assert (sv.bind_host, sv.bind_port) == ("0.0.0.0", 9999)
    assert sv.jitter_buffer_ms == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_stream_vc.py -v`
Expected: FAIL (`AttributeError` / `ImportError: cannot import name 'StreamVcRole'`).

- [ ] **Step 3: Implement the config changes**

In `vspeech/config.py`, add the role enum next to `TransportType` and the `udp` member:

```python
class TransportType(Enum):
    in_process = "in_process"
    udp = "udp"


class StreamVcRole(Enum):
    local = "local"  # M2: capture+vc+playback in one process (default, unchanged)
    producer = "producer"  # capture + vc + UDP send (GPU host)
    consumer = "consumer"  # UDP recv + jitter buffer + playback (no torch/GPU)
```

Add fields to `StreamVcConfig` (after `transport_type` / `max_queued_blocks`):

```python
    role: StreamVcRole = Field(
        default=StreamVcRole.local,
        description="local=M2 単一プロセス(既定)。producer=capture+vc+送信。"
        "consumer=受信+jitter buffer+再生(GPU/torch 不要)。ADR-0055",
    )
    # producer: 送信先。consumer: 待受。role=local では未使用。
    peer_host: str | None = Field(
        default=None, description="producer の送信先ホスト(consumer の bind と一致)"
    )
    peer_port: int | None = Field(
        default=None, gt=0, le=65535, description="producer の送信先ポート"
    )
    bind_host: str = Field(
        default="0.0.0.0", description="consumer の待受ホスト(既定 全 IF)"
    )
    bind_port: int | None = Field(
        default=None, gt=0, le=65535, description="consumer の待受ポート"
    )
    jitter_buffer_ms: float = Field(
        default=0.0,
        ge=0,
        description="consumer のジッタバッファ深さ ms。深さ=付加遅延なので既定は"
        "浅く保ち、実測ジッタから最小に詰める(ADR-0056)。round(jitter_buffer_ms/"
        "block_ms) ブロックを prebuffer してから再生を始める",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_stream_vc.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vspeech/config.py tests/test_config_stream_vc.py
git commit -m "feat(stream-vc): add role/udp/address/jitter config for M3 split (ADR-0055)"
```

---

### Task 2: UDP wire framing (`wire.py`)

**Files:**
- Create: `vspeech/stream_vc/wire.py`
- Test: `tests/test_stream_vc_wire.py`

**Interfaces:**
- Consumes: `StreamPacket` from `vspeech/stream_vc/packet.py` (`session_id: str` 32-hex, `seq: int`, `pts: float`, `pcm: bytes`, `sample_rate: int`).
- Produces: `encode_packet(p: StreamPacket) -> bytes`; `decode_packet(data: bytes) -> StreamPacket`; `class WireError(ValueError)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_wire.py
import pytest

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet


def _packet(seq=7, pcm=b"\x01\x02\x03\x04"):
    return StreamPacket(
        session_id="0123456789abcdef0123456789abcdef",
        seq=seq, pts=1.25, pcm=pcm, sample_rate=48000,
    )


def test_round_trip_preserves_all_fields():
    p = _packet()
    got = decode_packet(encode_packet(p))
    assert got == p


def test_round_trip_empty_and_large_pcm():
    for pcm in (b"", bytes(range(256)) * 60):  # ~15KB, > MTU
        p = _packet(pcm=pcm)
        assert decode_packet(encode_packet(p)).pcm == pcm


def test_decode_rejects_short_or_bad_magic():
    with pytest.raises(WireError):
        decode_packet(b"too-short")
    good = bytearray(encode_packet(_packet()))
    good[0] = ord("X")  # corrupt magic
    with pytest.raises(WireError):
        decode_packet(bytes(good))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_wire.py -v`
Expected: FAIL (`ModuleNotFoundError: vspeech.stream_vc.wire`).

- [ ] **Step 3: Implement `wire.py`**

```python
"""ストリーミング VC の UDP ワイヤ書式(ADR-0051 T3)。

固定ヘッダ + PCM ペイロードの 1 ブロック 1 データグラム。session_id は 32-hex を
16 バイト生 UUID に詰める。1 ブロックは MTU(1500)を超える(160ms/48kHz int16 で
~15KB)が UDP の 64KB 上限には収まるので、そのまま送って IP 層に断片化させる
(断片欠落はブロック単位の loss = seq gap として観測される, ADR-0056)。
"""

from __future__ import annotations

import struct

from vspeech.stream_vc.packet import StreamPacket

_MAGIC = b"SV"
_VERSION = 1
# network byte order: magic(2s) version(B) flags(B) session(16s) seq(Q) pts(d) rate(I)
_HEADER = struct.Struct("!2sBB16sQdI")


class WireError(ValueError):
    """データグラムが本コーデックの書式でない/壊れている。"""


def encode_packet(p: StreamPacket) -> bytes:
    header = _HEADER.pack(
        _MAGIC, _VERSION, 0, bytes.fromhex(p.session_id), p.seq, p.pts, p.sample_rate
    )
    return header + p.pcm


def decode_packet(data: bytes) -> StreamPacket:
    if len(data) < _HEADER.size:
        raise WireError(f"datagram too short: {len(data)} < {_HEADER.size}")
    magic, version, _flags, session, seq, pts, rate = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION:
        raise WireError(f"bad magic/version: {magic!r}/{version}")
    return StreamPacket(
        session_id=session.hex(),
        seq=seq,
        pts=pts,
        pcm=data[_HEADER.size :],
        sample_rate=rate,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_wire.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/wire.py tests/test_stream_vc_wire.py
git commit -m "feat(stream-vc): UDP wire framing for StreamPacket (ADR-0051)"
```

---

### Task 3: Jitter buffer (`jitter.py`)

The reorder / prebuffer / conceal / overflow / late-drop policy (ADR-0056). Prebuffer depth **is** the reorder tolerance: by the time we pop `next_seq`, we are `depth` blocks behind newest, so a merely-reordered packet is already buffered; a genuine miss at pop time is real loss → conceal + advance.

**Files:**
- Create: `vspeech/stream_vc/jitter.py`
- Test: `tests/test_stream_vc_jitter.py`

**Interfaces:**
- Consumes: `StreamPacket`.
- Produces: `class PopKind(Enum)` (`PREBUFFER`, `NORMAL`, `CONCEAL`); `@dataclass PopResult(pcm: bytes, kind: PopKind, gap: int, dropped: int)`; `class JitterBuffer(target_depth: int)` with `push(p: StreamPacket) -> bool`, `pop() -> PopResult`, `reset() -> None`, and property `depth -> int` (current buffered count).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_jitter.py
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket


def _pkt(seq, byte):
    return StreamPacket(
        session_id="00" * 16, seq=seq, pts=0.0,
        pcm=bytes([byte, byte]), sample_rate=16000,
    )


def _prime(buf, seqs):
    for s in seqs:
        buf.push(_pkt(s, s % 256))


def test_prebuffer_emits_silence_until_primed():
    buf = JitterBuffer(target_depth=2)
    buf.push(_pkt(0, 9))
    r = buf.pop()  # only 1 buffered, need depth+1=3
    assert r.kind is PopKind.PREBUFFER
    assert set(r.pcm) == {0}  # silence, sized to block


def test_in_order_playout_after_prime():
    buf = JitterBuffer(target_depth=1)
    _prime(buf, [0, 1])  # depth+1=2 -> primes at seq 0
    assert buf.pop().pcm == bytes([0, 0])
    assert buf.pop().pcm == bytes([1, 1])


def test_reorder_within_depth_is_recovered():
    buf = JitterBuffer(target_depth=1)
    buf.push(_pkt(0, 0))
    buf.push(_pkt(2, 2))  # 1 arrives late, out of order
    buf.push(_pkt(1, 1))
    assert buf.pop().pcm == bytes([0, 0])
    assert buf.pop().pcm == bytes([1, 1])  # recovered in order
    assert buf.pop().pcm == bytes([2, 2])


def test_missing_packet_conceals_and_advances():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(0, 5))
    assert buf.pop().kind is PopKind.NORMAL  # seq 0
    # seq 1 never arrives; seq 2 present -> pop expects 1 -> conceal
    buf.push(_pkt(2, 2))
    r = buf.pop()
    assert r.kind is PopKind.CONCEAL
    assert r.gap == 1
    assert buf.pop().pcm == bytes([2, 2])  # resumes at 2


def test_late_packet_after_playout_is_dropped():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(0, 0))
    buf.pop()  # plays 0, next_seq=1
    assert buf.push(_pkt(0, 0)) is False  # seq 0 already gone -> dropped


def test_overflow_fast_forwards_to_bound_latency():
    buf = JitterBuffer(target_depth=1)
    # far-ahead burst while next_seq stuck at 0 (0 never came)
    for s in range(1, 12):
        buf.push(_pkt(s, s % 256))
    r = buf.pop()  # newest=11, depth=1 -> ff next_seq near 10, dropping the middle
    assert r.dropped > 0
    assert buf.pop().kind is PopKind.NORMAL


def test_reset_clears_state():
    buf = JitterBuffer(target_depth=0)
    buf.push(_pkt(5, 5))
    buf.pop()
    buf.reset()
    buf.push(_pkt(0, 0))
    assert buf.pop().pcm == bytes([0, 0])  # next_seq re-primes from scratch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_jitter.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `jitter.py`**

```python
"""consumer 側ジッタバッファ(ADR-0056)。

並べ替え・prebuffer・穴埋め・遅延上限を transport 非依存に持つ純ロジック。
prebuffer 深さがそのまま並べ替え耐性になる: pop 時点で newest から depth ブロック
遅れて読むので、単に順序が入れ替わっただけの packet は既にバッファにある。pop 時に
本当に無い = 実 loss として直前ブロックをフェードした concealment を出して advance する。
numpy は concealment のときだけメソッド内 import する(module を import 軽量に保つ)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vspeech.stream_vc.packet import StreamPacket

# overflow: newest_seq - next_seq がこの余裕を超えたら near-live へ fast-forward。
_OVERFLOW_SLACK = 4


class PopKind(Enum):
    PREBUFFER = 0  # まだ primed でない(起動時)。silence を出す。
    NORMAL = 1  # 期待 seq が在って再生した。
    CONCEAL = 2  # 期待 seq が欠落。直前ブロックをフェードして穴埋めした。


@dataclass
class PopResult:
    pcm: bytes
    kind: PopKind
    gap: int  # このpopで欠落と確定した packet 数(telemetry 用)
    dropped: int  # overflow fast-forward で捨てた packet 数


class JitterBuffer:
    def __init__(self, target_depth: int) -> None:
        self.target_depth = target_depth
        self._buf: dict[int, bytes] = {}
        self._next_seq: int | None = None  # None = not primed
        self._last_good: bytes | None = None
        self._concealed_since_good = 0
        self._block_bytes = 0  # 最初の push で確定

    @property
    def depth(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
        self._next_seq = None
        self._last_good = None
        self._concealed_since_good = 0

    def push(self, packet: StreamPacket) -> bool:
        if not self._block_bytes:
            self._block_bytes = len(packet.pcm)
        if self._next_seq is not None and packet.seq < self._next_seq:
            return False  # 再生済みより古い = late。捨てる(呼び出しが記録)。
        if packet.seq in self._buf:
            return False  # 重複
        self._buf[packet.seq] = packet.pcm
        return True

    def _silence(self) -> bytes:
        return bytes(self._block_bytes)

    def _conceal(self) -> bytes:
        # 初回の欠落は直前 good ブロックを無音へフェード、以降連続の欠落は無音。
        if self._last_good is None or self._concealed_since_good > 0:
            self._concealed_since_good += 1
            return self._silence()
        self._concealed_since_good += 1
        import numpy as np

        a = np.frombuffer(self._last_good, dtype=np.int16).astype(np.float32)
        fade = np.linspace(1.0, 0.0, a.shape[0], dtype=np.float32)
        return np.rint(a * fade).astype(np.int16).tobytes()

    def pop(self) -> PopResult:
        if self._next_seq is None:
            if len(self._buf) > self.target_depth:
                self._next_seq = min(self._buf)
            else:
                return PopResult(self._silence(), PopKind.PREBUFFER, gap=0, dropped=0)
        dropped = 0
        # 遅延上限: backlog が余裕を超えたら near-live へ飛ばす(捨てた分を記録)。
        if self._buf:
            newest = max(self._buf)
            if newest - self._next_seq > self.target_depth + _OVERFLOW_SLACK:
                target = newest - self.target_depth
                for s in list(self._buf):
                    if s < target:
                        del self._buf[s]
                        dropped += 1
                self._next_seq = target
        pcm = self._buf.pop(self._next_seq, None)
        if pcm is not None:
            self._next_seq += 1
            self._last_good = pcm
            self._concealed_since_good = 0
            return PopResult(pcm, PopKind.NORMAL, gap=0, dropped=dropped)
        # 欠落 = 実 loss(reorder は depth が吸収済み)。conceal して advance。
        self._next_seq += 1
        return PopResult(self._conceal(), PopKind.CONCEAL, gap=1, dropped=dropped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_jitter.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/jitter.py tests/test_stream_vc_jitter.py
git commit -m "feat(stream-vc): transport-agnostic jitter buffer (ADR-0056)"
```

---

### Task 4: UDP transports (`udp.py`)

**Files:**
- Create: `vspeech/stream_vc/udp.py`
- Test: `tests/test_stream_vc_udp.py`

**Interfaces:**
- Consumes: `Transport` ABC (`vspeech/stream_vc/transport.py`), `encode_packet`/`decode_packet` (Task 2), `StreamPacket`.
- Produces:
  - `async def create_udp_producer_transport(peer_host: str, peer_port: int) -> UdpProducerTransport`
  - `async def create_udp_consumer_transport(bind_host: str, bind_port: int, max_queued: int) -> UdpConsumerTransport`
  - `UdpProducerTransport.send(p) -> bool` (False on socket error), `.close()`.
  - `UdpConsumerTransport.recv() -> StreamPacket`, `.poll() -> list[StreamPacket]` (non-blocking drain of already-arrived datagrams), `.close()`.
- Also adds `poll(self) -> list[StreamPacket]` returning `[]` to the `Transport` ABC in `transport.py` (default; consumer overrides).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_udp.py
import pytest

from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.udp import create_udp_consumer_transport
from vspeech.stream_vc.udp import create_udp_producer_transport


def _pkt(seq):
    return StreamPacket(
        session_id="ab" * 16, seq=seq, pts=float(seq),
        pcm=bytes([seq % 256]) * 320, sample_rate=16000,
    )


async def test_producer_to_consumer_loopback():
    consumer = await create_udp_consumer_transport("127.0.0.1", 0, max_queued=8)
    port = consumer.local_port
    producer = await create_udp_producer_transport("127.0.0.1", port)
    try:
        assert await producer.send(_pkt(0)) is True
        got = await consumer.recv()
        assert got == _pkt(0)
    finally:
        producer.close()
        consumer.close()


async def test_consumer_poll_drains_all_arrived():
    consumer = await create_udp_consumer_transport("127.0.0.1", 0, max_queued=8)
    producer = await create_udp_producer_transport("127.0.0.1", consumer.local_port)
    try:
        for s in range(3):
            await producer.send(_pkt(s))
        first = await consumer.recv()
        rest = consumer.poll()
        seqs = [first.seq, *[p.seq for p in rest]]
        assert sorted(seqs) == [0, 1, 2]
    finally:
        producer.close()
        consumer.close()
```

> Note: `asyncio_mode = "auto"` is set, so these `async def test_...` run without a marker. The loopback send→recv has a tiny scheduling delay; `recv()` awaits so it is deterministic. If `poll()` races the last datagram, `recv()` again — but on loopback all three are delivered before the first `recv()` returns in practice; keep the test as written and, if flaky in CI, add one `await asyncio.sleep(0)` before `poll()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_udp.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `udp.py` and extend the ABC**

First add `poll` to `vspeech/stream_vc/transport.py` `Transport` ABC (after `drain_to_latest`):

```python
    def poll(self) -> list[StreamPacket]:
        """到着済みで待機中の packet を全て非ブロッキングに取り出して返す。

        consumer が recv 後に呼び、socket キューにある残りを一括で jitter buffer へ
        push するためのもの(並べ替えを buffer に見せる)。キューを覗けない transport は
        既定で何もしない。
        """
        return []
```

Then create `vspeech/stream_vc/udp.py`:

```python
"""ストリーミング VC の生 UDP transport(ADR-0051 T3)。

producer は 1 ブロック 1 データグラムを送るだけ、consumer は受信を Queue に積み
recv/poll で出す。並べ替え・穴埋め・遅延上限は持たない(それは JitterBuffer =
ADR-0056)。asyncio の datagram endpoint を使うので追加依存は無い。
"""

from __future__ import annotations

from asyncio import DatagramProtocol
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull
from asyncio import get_running_loop
from typing import Any

from vspeech.logger import logger
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport
from vspeech.stream_vc.wire import WireError
from vspeech.stream_vc.wire import decode_packet
from vspeech.stream_vc.wire import encode_packet


class UdpProducerTransport(Transport):
    def __init__(self, transport: Any) -> None:
        self._transport = transport

    async def send(self, packet: StreamPacket) -> bool:
        try:
            self._transport.sendto(encode_packet(packet))
            return True
        except OSError as e:  # socket buffer full / route gone: drop, don't crash
            logger.warning("stream_vc udp send failed; dropping packet: %r", e)
            return False

    async def recv(self) -> StreamPacket:
        raise NotImplementedError("producer transport does not receive")

    def close(self) -> None:
        self._transport.close()


class _RecvProtocol:
    """datagram を decode して Queue へ。満杯なら最古を捨てる(遅延の張り付き防止)。"""

    def __init__(self, queue: Queue[StreamPacket]) -> None:
        self._queue = queue

    def connection_made(self, transport: Any) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: Any) -> None:
        try:
            packet = decode_packet(data)
        except WireError as e:
            logger.warning("stream_vc udp: dropping malformed datagram: %r", e)
            return
        try:
            self._queue.put_nowait(packet)
        except QueueFull:
            try:
                self._queue.get_nowait()
            except QueueEmpty:
                pass
            self._queue.put_nowait(packet)

    def error_received(self, exc: Exception) -> None:
        logger.warning("stream_vc udp recv error: %r", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            logger.warning("stream_vc udp connection lost: %r", exc)


class UdpConsumerTransport(Transport):
    def __init__(self, transport: Any, queue: Queue[StreamPacket]) -> None:
        self._transport = transport
        self._queue = queue

    @property
    def local_port(self) -> int:
        return int(self._transport.get_extra_info("sockname")[1])

    async def send(self, packet: StreamPacket) -> bool:
        raise NotImplementedError("consumer transport does not send")

    async def recv(self) -> StreamPacket:
        return await self._queue.get()

    def poll(self) -> list[StreamPacket]:
        out: list[StreamPacket] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except QueueEmpty:
                return out

    def close(self) -> None:
        self._transport.close()


async def create_udp_producer_transport(
    peer_host: str, peer_port: int
) -> UdpProducerTransport:
    loop = get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        DatagramProtocol,  # producer never receives; base no-op protocol
        remote_addr=(peer_host, peer_port),
    )
    return UdpProducerTransport(transport)


async def create_udp_consumer_transport(
    bind_host: str, bind_port: int, max_queued: int
) -> UdpConsumerTransport:
    loop = get_running_loop()
    queue: Queue[StreamPacket] = Queue(maxsize=max_queued)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RecvProtocol(queue), local_addr=(bind_host, bind_port)
    )
    return UdpConsumerTransport(transport, queue)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_udp.py tests/test_stream_vc_transport.py -v`
Expected: PASS (loopback + existing transport tests still green).

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/udp.py vspeech/stream_vc/transport.py tests/test_stream_vc_udp.py
git commit -m "feat(stream-vc): raw UDP producer/consumer transports (ADR-0051)"
```

---

### Task 5: Consumer playback loop (`consumer.py`)

Recv → push into jitter buffer → drain the rest with `poll()` → pop one block → write to the output. Reuses `playback.py`'s output-open + throttled loggers. Handles session-id change (producer restart → buffer reset) and output-device faults via the existing retry. Records skew-immune telemetry only (inter-arrival jitter, buffer depth, conceal, gap, drop). The per-cycle core is a pure helper so it is CPU-testable with a fake transport.

**Files:**
- Create: `vspeech/stream_vc/consumer.py`
- Reuse: `vspeech/stream_vc/playback.py` (`open_stream_vc_output_stream`, `detect_gap`, the `should_log_*` helpers)
- Test: `tests/test_stream_vc_consumer.py`

**Interfaces:**
- Consumes: `Transport` (ABC, with `recv`/`poll`), `JitterBuffer`/`PopKind`/`PopResult` (Task 3), `StreamVcConfig`.
- Produces:
  - `def consume_into_buffer(transport, buffer, first) -> None` — push `first` + everything from `poll()`.
  - `async def network_playback_loop(config: StreamVcConfig, transport: Transport) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_consumer.py
from vspeech.stream_vc.consumer import consume_into_buffer
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.packet import StreamPacket
from vspeech.stream_vc.transport import Transport


class _FakeTransport(Transport):
    def __init__(self, queued):
        self._queued = list(queued)

    async def send(self, packet):  # unused
        raise NotImplementedError

    async def recv(self):
        return self._queued.pop(0)

    def poll(self):
        out, self._queued = self._queued, []
        return out


def _pkt(seq):
    return StreamPacket(
        session_id="cd" * 16, seq=seq, pts=0.0, pcm=bytes([seq % 256]) * 4,
        sample_rate=16000,
    )


async def test_consume_into_buffer_drains_recv_and_poll():
    t = _FakeTransport([_pkt(0), _pkt(1), _pkt(2)])
    buf = JitterBuffer(target_depth=0)
    first = await t.recv()
    consume_into_buffer(t, buf, first)
    assert buf.depth == 3  # first(0) + poll(1,2)
    assert buf.pop().kind is PopKind.NORMAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_consumer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `consumer.py`**

```python
"""ストリーミング VC の consumer 再生ループ(role=consumer, ADR-0055/0056)。

torch/RVC/GPU を一切 import しない(再生専任マシンは変換音声を鳴らすだけ)。
transport.recv → jitter buffer push → poll で残りも push → pop 1 ブロック → 出力 write。
遅延の計測は skew 免疫の量だけ: 到着間隔ジッタと seq gap(片道遅延は clock skew に
汚染されるので測らない, ADR-0056)。出力デバイス障害は playback.py と同じく
self-heal(次パケットで lazy 再 open)。
"""

from __future__ import annotations

from asyncio import CancelledError
from asyncio import to_thread
from time import perf_counter

import sounddevice as sd

from vspeech.config import StreamVcConfig
from vspeech.exceptions import shutdown_worker
from vspeech.exceptions import worker_startup
from vspeech.lib.telemetry import telemetry
from vspeech.logger import logger
from vspeech.stream_vc.jitter import JitterBuffer
from vspeech.stream_vc.jitter import PopKind
from vspeech.stream_vc.playback import open_stream_vc_output_stream
from vspeech.stream_vc.playback import should_log_gap
from vspeech.stream_vc.playback import should_log_underflow
from vspeech.stream_vc.retry import close_quietly
from vspeech.stream_vc.transport import Transport


def consume_into_buffer(
    transport: Transport, buffer: JitterBuffer, first
) -> None:
    """recv した first と poll した残り全部を jitter buffer へ push する。"""
    buffer.push(first)
    for packet in transport.poll():
        buffer.push(packet)


async def network_playback_loop(config: StreamVcConfig, transport: Transport) -> None:
    target_depth = round(config.jitter_buffer_ms / config.block_ms)
    buffer = JitterBuffer(target_depth=target_depth)
    logger.info("stream_vc consumer jitter buffer depth: %d block(s)", target_depth)
    stream: sd.RawOutputStream | None = None
    session: str | None = None
    prev_recv: float | None = None
    started = False
    underflow_count = 0
    gap_count = 0
    try:
        while True:
            packet = await transport.recv()
            now = perf_counter()
            if prev_recv is not None:
                telemetry.record("stream_vc_interarrival", now - prev_recv)
            prev_recv = now
            if packet.session_id != session:
                if session is not None:
                    logger.info("stream_vc consumer: producer session changed; reset")
                session = packet.session_id
                buffer.reset()
            consume_into_buffer(transport, buffer, packet)
            result = buffer.pop()
            telemetry.record("stream_vc_jitter_buffer_depth", float(buffer.depth))
            if result.kind is PopKind.CONCEAL:
                telemetry.record("stream_vc_conceal", 1.0)
            if result.gap:
                telemetry.record("stream_vc_gap", float(result.gap))
                gap_count += 1
                if should_log_gap(gap_count):
                    logger.warning(
                        "stream_vc consumer gap: %d packet(s) missing (total %d)",
                        result.gap, gap_count,
                    )
            if result.dropped:
                telemetry.record("stream_vc_playback_drop", float(result.dropped))
            try:
                if stream is None:
                    if started:
                        stream = open_stream_vc_output_stream(config, packet.sample_rate)
                        logger.info("stream vc consumer playback reopened")
                    else:
                        with worker_startup("stream_vc"):
                            stream = open_stream_vc_output_stream(
                                config, packet.sample_rate
                            )
                        started = True
                        logger.info("stream vc consumer playback started")
                underflowed = await to_thread(stream.write, result.pcm)
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    underflow_count += 1
                    if should_log_underflow(underflow_count):
                        logger.warning(
                            "stream_vc consumer output underflow (total %d)",
                            underflow_count,
                        )
            except (OSError, sd.PortAudioError) as e:
                logger.warning("stream_vc consumer output fault; reopen: %r", e)
                telemetry.record("stream_vc_playback_reopen", 1.0)
                if stream is not None:
                    close_quietly(stream)
                stream = None
    except CancelledError as e:
        raise shutdown_worker(e)
    finally:
        if stream is not None:
            close_quietly(stream)
```

> Note on gap accounting: producer-side `seq` gaps (packets the producer dropped before send) are surfaced by the `JitterBuffer` at pop time as `CONCEAL`/`gap`, which is the single authority here — do **not** also diff `packet.seq` against a previous seq on arrival, or reordered arrivals would double-count. (`detect_gap` in `playback.py` is used only by the local M2 path; the consumer does not import it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stream_vc_consumer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/consumer.py tests/test_stream_vc_consumer.py
git commit -m "feat(stream-vc): consumer playback loop over jitter buffer (ADR-0056)"
```

---

### Task 6: Subsystem role wiring (`subsystem.py`)

Build the transport + loop set per role. `local` = M2 verbatim. `producer` = capture + vc (+ UDP send transport, no playback). `consumer` = network playback only (no capture/vc/torch). Lazy imports keep the consumer path torch-free.

**Files:**
- Modify: `vspeech/stream_vc/subsystem.py`
- Test: `tests/test_stream_vc_subsystem.py` (extend), plus a forbidden-import assertion in `tests/test_forbidden_imports.py` (extend the existing one)

**Interfaces:**
- Consumes: `StreamVcRole`, `TransportType`, `create_udp_producer_transport`/`create_udp_consumer_transport` (Task 4), `network_playback_loop` (Task 5), existing `capture_loop`/`vc_loop`/`playback_loop`, `InProcessTransport`.
- Produces: `def loops_for_role(role: StreamVcRole) -> frozenset[str]` (pure, testable: which loop names run) — the branching authority.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_vc_subsystem.py  (add)
from vspeech.config import StreamVcRole
from vspeech.stream_vc.subsystem import loops_for_role


def test_loops_for_role():
    assert loops_for_role(StreamVcRole.local) == frozenset(
        {"capture", "vc", "playback"}
    )
    assert loops_for_role(StreamVcRole.producer) == frozenset({"capture", "vc"})
    assert loops_for_role(StreamVcRole.consumer) == frozenset({"playback"})
```

Add to `tests/test_forbidden_imports.py` a check that importing the consumer path pulls no torch:

```python
def test_consumer_path_is_torch_free():
    import importlib
    import sys

    for mod in ("vspeech.stream_vc.consumer", "vspeech.stream_vc.udp",
                "vspeech.stream_vc.jitter", "vspeech.stream_vc.wire"):
        importlib.import_module(mod)
    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stream_vc_subsystem.py::test_loops_for_role tests/test_forbidden_imports.py::test_consumer_path_is_torch_free -v`
Expected: FAIL (`ImportError: cannot import name 'loops_for_role'`). The torch-free test may pass already; keep it as a guard.

> Caution: if any earlier test in the process imported torch, `test_consumer_path_is_torch_free` sees a polluted `sys.modules`. Run it in isolation (as above) or `-p no:randomly`; the assertion is meaningful only for the consumer modules' own transitive imports.

- [ ] **Step 3: Implement the role wiring**

Rewrite the body of `_stream_vc_subsystem` in `vspeech/stream_vc/subsystem.py` to branch on role. Keep the existing `_iter_leaves` and the `except CancelledError / except BaseExceptionGroup` handling verbatim — only the transport construction and `tg.create_task` set change. Add the module-level imports these use: `from vspeech.config import StreamVcRole`, `from vspeech.exceptions import worker_startup` (the current module does not import `worker_startup`). `Queue`, `Event`, `Any`, `uuid4`, `shutdown_worker`, `TaskGroup` are already imported. (`TransportType` is not needed here — the factory branches on role.)

```python
from vspeech.config import StreamVcRole
from vspeech.config import TransportType


def loops_for_role(role: StreamVcRole) -> frozenset[str]:
    """role が起動するループ名の集合(純関数=分岐の唯一の権威)。"""
    if role is StreamVcRole.producer:
        return frozenset({"capture", "vc"})
    if role is StreamVcRole.consumer:
        return frozenset({"playback"})
    return frozenset({"capture", "vc", "playback"})  # local


async def _build_transport(sv_config):
    """role から transport を作る。UDP endpoint 生成は async。

    bind/接続失敗は worker_startup で fail-loud(設定不備を隠さない, ADR-0038)。
    role=producer/consumer で transport_type が udp でない設定は preflight で弾く
    (role≠local ⇒ udp 必須)。2 つ目の網 transport(TCP/bidi)が来たら、下の
    producer/consumer の中で transport_type を見て分岐する。
    """
    role = sv_config.role
    if role is StreamVcRole.local:
        from vspeech.stream_vc.transport import InProcessTransport

        return InProcessTransport(max_queued=sv_config.max_queued_blocks)
    with worker_startup("stream_vc"):
        if role is StreamVcRole.producer:
            from vspeech.stream_vc.udp import create_udp_producer_transport

            return await create_udp_producer_transport(
                sv_config.peer_host, sv_config.peer_port
            )
        from vspeech.stream_vc.udp import create_udp_consumer_transport

        return await create_udp_consumer_transport(
            sv_config.bind_host, sv_config.bind_port, sv_config.max_queued_blocks
        )
```

Then in `_stream_vc_subsystem`, after `sv_config = context.config.stream_vc`:

```python
    role = sv_config.role
    runs = loops_for_role(role)
    session_id = uuid4().hex
    transport = await _build_transport(sv_config)
    try:
        async with TaskGroup() as tg:
            if "capture" in runs or "vc" in runs:
                from vspeech.stream_vc.capture import capture_loop
                from vspeech.stream_vc.capture import ms_to_samples
                from vspeech.stream_vc.runner import vc_loop

                hop = ms_to_samples(sv_config.block_ms)
                capture_queue: Queue[Any] = Queue(maxsize=sv_config.max_queued_blocks)
                vc_ready = Event()
                tg.create_task(
                    capture_loop(sv_config, capture_queue, hop, vc_ready),
                    name="stream_vc_capture",
                )
                tg.create_task(
                    vc_loop(context, sv_config, capture_queue, transport,
                            session_id, vc_ready),
                    name="stream_vc_runner",
                )
            if role is StreamVcRole.local:
                from vspeech.stream_vc.playback import playback_loop

                tg.create_task(
                    playback_loop(sv_config, transport), name="stream_vc_playback"
                )
            elif role is StreamVcRole.consumer:
                from vspeech.stream_vc.consumer import network_playback_loop

                tg.create_task(
                    network_playback_loop(sv_config, transport),
                    name="stream_vc_playback",
                )
    except CancelledError as e:
        raise shutdown_worker(e)
    except BaseExceptionGroup as eg:
        # ... unchanged fail-loud logging + re-raise (keep verbatim) ...
```

Keep the imports the consumer path must NOT pull (`runner`, `capture`, `lib.stream_vc`) inside the `if "capture" in runs or "vc" in runs:` block so the consumer role never imports them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stream_vc_subsystem.py tests/test_forbidden_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vspeech/stream_vc/subsystem.py tests/test_stream_vc_subsystem.py tests/test_forbidden_imports.py
git commit -m "feat(stream-vc): role-based subsystem wiring for producer/consumer/local (ADR-0055)"
```

---

### Task 7: Role-aware preflight (`preflight.py`)

`local` keeps today's checks (RVC assets + both devices). `producer` needs RVC assets + input device + `peer_host`/`peer_port` (no output device). `consumer` needs output device + `bind_port` only — **no RVC assets, no input device** (it has no GPU).

**Files:**
- Modify: `vspeech/preflight.py` (`_check_stream_vc`, lines ~304-348)
- Test: `tests/test_preflight.py` (extend)

**Interfaces:**
- Consumes: `StreamVcRole`, `ConfigProblem`, existing `_check_rvc_assets` / `_check_vad_gate` / `resolve_stream_vc_input_device` / `resolve_stream_vc_output_device`.
- Produces: role-conditional `list[ConfigProblem]` from `_check_stream_vc`. `ConfigProblem.field` values: `stream_vc.peer_port`, `stream_vc.bind_port` for the new address checks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preflight.py  (add)
from vspeech.config import Config
from vspeech.preflight import collect_problems


def _fields(problems):
    return {p.field for p in problems}


def test_consumer_requires_bind_not_rvc_or_input():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "consumer", "transport_type": "udp"}}
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.bind_port" in fields
    assert not any(f.startswith("stream_vc.rvc") for f in fields)
    assert "stream_vc.input_device_index" not in fields


def test_producer_requires_peer_and_input_not_output():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "producer", "transport_type": "udp"}}
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.peer_port" in fields
    assert "stream_vc.output_device_index" not in fields


def test_non_local_role_requires_udp_transport():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "consumer"}}  # transport defaults in_process
    )
    assert "stream_vc.transport_type" in _fields(collect_problems(cfg))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preflight.py -k "consumer_requires or producer_requires" -v`
Expected: FAIL (consumer currently checks RVC assets + input device unconditionally).

- [ ] **Step 3: Implement role-aware `_check_stream_vc`**

Replace the body of `_check_stream_vc` so device/asset/address checks are gated by role:

```python
def _check_stream_vc(config: Config) -> list[ConfigProblem]:
    if not config.stream_vc.enable:
        return []
    from vspeech.config import StreamVcRole
    from vspeech.config import TransportType
    from vspeech.exceptions import DeviceNotFoundError
    from vspeech.lib.audio import resolve_stream_vc_input_device
    from vspeech.lib.audio import resolve_stream_vc_output_device
    from vspeech.stream_vc.capture import ms_to_samples

    w = "stream_vc"
    sv = config.stream_vc
    role = sv.role
    does_vc = role in (StreamVcRole.local, StreamVcRole.producer)
    does_play = role in (StreamVcRole.local, StreamVcRole.consumer)
    problems: list[ConfigProblem] = []

    if does_vc:
        problems += _check_rvc_assets(sv.rvc, w, "stream_vc.rvc")
        cf = ms_to_samples(sv.crossfade_ms)
        blk = ms_to_samples(sv.block_ms)
        ctx = ms_to_samples(sv.context_ms)
        if cf >= blk:
            problems.append(ConfigProblem(
                w, f"crossfade_ms ({sv.crossfade_ms}) は block_ms ({sv.block_ms}) 未満が必須です",
                field="stream_vc.crossfade_ms"))
        if cf > ctx:
            problems.append(ConfigProblem(
                w, f"crossfade_ms ({sv.crossfade_ms}) は context_ms ({sv.context_ms}) 以下が必須です",
                field="stream_vc.crossfade_ms"))
        problems.extend(_check_vad_gate(sv, w))
        try:
            resolve_stream_vc_input_device(sv)
        except DeviceNotFoundError as e:
            problems.append(ConfigProblem(w, str(e), field="stream_vc.input_device_index"))
    if does_play:
        try:
            resolve_stream_vc_output_device(sv)
        except DeviceNotFoundError as e:
            problems.append(ConfigProblem(w, str(e), field="stream_vc.output_device_index"))

    # role≠local は網 transport が要る。in_process のままだと vc の送信を誰も受けず
    # 全ブロックが黙って drop される(silent misconfig)ので fail-loud で弾く。
    if role is not StreamVcRole.local and sv.transport_type is not TransportType.udp:
        problems.append(ConfigProblem(
            w, "role=producer/consumer は transport_type=udp が必須です",
            field="stream_vc.transport_type"))
    # UDP なら role ごとにアドレスが要る。in_process(local)は不要。
    if sv.transport_type is TransportType.udp:
        if role is StreamVcRole.producer and not (sv.peer_host and sv.peer_port):
            problems.append(ConfigProblem(
                w, "role=producer は peer_host/peer_port(送信先)が必須です",
                field="stream_vc.peer_port"))
        if role is StreamVcRole.consumer and not sv.bind_port:
            problems.append(ConfigProblem(
                w, "role=consumer は bind_port(待受ポート)が必須です",
                field="stream_vc.bind_port"))
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: PASS (new role tests + existing preflight tests).

- [ ] **Step 5: Commit**

```bash
git add vspeech/preflight.py tests/test_preflight.py
git commit -m "feat(stream-vc): role-aware preflight; consumer needs no GPU/input (ADR-0045/0055)"
```

---

### Task 8: Docs — config example + ADR promotions

**Files:**
- Modify: `config.toml.example` (the `[stream_vc]` section)
- Modify: `docs/adr/0051-stream-transport-swappable-tiered.md`, `docs/adr/0055-*.md`, `docs/adr/0056-*.md`, `docs/adr/README.md`

- [ ] **Step 1: Document the new keys in `config.toml.example`**

In the existing `[stream_vc]` block, add (keep the file's comment style; use `<NAS_HOST>`-style placeholders, not real LAN IPs, per the gitleaks gate):

```toml
# --- M3 2-machine split (ADR-0055/0056). role=local(既定)は単一マシン=M2 と同一 ---
# role = "local"            # local | producer | consumer
# transport_type = "udp"    # local は in_process、split は udp
# producer(録音+VC 機)側:
# peer_host = "<PLAYBACK_HOST>"   # consumer の bind と一致させる
# peer_port = 9999
# consumer(再生専任機)側:
# bind_host = "0.0.0.0"
# bind_port = 9999
# jitter_buffer_ms = 0.0    # 深さ=付加遅延。まず 0 で実測し最小に詰める(ADR-0056)
```

- [ ] **Step 2: Promote the ADRs (the one allowed after-the-fact 1-line update)**

- `0051`: `Status: Proposed` → `Status: Accepted`, and append one sentence to Consequences that a second transport (raw UDP, `vspeech/stream_vc/udp.py`) now realizes the swap layer, sent one datagram per block with IP fragmentation for >MTU blocks; reorder/loss/latency-bound moved to the `JitterBuffer` (ADR-0056), not the transport.
- `0055`: `Status: Proposed` → `Status: Accepted`.
- `0056`: `Status: Proposed` → `Status: Accepted`.
- `README.md`: flip the three rows' Status column to `Accepted`.

- [ ] **Step 3: Verify no secrets/PII tripwire and docs render**

Run: `uv run pytest -q` (full suite) and, if `pre-commit` is installed, `pre-commit run gitleaks --all-files`.
Expected: suite green; gitleaks clean (placeholders only).

- [ ] **Step 4: Commit**

```bash
git add config.toml.example docs/adr/0051-stream-transport-swappable-tiered.md docs/adr/0055-stream-vc-producer-consumer-role-split.md docs/adr/0056-stream-vc-consumer-jitter-buffer.md docs/adr/README.md
git commit -m "docs(stream-vc): document M3 split config; promote ADR-0051/0055/0056 to Accepted"
```

---

### Task 9: Whole-branch gate + on-machine measurement (validation)

This task has **no new production code** — it is the verification the milestone exists for. It ends with tuned defaults and a go/no-go, mirroring M1/M2.

- [ ] **Step 1: Full local gate**

Run: `uv run poe check` and `uv run pytest -q`.
Expected: ty/ruff clean; suite green; `poe check` exit 1 only on the pre-existing accepted findings (torch CVE + subtitle_tk/tts vr2_config deadcode). Any new finding is a fix, not an accept.

- [ ] **Step 2: Entry-point boot smoke (both roles), on the dev box**

Run (no GPU needed for `--help`, and boot for the consumer role is torch-free):
```bash
uv run python -m vspeech --help
```
Then, with a minimal `role="consumer"` toml (udp + bind_port + a valid output device), confirm `python -m vspeech --config <consumer.toml>` boots to "stream vc consumer playback started" without importing torch (it will block on recv — Ctrl-C to stop). This catches the class of entry-point-only bugs that unit tests miss (a lesson from M1/M2: run the entry point, not just tests).

- [ ] **Step 3: Two-machine run + skew-immune measurement**

On the GPU host (`.149`): `role="producer"`, `transport_type="udp"`, `peer_host=<playback-host>`, `peer_port=9999`, `jitter_buffer_ms=0`. On the playback host (`.150`): `role="consumer"`, `bind_host="0.0.0.0"`, `bind_port=9999`. Also start the existing 発話系 pipeline on `.149` from the same mic (ADR-0052 dual capture) and confirm transcription/translation/subtitle are unaffected.

Collect the consumer's telemetry JSONL and read the **skew-immune** metrics only:
- `stream_vc_interarrival` — inter-arrival jitter (std / p95). This sizes the buffer.
- `stream_vc_gap` / `stream_vc_conceal` / `stream_vc_playback_drop` — loss/reorder and how often concealment fired.
- `stream_vc_jitter_buffer_depth` — steady-state depth.

Do **not** compute a cross-machine one-way latency from `pts` (clock skew, ADR-0006/0056).

- [ ] **Step 4: Tune `jitter_buffer_ms` from the measurement**

Start at 0; raise `jitter_buffer_ms` just past the measured inter-arrival p95 jitter until `stream_vc_conceal` from reorder (not real loss) stops, then stop — depth is added latency. Record the chosen default and the numbers in the branch (a commit message or a comment at the config default), the way M2 recorded 160/500/25.

- [ ] **Step 5: Ear check + go/no-go**

Confirm on the real playback machine: continuous audio, no click at block boundaries, pitch continuous, no silent holes (gaps are concealed audibly-benignly). If clean, this backs ADR-0053's crossfade continuity claim across the wire and ADR-0056's buffer policy. Write the go/no-go into the branch memory. Then run `superpowers:finishing-a-development-branch` (Integration has been "keep as-is / unpushed" for M1/M2 — confirm with the user).

---

## Self-Review

**Spec coverage** (against `2026-07-22-...-split-machine-design.md` 受入基準):
- 変換音声のみがマシン間を渡り連続再生 → Tasks 4/5/6. ✅
- クリック無し / ピッチ連続 → reuse of `StreamingVc` (unchanged) + jitter conceal (Task 3/5) + ear check (Task 9). ✅
- 発話系無改変で並走 → no 発話系 file touched; validated Task 9. ✅
- 遅延単調増加せず落としたことが観測可能 → jitter overflow fast-forward + telemetry (Task 3/5). ✅
- 欠落・並べ替えを検出し穴埋め・観測、無音の穴を黙って作らない → jitter conceal + `stream_vc_conceal`/`stream_vc_gap` (Task 3/5). ✅
- transport を設定で切替、他ロジック不変 → `Transport` ABC + `transport_type` + role wiring (Task 1/4/6). ✅
- 起動時 fail-loud + GUI readiness 追従 → `worker_startup` transport build (Task 6) + role-aware preflight (Task 7, GUI auto-follows via ADR-0045). ✅
- RTF/遅延計測で予算判定 → skew-immune measurement (Task 9). ✅

**Placeholder scan:** no TBD/"handle errors"/"similar to Task N" — every code step shows full code. ✅

**Type consistency:** `StreamPacket` fields match `packet.py`; `PopResult(pcm, kind, gap, dropped)` used identically in Task 3 impl, Task 5 loop, and tests; `loops_for_role`/`_build_transport`/`network_playback_loop`/`consume_into_buffer`/`create_udp_*` signatures match across Tasks 4-7; `poll()` added to the ABC in Task 4 and consumed in Task 5. ✅
