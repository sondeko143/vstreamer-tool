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
