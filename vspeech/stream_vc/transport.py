"""ストリーミング VC の transport 差し替え層(ADR-0051)。

現状は in-process(asyncio.Queue)実装のみ。producer/consumer はこの interface
の背後に置くので、網実装(UDP/TCP/bidi)へ VC・再生の他ロジックを変えずに
差し替えられる。送信側は満杯時に最古を捨てて遅延の単調増加を防ぐ(受入基準)。
受信側は `drain_to_latest` で到着済みバックログを最新へ畳み、一過性ストール後に
残る恒久遅延を抑える(consumer は実時間消費なので RTF<1 でも自然には減らない)。
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull

from vspeech.stream_vc.packet import StreamPacket


def drop_oldest_put[T](q: Queue[T], item: T) -> bool:
    """満杯なら最古を捨てて `item` を入れる。捨てたら False。

    capture/transport のバックプレッシャ共通処理。VC/GPU が実時間に追いつかない
    ときにキューが伸び続けるのを防ぎ、落としたことを呼び出し側が観測できるよう
    bool を返す(受入基準:遅延が単調増加せず落としたことが記録可能)。

    単一イベントループ前提: この関数は await を挟まないので put_nowait と
    get_nowait の間で他コルーチンが割り込まない。満杯確定直後の get_nowait は
    必ず成功し、直後の put_nowait も解放済みスロットへ必ず入る(防御的 try は不要)。
    """
    try:
        q.put_nowait(item)
        return True
    except QueueFull:
        q.get_nowait()  # 最古を捨てる(満杯確定直後なので必ず成功)
        q.put_nowait(item)  # 解放済みスロットへ(await 無しなので必ず成功)
        return False


class Transport(ABC):
    @abstractmethod
    async def send(self, packet: StreamPacket) -> bool:
        """packet を送る。バックプレッシャで最古を捨てたら False。"""

    @abstractmethod
    async def recv(self) -> StreamPacket:
        """次の packet を受け取る(無ければ待つ)。"""

    def drain_to_latest(self, keep: int = 1) -> list[StreamPacket]:
        """到着済みで待機中の packet を最新 `keep` 個だけ残し、それ以外を
        非ブロッキングに取り出して返す(捨てた古い packet を返す)。

        既に届いているキューだけを触り await を挟まないので同期処理でよい。
        キューを覗けない transport(まだ実装なし)は既定で何もしない。
        """
        return []


class InProcessTransport(Transport):
    """同一プロセス内の asyncio.Queue 実装(ADR-0051 tier-0)。"""

    def __init__(self, max_queued: int) -> None:
        self._q: Queue[StreamPacket] = Queue(maxsize=max_queued)
        self.dropped = 0

    async def send(self, packet: StreamPacket) -> bool:
        ok = drop_oldest_put(self._q, packet)
        if not ok:
            self.dropped += 1
        return ok

    async def recv(self) -> StreamPacket:
        return await self._q.get()

    def drain_to_latest(self, keep: int = 1) -> list[StreamPacket]:
        """待機中の packet を最新 `keep` 個残して非ブロッキングに取り出す。

        consumer(playback)は出力デバイスクロック=実時間で消費するため、
        一過性ストールで積み上がったバックログは RTF<1 でも自然には減らず恒久
        遅延として張り付く。recv 後にこれを呼び、既に届いている古い packet を捨てて
        near-live を保つ。捨てた分を返し、呼び出し側が seq/telemetry で観測できる。
        単一ループ前提: await を挟まないので他コルーチンは割り込まない。
        """
        dropped: list[StreamPacket] = []
        while self._q.qsize() > keep:
            try:
                dropped.append(self._q.get_nowait())
            except QueueEmpty:  # 単一ループでは起きえないが防御的に打ち切る
                break
        return dropped
