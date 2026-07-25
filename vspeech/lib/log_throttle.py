"""繰り返す警告の時間ベース間引き(ADR-0062)。

実時間ループの警告(underflow / drop / gap / GPU の transient error / UDP の
プロトコルエラー)は、起きるときは毎ブロック・毎パケット起きる。無条件に出すと
警告自体がログを埋めて診断価値を失うので、エピソード単位で時間レート制限する。
"""

from collections.abc import Callable
from time import perf_counter

# 障害が継続している間の出力上限。イベント率ではなく時間で切るので、block_ms や
# transport の種類を変えても行数のレートが変わらない(ADR-0062)。
DEFAULT_MIN_INTERVAL_S = 5.0
# これだけ発生が途切れたら、次は「別のインシデント」として先頭から出し直す。
DEFAULT_QUIET_S = 10.0


class LogThrottle:
    """同一条件の繰り返し警告を、エピソード単位で時間レート制限する。

    `hit()` が発生を 1 件記録し、ログを出すなら「そのエピソードでの通算件数」を、
    出さないなら None を返す。呼び出し側は件数カウンタを持たない:

        if (n := self._underflow.hit()) is not None:
            logger.warning("... (total %d)", n)

    エピソードの終わりは次の `hit()` の中で遅延判定する — 成功パスへ「収まった」を
    知らせる呼び出しを足さないため(実時間ループのホットパスを増やさない)。代償として
    明示的な「回復しました」の行は出ず、次のエピソードの先頭行が境界になる。

    単一 event loop 前提でロックは持たない(対象の呼び出し元は UDP のプロトコル
    コールバックも含めすべてループ上で動く)。
    """

    def __init__(
        self,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        quiet_s: float = DEFAULT_QUIET_S,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._quiet_s = quiet_s
        self._clock = clock
        self._count = 0
        self._last_hit: float | None = None
        self._last_log = 0.0

    def hit(self) -> int | None:
        """発生を 1 件記録する。ログを出すなら通算件数、出さないなら None。"""
        now = self._clock()
        # 初回、または quiet_s ぶん途切れた後の再発 = 新しいエピソード。件数を戻して
        # 必ず 1 行出す(累積カウンタだと「初回」が一生に一度しか使えない)。
        if self._last_hit is None or now - self._last_hit > self._quiet_s:
            self._count = 1
            self._last_hit = now
            self._last_log = now
            return 1
        # エピソード継続中。件数は常に進め、出力だけを min_interval_s で絞る。
        # エピソードの境界は「最後にログした時刻」ではなく「最後に発生した時刻」で
        # 測る — でないと、間引かれ続けている最中に区切りが入ってしまう。
        self._count += 1
        self._last_hit = now
        if now - self._last_log >= self._min_interval_s:
            self._last_log = now
            return self._count
        return None
