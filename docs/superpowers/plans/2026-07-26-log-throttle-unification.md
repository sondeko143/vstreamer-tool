# ログ間引き共通化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `stream_vc` に 6 定義ある「初回 + N 回ごと」のログ間引きを、`vspeech/lib/log_throttle.py` の `LogThrottle` 1 クラス(時間ベース + エピソード境界)へ統一する。

**Architecture:** `LogThrottle.hit()` が発生を 1 件記録し、ログを出すなら「そのエピソードでの通算件数」を、出さないなら `None` を返す。呼び出し側は条件ごとに 1 インスタンスを持ち、件数カウンタを自前で持たない。エピソードの終わりは次の `hit()` の中で遅延判定するので、成功パスに呼び出しを足さない。

**Tech Stack:** Python 3.14 / pytest (`asyncio_mode = "auto"`) / ruff / ty / uv

**ADR:** [0062](../../adr/0062-log-throttle-time-based-episodes.md)(Proposed → Task 6 で Accepted へ昇格)
**Spec:** [2026-07-26-log-throttle-design.md](../specs/2026-07-26-log-throttle-design.md)

## Global Constraints

- Python は 3.14 のみ(`>=3.14,<3.15`)。PEP 695 の `type X = ...` / `class C[T]` が使える。
- import は 1 行 1 つ(ruff `force-single-line = true`)。`from x import y` 形式で、モジュール import にまとめない。
- テストは `asyncio_mode = "auto"` なので `async def test_...` に `@pytest.mark.asyncio` は不要。
- コード内コメント・docstring は日本語(既存 `stream_vc` に合わせる)。
- 検証コマンドは必ず `uv run --no-sync` を付ける(`--no-sync` 無しだと稼働中の vspeech が `.venv` を掴んでいて `uv` が os error 32 で落ちる)。
- 終了コードはパイプ越しに読まない。`cmd | tail` の `$?` は `tail` のもの。必要ならファイルへ落として `echo $?` を見る。
- ログ文言は変えない(全サイト既に `(total %d)` を持つ)。telemetry の記録も現状どおり毎回行い、間引きの影響を受けさせない。
- `LogThrottle` の閾値は設定ファイルへ露出しない(モジュール定数)。

---

### Task 1: `LogThrottle` の新設

**Files:**
- Create: `vspeech/lib/log_throttle.py`
- Test: `tests/test_log_throttle.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `LogThrottle(min_interval_s: float = 5.0, quiet_s: float = 10.0, clock: Callable[[], float] = perf_counter)`
  - `LogThrottle.hit() -> int | None` — 出すなら通算件数(1 以上)、出さないなら `None`
  - `DEFAULT_MIN_INTERVAL_S: float` / `DEFAULT_QUIET_S: float`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_log_throttle.py` を新規作成:

```python
from vspeech.lib.log_throttle import LogThrottle


class _FakeClock:
    """単調増加の偽クロック。テストを決定的にする。"""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _throttle(clock: _FakeClock) -> LogThrottle:
    return LogThrottle(min_interval_s=5.0, quiet_s=10.0, clock=clock)


def test_first_hit_always_logs():
    clock = _FakeClock()
    assert _throttle(clock).hit() == 1


def test_hits_within_min_interval_are_silent():
    clock = _FakeClock()
    t = _throttle(clock)
    assert t.hit() == 1
    for _ in range(4):
        clock.advance(0.1)
        assert t.hit() is None


def test_logs_again_after_min_interval_with_suppressed_total():
    """再開したログの件数は、間引かれたぶんも含む通算でなければ意味がない。"""
    clock = _FakeClock()
    t = _throttle(clock)
    assert t.hit() == 1
    for _ in range(4):
        clock.advance(0.1)
        t.hit()
    clock.advance(4.6)  # 前回ログからちょうど 5.0s
    assert t.hit() == 6


def test_quiet_period_rearms_and_resets_the_count():
    """静穏を挟んだ再発は「別のインシデント」= 先頭で必ず 1 行出る。"""
    clock = _FakeClock()
    t = _throttle(clock)
    assert t.hit() == 1
    clock.advance(3.0)
    assert t.hit() is None
    clock.advance(10.1)  # quiet_s 超 = 新しいエピソード
    assert t.hit() == 1


def test_episodes_are_measured_from_the_last_hit_not_the_last_log():
    """発生が途切れずに続いている限り、ログが出ていなくても同じエピソード。"""
    clock = _FakeClock()
    t = _throttle(clock)
    assert t.hit() == 1
    for _ in range(20):  # 20 x 0.6s = 12s > quiet_s だが、途切れてはいない
        clock.advance(0.6)
        t.hit()
    clock.advance(0.6)
    assert t.hit() == 22  # 1 に戻らない
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run --no-sync pytest tests/test_log_throttle.py -q`
Expected: FAIL(collection error: `ModuleNotFoundError: No module named 'vspeech.lib.log_throttle'`)

- [ ] **Step 3: 実装する**

`vspeech/lib/log_throttle.py` を新規作成:

```python
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
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run --no-sync pytest tests/test_log_throttle.py -q`
Expected: PASS (5 passed)

Run: `uv run --no-sync ruff format . && uv run --no-sync ruff check . && uv run --no-sync ty check`
Expected: すべてエラー無し

- [ ] **Step 5: コミット**

```bash
git add vspeech/lib/log_throttle.py tests/test_log_throttle.py
git commit -m "feat(log-throttle): add time-based, episode-scoped LogThrottle (ADR-0062)"
```

---

### Task 2: `playback.py` と `consumer.py` を移行

`consumer.py` は `playback.py` から述語を import しているので、この 2 ファイルは同時に移行する。

**Files:**
- Modify: `vspeech/stream_vc/playback.py`(定数/述語 38-67 行、カウンタ 96-98 行、呼び出し 4 箇所)
- Modify: `vspeech/stream_vc/consumer.py`(import 28-29 行、カウンタ 67-68 行、呼び出し 2 箇所)
- Test: `tests/test_stream_vc_playback.py`(述語テスト 3 本を削除)

**Interfaces:**
- Consumes: `LogThrottle` / `LogThrottle.hit()`(Task 1)
- Produces: `playback.py` から `UNDERFLOW_LOG_EVERY` / `DROP_LOG_EVERY` / `GAP_LOG_EVERY` / `should_log_underflow` / `should_log_drop` / `should_log_gap` が消える(Task 6 の grep 確認対象)

- [ ] **Step 1: 述語テストを削除する**

`tests/test_stream_vc_playback.py` の先頭 import から `UNDERFLOW_LOG_EVERY` と `should_log_underflow` の行を削除し、テスト関数 `test_underflow_logs_first` / `test_underflow_log_is_rate_limited` / `test_underflow_logs_every_nth`(23-34 行付近)を削除する。`detect_gap` のテストはそのまま残す。

- [ ] **Step 2: 削除でテストが落ちないことを確認する**

Run: `uv run --no-sync pytest tests/test_stream_vc_playback.py -q`
Expected: PASS(残った `detect_gap` 系のみ)。まだ `playback.py` 側は無傷なので緑のまま。

- [ ] **Step 3: `playback.py` の定数と述語を削除する**

38-67 行の 3 ブロック(`UNDERFLOW_LOG_EVERY` / `should_log_underflow` / `DROP_LOG_EVERY` / `should_log_drop` / `GAP_LOG_EVERY` / `should_log_gap` とそれぞれのコメント)を丸ごと削除し、import に 1 行足す:

```python
from vspeech.lib.log_throttle import LogThrottle
```

- [ ] **Step 4: カウンタを throttle に置き換える**

`playback_loop` の

```python
    underflow_count = 0
    drop_count = 0
    gap_count = 0
```

を、次に置き換える(ローカル変数 `gap` = `detect_gap` の戻り値と衝突しないよう `_throttle` を付ける):

```python
    # 条件ごとに 1 つ。出力 underflow も stale drop も seq gap も、起きるときは
    # 毎ブロック起きる(block_ms=160 なら ~6 回/秒)ので、警告自体がログを埋めない
    # よう時間で絞る。telemetry は間引きに関係なく毎回記録する(ADR-0062)。
    underflow_throttle = LogThrottle()
    drop_throttle = LogThrottle()
    gap_throttle = LogThrottle()
```

- [ ] **Step 5: 呼び出し 4 箇所を書き換える**

stale packet を畳むループの中の gap:

```python
                    gap = detect_gap(prev_seq, old.seq)
                    if gap > 0:
                        telemetry.record("stream_vc_gap", float(gap))
                        if (n := gap_throttle.hit()) is not None:
                            logger.warning(
                                "stream_vc playback gap: %d packet(s) missing "
                                "(total %d)",
                                gap,
                                n,
                            )
```

同じループの drop:

```python
                    prev_seq = old.seq
                    telemetry.record("stream_vc_playback_drop", 1.0)
                    if (n := drop_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback dropped stale packet(s) to bound "
                            "latency (total %d)",
                            n,
                        )
```

主経路の gap:

```python
                gap = detect_gap(prev_seq, packet.seq)
                if gap > 0:
                    telemetry.record("stream_vc_gap", float(gap))
                    if (n := gap_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback gap: %d packet(s) missing (total %d)",
                            gap,
                            n,
                        )
```

underflow:

```python
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    if (n := underflow_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc playback output underflow (total %d)", n
                        )
```

- [ ] **Step 6: `consumer.py` を書き換える**

import の

```python
from vspeech.stream_vc.playback import should_log_gap
from vspeech.stream_vc.playback import should_log_underflow
```

を削除し、

```python
from vspeech.lib.log_throttle import LogThrottle
```

を(1 行 1 import の並び順を保って)足す。カウンタ

```python
    underflow_count = 0
    gap_count = 0
```

を

```python
    # playback.py と同じ理由の時間ベース間引き(ADR-0062)。
    underflow_throttle = LogThrottle()
    gap_throttle = LogThrottle()
```

に置き換え、呼び出し 2 箇所を書き換える:

```python
            if result.gap:
                telemetry.record("stream_vc_gap", float(result.gap))
                if (n := gap_throttle.hit()) is not None:
                    logger.warning(
                        "stream_vc consumer gap: %d packet(s) missing (total %d)",
                        result.gap,
                        n,
                    )
```

```python
                if underflowed:
                    telemetry.record("stream_vc_playback_underflow", 1.0)
                    if (n := underflow_throttle.hit()) is not None:
                        logger.warning(
                            "stream_vc consumer output underflow (total %d)", n
                        )
```

- [ ] **Step 7: テストと静的検査を通す**

Run: `uv run --no-sync pytest tests/test_stream_vc_playback.py tests/test_stream_vc_consumer.py -q`
Expected: PASS

Run: `uv run --no-sync ruff format . && uv run --no-sync ruff check . && uv run --no-sync ty check`
Expected: すべてエラー無し

- [ ] **Step 8: コミット**

```bash
git add vspeech/stream_vc/playback.py vspeech/stream_vc/consumer.py tests/test_stream_vc_playback.py
git commit -m "refactor(stream-vc): move playback/consumer log throttling to LogThrottle"
```

---

### Task 3: `capture.py` を移行

**Files:**
- Modify: `vspeech/stream_vc/capture.py`(定数/述語 43-51 行、カウンタ 97 行、呼び出し 1 箇所)
- Test: `tests/test_stream_vc_capture.py`(述語テスト 3 本を削除、挙動テスト 1 本を修正)

**Interfaces:**
- Consumes: `LogThrottle`(Task 1)
- Produces: `capture.py` から `CAPTURE_DROP_LOG_EVERY` / `should_log_capture_drop` が消える

- [ ] **Step 1: テストを先に書き換える**

`tests/test_stream_vc_capture.py` から `CAPTURE_DROP_LOG_EVERY` と `should_log_capture_drop` の import 2 行、およびテスト関数 `test_capture_drop_logs_first` / `test_capture_drop_log_is_rate_limited` / `test_capture_drop_logs_every_nth` を削除する。

`test_capture_drop_while_running_warns_and_throttles` を次に置き換える(タイトループなので 51 件すべてが `min_interval_s` 内に収まり、警告は先頭 1 本だけになる):

```python
async def test_capture_drop_while_running_warns_once_per_episode(
    caplog, enabled_telemetry
):
    """running 中の drop は本物のバックプレッシャ。エピソード先頭の 1 本だけ出す。"""
    hop = 4
    running = Event()
    running.set()
    n = 51
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError):
            await _capture_read_loop(
                _FakeStream(n),  # ty: ignore[invalid-argument-type]
                hop,
                _full_queue(hop),
                running,
            )
    warnings = [r for r in caplog.records if "capture queue full" in r.getMessage()]
    assert len(warnings) == 1  # タイトループ = すべて min_interval_s 内
    assert "(total 1)" in warnings[0].getMessage()
    summary = enabled_telemetry.summary()
    assert summary["stream_vc_capture_drop"]["count"] == n  # telemetry は毎回
    assert "stream_vc_capture_drop_paused" not in summary
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run --no-sync pytest tests/test_stream_vc_capture.py -q`
Expected: FAIL(`assert 2 == 1` — まだ回数ベースなので警告が 2 本出る)

- [ ] **Step 3: `capture.py` を書き換える**

`CAPTURE_DROP_LOG_EVERY` と `should_log_capture_drop`(コメント含む 43-51 行)を削除し、import に

```python
from vspeech.lib.log_throttle import LogThrottle
```

を足す。`_capture_read_loop` の `drop_count = 0` を

```python
    # running 中の drop = 本物のバックプレッシャ。時間で絞る(ADR-0062)。
    drop_throttle = LogThrottle()
```

に置き換え、呼び出しを書き換える:

```python
            telemetry.record("stream_vc_capture_drop", 1.0)
            if (n := drop_throttle.hit()) is not None:
                logger.warning(
                    "stream_vc capture queue full; dropped oldest block (total %d)",
                    n,
                )
```

pause 分岐(`if not running.is_set():` の中の `stream_vc_capture_drop_paused`)は**触らない**。

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run --no-sync pytest tests/test_stream_vc_capture.py -q`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add vspeech/stream_vc/capture.py tests/test_stream_vc_capture.py
git commit -m "refactor(stream-vc): move capture drop throttling to LogThrottle"
```

---

### Task 4: `runner.py` を移行

**Files:**
- Modify: `vspeech/stream_vc/runner.py`(定数/述語 47-56 行、カウンタ 282 行、呼び出し 355-366 行付近)

**Interfaces:**
- Consumes: `LogThrottle`(Task 1)
- Produces: `runner.py` から `VC_ERROR_LOG_EVERY` / `should_log_vc_error` が消える

- [ ] **Step 1: 定数と述語を削除する**

`VC_ERROR_LOG_EVERY` と `should_log_vc_error`(直前のコメント 47-51 行含む)を削除し、import に

```python
from vspeech.lib.log_throttle import LogThrottle
```

を足す。

- [ ] **Step 2: カウンタと呼び出しを書き換える**

`vc_loop` の `vc_error_count = 0` を

```python
    # transient な process_block drop の警告を時間で絞る。連続失敗の tear-down 判定
    # (consecutive_errors / _MAX_CONSECUTIVE_VC_ERRORS)とは別物なので混ぜない。
    vc_error_throttle = LogThrottle()
```

に置き換える(`consecutive_errors = 0` は別目的なのでそのまま残す)。`except RuntimeError as e:` の中を書き換える:

```python
            except RuntimeError as e:
                consecutive_errors += 1
                telemetry.record("stream_vc_process_error", 1.0)
                if (n := vc_error_throttle.hit()) is not None:
                    logger.warning(
                        "stream_vc process_block failed; dropping block (total %d): %r",
                        n,
                        e,
                    )
```

`_MAX_CONSECUTIVE_VC_ERRORS` の判定と `consecutive_errors = 0` のリセット行は変更しない。

- [ ] **Step 3: 古いコメントを直す**

`consecutive_errors = 0` のリセット直前にある「警告の間引きは通算カウンタ `vc_error_count` なので reset しない」というコメントを、throttle が通算を持つ旨に書き換える:

```python
            # 連続失敗カウンタだけ回復でリセットする(tear-down 判定用)。警告の間引きは
            # LogThrottle が自前でエピソードを見るので、ここでは触らない。
            consecutive_errors = 0
```

- [ ] **Step 4: テストと静的検査を通す**

Run: `uv run --no-sync pytest tests/test_stream_vc_runner.py tests/test_stream_vc_gate.py -q`
Expected: PASS

Run: `uv run --no-sync ruff format . && uv run --no-sync ruff check . && uv run --no-sync ty check`
Expected: すべてエラー無し

- [ ] **Step 5: コミット**

```bash
git add vspeech/stream_vc/runner.py
git commit -m "refactor(stream-vc): move vc_loop error throttling to LogThrottle"
```

---

### Task 5: `udp.py` を移行

**Files:**
- Modify: `vspeech/stream_vc/udp.py`(定数/述語 25-32 行、`_SendProtocol` 45-52 行、`_RecvProtocol` 84-115 行付近)
- Test: `tests/test_stream_vc_udp.py`(`error_count` を参照する 2 本を書き換え)

**Interfaces:**
- Consumes: `LogThrottle`(Task 1)
- Produces: `udp.py` から `_LOG_EVERY` / `_should_log` / `_SendProtocol.error_count` / `_RecvProtocol._malformed_count` / `_RecvProtocol._error_count` が消える

- [ ] **Step 1: テストを先に書き換える**

`tests/test_stream_vc_udp.py` の `test_send_protocol_error_received_counts_and_logs` と `test_send_protocol_error_logging_is_throttled` を、次の 2 本に置き換える(件数の観測は telemetry 側へ移る):

```python
def test_send_protocol_error_received_records_telemetry():
    from unittest.mock import patch

    from vspeech.stream_vc.udp import _SendProtocol

    proto = _SendProtocol()
    with patch("vspeech.stream_vc.udp.telemetry") as mock_telemetry:
        proto.error_received(OSError("route gone"))
        proto.error_received(OSError("again"))
    assert mock_telemetry.record.call_count == 2


def test_send_protocol_error_logging_is_throttled():
    from unittest.mock import patch

    from vspeech.stream_vc.udp import _SendProtocol

    proto = _SendProtocol()
    with patch("vspeech.stream_vc.udp.logger") as mock_logger:
        with patch("vspeech.stream_vc.udp.telemetry") as mock_telemetry:
            for _ in range(120):
                proto.error_received(OSError("peer down"))
    # telemetry は毎回。ログはエピソード先頭の 1 本だけ(タイトループ = 同一エピソード)。
    assert mock_telemetry.record.call_count == 120
    assert mock_logger.warning.call_count == 1
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run --no-sync pytest tests/test_stream_vc_udp.py -q`
Expected: FAIL(`assert 3 == 1` — まだ回数ベースなので 120 件で 3 本出る)

- [ ] **Step 3: `udp.py` を書き換える**

`_LOG_EVERY` と `_should_log`(25-32 行)を削除し、import に

```python
from vspeech.lib.log_throttle import LogThrottle
```

を足す。`_SendProtocol`:

```python
    def __init__(self) -> None:
        # UDP のプロトコルコールバックはパケットレートで発火しうる(peer down で
        # datagram ごとに ICMP)。ログは時間で絞り、telemetry は毎回記録する。
        self._error_throttle = LogThrottle()

    def error_received(self, exc: Exception) -> None:
        telemetry.record("stream_vc_send_error", 1.0)
        if (n := self._error_throttle.hit()) is not None:
            logger.warning("stream_vc udp send error (async, total %d): %r", n, exc)
```

`_RecvProtocol`:

```python
        self._malformed_throttle = LogThrottle()
        self._error_throttle = LogThrottle()
```

```python
        except WireError as e:
            telemetry.record("stream_vc_malformed_drop", 1.0)
            if (n := self._malformed_throttle.hit()) is not None:
                logger.warning(
                    "stream_vc udp: dropping malformed datagram (total %d): %r",
                    n,
                    e,
                )
            return
```

```python
    def error_received(self, exc: Exception) -> None:
        telemetry.record("stream_vc_recv_error", 1.0)
        if (n := self._error_throttle.hit()) is not None:
            logger.warning("stream_vc udp recv error (total %d): %r", n, exc)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run --no-sync pytest tests/test_stream_vc_udp.py -q`
Expected: PASS

Run: `uv run --no-sync ruff format . && uv run --no-sync ruff check . && uv run --no-sync ty check`
Expected: すべてエラー無し

- [ ] **Step 5: コミット**

```bash
git add vspeech/stream_vc/udp.py tests/test_stream_vc_udp.py
git commit -m "refactor(stream-vc): move udp protocol log throttling to LogThrottle"
```

---

### Task 6: 全体検証と ADR の昇格

**Files:**
- Modify: `docs/adr/0062-log-throttle-time-based-episodes.md`(Status)
- Modify: `docs/adr/README.md`(索引の Status 欄)

**Interfaces:**
- Consumes: Task 1-5 の全成果
- Produces: なし(最終ゲート)

- [ ] **Step 1: 旧 idiom が残っていないことを確認する**

Run: `grep -rn "LOG_EVERY\|should_log\|_should_log" vspeech/ tests/`
Expected: 出力ゼロ(warn-once 系の `warned` フラグは対象外なのでヒットしない)

- [ ] **Step 2: 全テストを流す(終了コードで確認する)**

```bash
# 終了コードをパイプ越しに読まないこと(`| tail` の $? は tail のもの)。
uv run --no-sync pytest -q > pytest.out 2>&1; echo "PYTEST_EXIT=$?"; tail -3 pytest.out; rm pytest.out
```

Expected: `PYTEST_EXIT=0`。件数は 699 + `test_log_throttle.py` の 5 − 削除した述語テスト 6 = **698 passed**(skip 12 はそのまま)

- [ ] **Step 3: 静的検査を終了コードで確認する**

```bash
uv run --no-sync ruff format --check . ; echo "FMT_EXIT=$?"
uv run --no-sync ruff check . ; echo "LINT_EXIT=$?"
uv run --no-sync ty check ; echo "TY_EXIT=$?"
```

Expected: すべて `_EXIT=0`

- [ ] **Step 4: ADR を Accepted へ昇格する**

`docs/adr/0062-log-throttle-time-based-episodes.md` の `- Status: Proposed` を `- Status: Accepted` に変え、`docs/adr/README.md` の 0062 行の Status 欄も `Proposed` → `Accepted` にする。本文は書き換えない(ADR は不変)。

- [ ] **Step 5: コミット**

```bash
git add docs/adr/0062-log-throttle-time-based-episodes.md docs/adr/README.md
git commit -m "docs(adr): promote ADR-0062 to Accepted (time-based log throttling)"
```

---

## Self-Review

**1. Spec coverage**

| 受入基準 | Task |
|---|---|
| 継続中の警告が時間で上限を持ち、ブロック長で変わらない | 1(実装)、3・5(挙動テスト) |
| 静穏を挟んだ再発は先頭で必ず 1 行出る | 1(`test_quiet_period_rearms_and_resets_the_count`) |
| 警告に通算件数が含まれる | 1(`test_logs_again_after_min_interval_with_suppressed_total`)、3(`"(total 1)"` を assert) |
| 間引き判定が 1 箇所のみ | 6 Step 1 の grep |
| telemetry は毎回記録 | 3(`count == n`)、5(`record.call_count == 120`) |
| pause 中の capture drop は警告されない | 3(既存テストを温存) |
| 既存テストが緑 | 6 Step 2 |

**2. Placeholder scan** — TBD / TODO / 「適切に」の類は無い。全ステップに実コードがある。

**3. Type consistency** — `hit() -> int | None` は全 10 サイトで `if (n := x.hit()) is not None:` として同一に使う。`LogThrottle` のコンストラクタ引数名(`min_interval_s` / `quiet_s` / `clock`)は Task 1 のテストと実装で一致。`playback.py` はローカル変数 `gap`(int)との衝突を避けて `gap_throttle` 命名にしてある。
