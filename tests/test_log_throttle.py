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
    for _ in range(20):  # 20 x 0.6s = 12s > quiet_s だが、発生は途切れていない
        clock.advance(0.6)
        t.hit()
    clock.advance(5.0)  # min_interval を跨がせて件数を観測する
    assert t.hit() == 22  # エピソード継続 = 1 に戻らない
