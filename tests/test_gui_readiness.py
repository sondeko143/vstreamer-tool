import pytest

from gui.readiness import WORKER_NAMES
from gui.readiness import enabled_workers
from gui.readiness import evaluate
from gui.readiness import flow_of
from vspeech.config import Config
from vspeech.config import PlaybackConfig
from vspeech.config import RecordingConfig
from vspeech.config import VcConfig


def test_nothing_enabled_is_ready():
    readiness = evaluate(Config())
    assert readiness.ok
    assert readiness.problem_count == 0
    assert readiness.workers == []


def test_enabled_workers_follows_enable_flags():
    config = Config(
        recording=RecordingConfig(enable=True), playback=PlaybackConfig(enable=True)
    )
    assert enabled_workers(config) == ["recording", "playback"]


def test_vc_without_assets_is_not_ready_and_groups_by_worker():
    readiness = evaluate(Config(vc=VcConfig(enable=True)))
    assert not readiness.ok
    assert [w.worker for w in readiness.workers] == ["vc"]
    fields = {p.field for p in readiness.workers[0].problems}
    assert "rvc.model_file" in fields
    assert not readiness.workers[0].ok


def test_flow_starts_at_recording_when_recording_seeds():
    config = Config(
        recording=RecordingConfig(enable=True, routes_list=[["vc", "playback"]])
    )
    assert flow_of(config) == [["recording", "vc", "playback"]]


def test_flow_uses_text_send_operations_without_recording():
    config = Config(text_send_operations=[["tts", "playback"]])
    assert flow_of(config) == [["(text)", "tts", "playback"]]


def test_worker_names_are_all_enableable_config_sections():
    # WORKER_NAMES は「.enable を持つ config セクション」の集合でなければならない
    # (ADR-0045 の enable ゲート境界)。preflight に新しい worker を足したのに
    # WORKER_NAMES を更新し忘れると readiness が黙って過少報告する — その drift の
    # WORKER_NAMES 側をここで固定する (typo / 消えたセクション名を検出)。
    config = Config()
    for name in WORKER_NAMES:
        section = getattr(config, name)
        assert isinstance(section.enable, bool)


def test_evaluation_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    def boom(_config: Config) -> list[object]:
        raise ImportError("sounddevice missing")

    monkeypatch.setattr("gui.readiness.collect_problems", boom)
    readiness = evaluate(Config(vc=VcConfig(enable=True)))
    assert not readiness.ok
    assert readiness.error is not None
    assert "sounddevice missing" in readiness.error
