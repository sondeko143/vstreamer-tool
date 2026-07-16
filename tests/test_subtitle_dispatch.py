import sys

from vspeech.config import Config
from vspeech.config import SubtitleWorkerType


def test_subtitle_worker_type_defaults_to_tk():
    # 既存 config (worker_type 未指定) の挙動を変えない。
    assert Config().subtitle.worker_type == SubtitleWorkerType.TK


def test_subtitle_worker_type_round_trips_through_toml():
    config = Config()
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    reloaded = Config.model_validate({"subtitle": {"worker_type": "OBS"}})
    assert reloaded.subtitle.worker_type == SubtitleWorkerType.OBS


def test_importing_the_subtitle_dispatcher_does_not_import_tkinter():
    """ヘッドレス目的の要。ディスパッチャ経由で tkinter が引き込まれないこと (ADR-0040)。

    tkinter は stdlib なので他経路で既に入っていることがある。ここでは
    「subtitle が tkinter に依存していないこと」ではなく「subtitle を import
    しても tkinter が新たに読み込まれないこと」を見たいので、一度落として
    から確かめる。
    """
    for name in list(sys.modules):
        if name == "tkinter" or name.startswith("tkinter."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("vspeech.worker.subtitle"):
            del sys.modules[name]

    import vspeech.worker.subtitle  # noqa: F401

    assert "tkinter" not in sys.modules
