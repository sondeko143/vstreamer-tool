import io
import sys

from pydantic import SecretStr

from vspeech.config import Config
from vspeech.config import SubtitleWorkerType


def test_subtitle_worker_type_defaults_to_tk():
    # 既存 config (worker_type 未指定) の挙動を変えない。
    assert Config().subtitle.worker_type == SubtitleWorkerType.TK


def test_subtitle_worker_type_round_trips_through_toml():
    config = Config()
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    toml_bytes = config.export_to_toml().encode("utf-8")
    toml_file = io.BytesIO(toml_bytes)
    toml_file.name = "config.toml"
    reloaded = Config.read_config_from_file(toml_file)
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


def test_obs_password_survives_a_toml_round_trip():
    """export_to_toml は SecretStr をハードコードで展開している。新しい secret を
    足したらここも足さないと、GUI の保存が config を壊す。"""
    from vspeech.config import Config

    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.password = SecretStr("hunter2")
    dumped = config.export_to_toml()
    assert "hunter2" in dumped
    assert "**" not in dumped
    reloaded = Config.read_config_from_file(
        _named_bytes_io(dumped.encode("utf-8"), "config.toml")
    )
    assert reloaded.subtitle.obs.password.get_secret_value() == "hunter2"


def _named_bytes_io(data: bytes, name: str):
    import io

    buf = io.BytesIO(data)
    buf.name = name
    return buf
