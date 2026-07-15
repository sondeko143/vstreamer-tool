from pathlib import Path

import pytest
from pydantic import SecretStr

from vspeech.config import AmiConfig
from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.exceptions import ConfigError
from vspeech.lib.audio import DeviceInfo
from vspeech.preflight import preflight


def _device(index: int = 1):
    return DeviceInfo(
        host_api=0,
        max_input_channels=2,
        max_output_channels=2,
        name="Line 4",
        index=index,
    )


def _acp(**ami_kw):
    return Config(
        transcription=TranscriptionConfig(
            enable=True, worker_type=TranscriptionWorkerType.ACP
        ),
        ami=AmiConfig(**ami_kw),
    )


def _full_acp():
    return _acp(
        appkey=SecretStr("k"), engine_uri="https://e", engine_name="g", service_id="s"
    )


def test_disabled_worker_is_not_checked():
    # transcription 無効なら ami 空でも問題なし
    preflight(Config())


def test_acp_missing_fields_are_all_reported():
    with pytest.raises(ConfigError) as ei:
        preflight(_acp())  # 4 フィールドすべて空
    details = [p.detail for p in ei.value.problems]
    assert any("ami.appkey" in d for d in details)
    assert any("ami.engine_uri" in d for d in details)
    assert any("ami.engine_name" in d for d in details)
    assert any("ami.service_id" in d for d in details)
    assert all(p.worker == "transcription" for p in ei.value.problems)


def test_acp_complete_config_passes():
    preflight(_full_acp())


def test_gcp_missing_key_file_is_reported():
    cfg = Config(
        transcription=TranscriptionConfig(
            enable=True, worker_type=TranscriptionWorkerType.GCP
        ),
        gcp=GcpConfig(service_account_file_path=Path("/no/such/key.json")),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("service_account_file_path" in p.detail for p in ei.value.problems)


def test_vad_gate_missing_model_is_reported():
    cfg = Config(
        transcription=TranscriptionConfig(
            enable=True,
            worker_type=TranscriptionWorkerType.ACP,
            vad_gate=True,
            vad_model_file=Path("/no/such/silero_vad.onnx"),
        ),
        ami=AmiConfig(
            appkey=SecretStr("k"),
            engine_uri="https://e",
            engine_name="g",
            service_id="s",
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("vad_model_file" in p.detail for p in ei.value.problems)


def test_cmd_exits_on_config_error(monkeypatch):
    import asyncio

    from vspeech.exceptions import ConfigError
    from vspeech.exceptions import ConfigProblem
    from vspeech.main import cmd

    def _boom(config):
        raise ConfigError([ConfigProblem("transcription", "boom")])

    monkeypatch.setattr("vspeech.main.preflight", _boom)
    monkeypatch.setattr("vspeech.main.configure_logger", lambda config: None)
    monkeypatch.setattr("vspeech.main.telemetry.configure", lambda **kw: None)
    asyncio.set_event_loop(None)
    assert cmd.callback is not None
    with pytest.raises(SystemExit) as ei:
        cmd.callback(config_file=None)
    assert ei.value.code == 1


def test_recording_device_not_found_is_reported(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "search_device", lambda **kw: None)
    cfg = Config(recording=RecordingConfig(enable=True, input_device_name="Ghost"))
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any(p.worker == "recording" for p in ei.value.problems)


def test_recording_bad_route_is_reported(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(audio, "get_device_info", lambda i: _device(i))
    cfg = Config(
        recording=RecordingConfig(
            enable=True, input_device_index=1, routes_list=[["not_an_event"]]
        )
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("routes_list" in p.detail for p in ei.value.problems)
