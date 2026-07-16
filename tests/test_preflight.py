from pathlib import Path

import pytest
from pydantic import SecretStr

from vspeech.config import AmiConfig
from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import SubtitleWorkerType
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.exceptions import ConfigError
from vspeech.lib.audio import DeviceInfo
from vspeech.preflight import _check_subtitle
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


def test_translation_missing_gcp_key_is_reported():
    from vspeech.config import GcpConfig
    from vspeech.config import TranslationConfig

    cfg = Config(
        translation=TranslationConfig(enable=True),
        gcp=GcpConfig(service_account_file_path=Path("/no/such/key.json")),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any(p.worker == "translation" for p in ei.value.problems)


def test_voicevox_missing_dirs_reported():
    from vspeech.config import TtsConfig
    from vspeech.config import TtsWorkerType
    from vspeech.config import VoicevoxConfig

    cfg = Config(
        tts=TtsConfig(enable=True, worker_type=TtsWorkerType.VOICEVOX),
        voicevox=VoicevoxConfig(
            openjtalk_dir=Path("/no/dict"), model_dir=Path("/no/models")
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    details = [p.detail for p in ei.value.problems]
    assert any("voicevox.openjtalk_dir" in d for d in details)
    assert any("voicevox.model_dir" in d for d in details)


def test_vr2_tts_passes_without_files():
    # VR2 は実初期化が層B。preflight は通す。
    from vspeech.config import TtsConfig

    preflight(Config(tts=TtsConfig(enable=True)))  # 既定 worker_type=VR2


def test_vc_unconfigured_hubert_dir_is_reported():
    from vspeech.config import VcConfig

    # RvcConfig defaults leave hubert_model_file = Path() (== "."), which is a
    # real directory; the check must still report it as unconfigured.
    cfg = Config(vc=VcConfig(enable=True))
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    assert any("rvc.hubert_model_file" in p.detail for p in ei.value.problems)


def test_vc_missing_model_files_reported():
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    cfg = Config(
        vc=VcConfig(enable=True),
        rvc=RvcConfig(
            model_file=Path("/no/model.onnx"),
            hubert_model_file=Path("/no/hubert"),
            rmvpe_model_file=Path("/no/rmvpe.onnx"),
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    details = [p.detail for p in ei.value.problems]
    assert any("rvc.model_file" in d for d in details)
    assert any("rvc.hubert_model_file" in d for d in details)
    assert any("rvc.rmvpe_model_file" in d for d in details)


def test_vc_all_present_passes(tmp_path):
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    hubert = tmp_path / "hubert"
    hubert.mkdir()
    rmvpe = tmp_path / "rmvpe.onnx"
    rmvpe.write_bytes(b"x")
    cfg = Config(
        vc=VcConfig(enable=True),
        rvc=RvcConfig(
            model_file=model, hubert_model_file=hubert, rmvpe_model_file=rmvpe
        ),
    )
    preflight(cfg)  # all present -> no ConfigError


def test_vc_non_rmvpe_extractor_skips_rmvpe_check(tmp_path):
    from vspeech.config import F0ExtractorType
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    hubert = tmp_path / "hubert"
    hubert.mkdir()
    # rmvpe_model_file left at its (missing) default; a non-rmvpe extractor must
    # NOT trigger the rmvpe existence check.
    cfg = Config(
        vc=VcConfig(enable=True),
        rvc=RvcConfig(
            model_file=model,
            hubert_model_file=hubert,
            f0_extractor_type=F0ExtractorType.dio,
        ),
    )
    preflight(cfg)  # rmvpe not checked -> no ConfigError


def test_subtitle_tk_backend_is_not_checked():
    # TK 構成に新しい失敗を持ち込まない (ADR-0042)。
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.TK
    config.subtitle.obs.url = ""
    assert _check_subtitle(config) == []


def test_disabled_subtitle_is_not_checked():
    config = Config()
    config.subtitle.enable = False
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = ""
    assert _check_subtitle(config) == []


def test_obs_backend_requires_a_url():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = ""
    config.subtitle.obs.text_source = "t"
    config.subtitle.obs.translated_source = "s"
    problems = _check_subtitle(config)
    assert len(problems) == 1
    assert "url" in problems[0].detail


def test_obs_backend_rejects_a_non_websocket_url():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.url = "http://127.0.0.1:4455"
    config.subtitle.obs.text_source = "t"
    config.subtitle.obs.translated_source = "s"
    problems = _check_subtitle(config)
    assert len(problems) == 1
    assert "ws://" in problems[0].detail


def test_obs_backend_requires_both_source_names_and_reports_both():
    # ADR-0038 は「全問題を集約」する。1 個目で打ち切らない。
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    problems = _check_subtitle(config)
    details = " ".join(p.detail for p in problems)
    assert "text_source" in details
    assert "translated_source" in details


def test_obs_backend_accepts_a_complete_config():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = "vspeech-translated"
    assert _check_subtitle(config) == []
