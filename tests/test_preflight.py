from pathlib import Path

import pytest
from pydantic import SecretStr

from vspeech.config import AmiConfig
from vspeech.config import Config
from vspeech.config import GcpConfig
from vspeech.config import SubtitleWorkerType
from vspeech.config import TranscriptionConfig
from vspeech.config import TranscriptionWorkerType
from vspeech.config import VcConfig
from vspeech.exceptions import ConfigError
from vspeech.exceptions import ConfigProblem
from vspeech.lib.audio import DeviceInfo
from vspeech.lib.subtitle_state import TRANSPARENT_BG_COLOR
from vspeech.preflight import _check_subtitle
from vspeech.preflight import collect_problems
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


def test_vc_fcpe_missing_model_file_reported():
    from vspeech.config import F0ExtractorType
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    cfg = Config(
        vc=VcConfig(enable=True),
        rvc=RvcConfig(
            model_file=Path("/no/model.onnx"),
            hubert_model_file=Path("/no/hubert"),
            f0_extractor_type=F0ExtractorType.fcpe,
            fcpe_model_file=Path("/no/fcpe.onnx"),
        ),
    )
    with pytest.raises(ConfigError) as ei:
        preflight(cfg)
    problems = ei.value.problems
    assert any(p.field == "rvc.fcpe_model_file" for p in problems)
    # fcpe 選択時は rmvpe_model_file 不在を咎めない
    assert not any(p.field == "rvc.rmvpe_model_file" for p in problems)


def test_vc_fcpe_present_model_file_passes(tmp_path):
    from vspeech.config import F0ExtractorType
    from vspeech.config import RvcConfig
    from vspeech.config import VcConfig

    model = tmp_path / "m.onnx"
    hub = tmp_path / "hub"
    fcpe = tmp_path / "fcpe.onnx"
    model.write_bytes(b"x")
    hub.mkdir()
    fcpe.write_bytes(b"x")
    cfg = Config(
        vc=VcConfig(enable=True, vad_gate=False),
        rvc=RvcConfig(
            model_file=model,
            hubert_model_file=hub,
            f0_extractor_type=F0ExtractorType.fcpe,
            fcpe_model_file=fcpe,
        ),
    )
    # 全アセット存在 -> ConfigError を上げない
    preflight(cfg)


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


def test_obs_backend_requires_text_source_but_not_translated_source():
    # text_source is the default route (ingest_text falls back to the "n"
    # panel when position is unset) so an empty one leaves the backend
    # doing nothing -- still fatal. translated_source has no such default
    # fallback (a routed p=s message just gets dropped with a warn-once,
    # see worker/subtitle_obs.py) so an empty one just means this pipeline
    # has no translation step -- not required (ADR-0041/0042).
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    problems = _check_subtitle(config)
    details = " ".join(p.detail for p in problems)
    assert "text_source" in details
    assert "translated_source" not in details


def test_obs_backend_accepts_an_empty_translated_source():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = ""
    assert _check_subtitle(config) == []


def test_obs_backend_still_rejects_an_empty_text_source():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = ""
    config.subtitle.obs.translated_source = "vspeech-translated"
    problems = _check_subtitle(config)
    assert len(problems) == 1
    assert "text_source" in problems[0].detail


def test_obs_backend_accepts_a_complete_config():
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "vspeech-text"
    config.subtitle.obs.translated_source = "vspeech-translated"
    assert _check_subtitle(config) == []


def _obs_config() -> Config:
    config = Config()
    config.subtitle.enable = True
    config.subtitle.worker_type = SubtitleWorkerType.OBS
    config.subtitle.obs.text_source = "t"
    config.subtitle.obs.translated_source = "s"
    return config


# A Tk-valid colour name ("white", "green",
# "#fff") is accepted by pydantic and by the TK backend but is not
# `#rrggbb`, so `hex_color_to_obs_int` raises `ValueError` at runtime deep in
# the OBS worker (build_text_settings). Flipping `worker_type` TK -> OBS is
# ADR-0040's advertised migration path, so this must be caught here (FATAL,
# startup) rather than crash the whole audio pipeline later.
@pytest.mark.parametrize("bad", ["white", "green", "#fff"])
def test_obs_backend_rejects_a_tk_only_font_color(bad: str):
    config = _obs_config()
    config.subtitle.text.font_color = bad
    problems = _check_subtitle(config)
    assert any("subtitle.text.font_color" in p.detail for p in problems)


def test_obs_backend_rejects_a_tk_only_outline_color():
    config = _obs_config()
    config.subtitle.text.outline_color = "white"
    problems = _check_subtitle(config)
    assert any("subtitle.text.outline_color" in p.detail for p in problems)


def test_obs_backend_rejects_a_tk_only_translated_font_color():
    config = _obs_config()
    config.subtitle.translated.font_color = "white"
    problems = _check_subtitle(config)
    assert any("subtitle.translated.font_color" in p.detail for p in problems)


def test_obs_backend_rejects_a_tk_only_translated_outline_color():
    config = _obs_config()
    config.subtitle.translated.outline_color = "white"
    problems = _check_subtitle(config)
    assert any("subtitle.translated.outline_color" in p.detail for p in problems)


def test_obs_backend_rejects_a_tk_only_bg_color():
    config = _obs_config()
    config.subtitle.bg_color = "white"
    problems = _check_subtitle(config)
    assert any("subtitle.bg_color" in p.detail for p in problems)


def test_obs_backend_accepts_the_transparent_bg_sentinel():
    # bg_color legitimately accepts the TRANSPARENT_BG_COLOR sentinel in
    # addition to #rrggbb -- lib/obs_text_settings.build_text_settings
    # special-cases it, and preflight must mirror that exactly rather than
    # reject it as a bad hex colour.
    config = _obs_config()
    config.subtitle.bg_color = TRANSPARENT_BG_COLOR
    assert _check_subtitle(config) == []


def test_obs_backend_reports_every_bad_color_not_just_the_first():
    # ADR-0038 aggregates all problems; a single bad-color check must not
    # stop at the first field.
    config = _obs_config()
    config.subtitle.text.font_color = "white"
    config.subtitle.text.outline_color = "green"
    config.subtitle.translated.font_color = "blue"
    config.subtitle.translated.outline_color = "red"
    config.subtitle.bg_color = "yellow"
    details = " ".join(p.detail for p in _check_subtitle(config))
    assert "subtitle.text.font_color" in details
    assert "subtitle.text.outline_color" in details
    assert "subtitle.translated.font_color" in details
    assert "subtitle.translated.outline_color" in details
    assert "subtitle.bg_color" in details


def test_collect_problems_returns_list_without_raising():
    problems = collect_problems(_acp())  # ACP 4 フィールドすべて空
    assert [p.worker for p in problems] == ["transcription"] * 4
    assert {p.field for p in problems} == {
        "ami.appkey",
        "ami.engine_uri",
        "ami.engine_name",
        "ami.service_id",
    }


def test_collect_problems_empty_for_clean_config():
    assert collect_problems(Config()) == []


def test_preflight_still_raises_on_problems():
    with pytest.raises(ConfigError):
        preflight(_acp())


def test_config_problem_str_is_unchanged_by_field():
    problem = ConfigProblem("vc", "rvc.model_file '' が存在しません", "rvc.model_file")
    assert str(problem) == "[vc] rvc.model_file '' が存在しません"


def test_vc_problems_carry_their_field():
    config = Config(vc=VcConfig(enable=True))
    fields = {p.field for p in collect_problems(config)}
    assert "rvc.model_file" in fields
    assert "rvc.hubert_model_file" in fields
    assert "rvc.rmvpe_model_file" in fields  # f0_extractor_type の既定は rmvpe


def _fields(problems):
    return {p.field for p in problems}


def test_consumer_requires_bind_not_rvc_or_input():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "consumer", "transport_type": "udp"}}
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.bind_port" in fields
    assert not any(f.startswith("stream_vc.rvc") for f in fields)
    assert "stream_vc.input_device_index" not in fields


def test_producer_requires_peer_and_input_not_output():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "producer", "transport_type": "udp"}}
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.peer_port" in fields
    assert "stream_vc.output_device_index" not in fields


def test_non_local_role_requires_udp_transport():
    cfg = Config.model_validate(
        {
            "stream_vc": {"enable": True, "role": "consumer"}
        }  # transport defaults in_process
    )
    assert "stream_vc.transport_type" in _fields(collect_problems(cfg))
