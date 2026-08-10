import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


# --- sounddevice stub (Task 9, ADR-0076) --------------------------------------------
#
# Task 9 makes preflight resolve+open the device rate for recording/playback/stream_vc,
# which reaches sd.query_hostapis / sd.query_devices / sd.default / sd.RawInputStream /
# sd.RawOutputStream directly (below the audio.get_device_info/search_device wrappers
# some tests above already monkeypatch). Without a stub, any of those tests that leave
# the device unconfigured (falling through to sd.default.device) would resolve to and
# briefly open this machine's REAL microphone/speaker on every test run. The fixture
# below is autouse so no test in this file can do that by omission; a test that wants a
# specific device table or open outcome overrides these further with its own
# monkeypatch calls (the same monkeypatch fixture instance, so later calls win).
_SD_HOSTAPIS = [{"name": "Windows WASAPI"}]
_SD_DEVICES = [
    {
        "index": 0,
        "name": "Test Mic",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "index": 1,
        "name": "Test Speaker",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
]


class _FakeSdStream:
    """Stands in for sd.RawInputStream/RawOutputStream. Opens successfully by default.

    `start_error`, when given, makes `start()` raise instead -- the
    Pa_OpenStream-succeeds/Pa_StartStream-fails case a real sounddevice stream can hit
    (e.g. the device grabbed exclusively between open and start). `closed` records
    whether `close()` ran, so a test can prove the stream is never leaked either way.
    """

    def __init__(self, start_error: Exception | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.samplerate = float(kwargs["samplerate"])
        self.closed = False
        self._start_error = start_error

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _stub_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    from vspeech.lib import audio

    def _query_devices(index: int | None = None):
        if index is None:
            return list(_SD_DEVICES)
        return next(d for d in _SD_DEVICES if d["index"] == index)

    monkeypatch.setattr(audio.sd, "query_devices", _query_devices)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: _SD_HOSTAPIS)
    # Device 0 (input) / 1 (output) so the role-branching tests below, which leave the
    # device unconfigured on purpose (they test which *fields* get checked per role, not
    # device specifics), resolve to a real-looking fake instead of sd.default's real
    # system device.
    monkeypatch.setattr(audio.sd, "default", SimpleNamespace(device=(0, 1)))
    monkeypatch.setattr(audio.sd, "RawInputStream", lambda **kw: _FakeSdStream(**kw))
    monkeypatch.setattr(audio.sd, "RawOutputStream", lambda **kw: _FakeSdStream(**kw))


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
    # With transcription disabled, an empty ami is fine
    preflight(Config())


def test_acp_missing_fields_are_all_reported():
    with pytest.raises(ConfigError) as ei:
        preflight(_acp())  # all four fields empty
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


def test_cmd_exits_on_config_error(monkeypatch, tmp_path):
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
    config_file = tmp_path / "config.toml"
    config_file.write_text("", encoding="utf-8")
    with config_file.open("rb") as opened:
        with pytest.raises(SystemExit) as ei:
            cmd.callback(config_file=opened)
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


# --- recording: device rate resolution + open probe (Task 9, ADR-0076) -------------


def test_recording_rate_unresolved_is_reported_on_its_own_field(monkeypatch):
    """DeviceRateUnresolvedError is a DeviceNotFoundError subclass -- the trap this
    task warns about is the *index* handler swallowing it under the wrong field. Device
    resolution itself must succeed (no input_device_index problem) while the *rate*
    handler reports it separately, under input_device_rate."""
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio,
        "get_device_info",
        lambda i: DeviceInfo(
            host_api=0,
            max_input_channels=2,
            max_output_channels=0,
            name="Ghost Mic",  # not in _SD_DEVICES and not a WASAPI row itself
            index=i,
        ),
    )
    cfg = Config(recording=RecordingConfig(enable=True, input_device_index=5))
    problems = collect_problems(cfg)
    assert any(p.field == "recording.input_device_rate" for p in problems), problems
    assert not any(p.field == "recording.input_device_index" for p in problems)


def test_recording_device_open_failure_reports_the_rate(monkeypatch):
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )

    def _boom(**kw):
        raise audio.sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(audio.sd, "RawInputStream", _boom)
    cfg = Config(recording=RecordingConfig(enable=True, input_device_index=0))
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "recording.input_device_rate"]
    assert len(matches) == 1, problems
    assert "48000" in matches[0].detail  # the resolved rate, not just the error text
    # The probe's own shape, so a future reader does not blame the rate alone for a
    # failure that might really be about channels/format/device contention.
    assert "channels=1" in matches[0].detail
    assert "dtype=int16" in matches[0].detail


def test_recording_start_failure_still_closes_the_stream(monkeypatch):
    """Pa_OpenStream succeeding but Pa_StartStream failing must still close the stream
    -- an unclosed sounddevice stream has no __del__ and leaks the native handle for
    good (found in review, ADR-0076). Direct assertion, not a mutation: this proves the
    fix, not merely that some other test would fail without it."""
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    made: list[_FakeSdStream] = []

    def _open(**kw):
        stream = _FakeSdStream(start_error=audio.sd.PortAudioError("device busy"), **kw)
        made.append(stream)
        return stream

    monkeypatch.setattr(audio.sd, "RawInputStream", _open)
    cfg = Config(recording=RecordingConfig(enable=True, input_device_index=0))
    problems = collect_problems(cfg)
    assert any(p.field == "recording.input_device_rate" for p in problems), problems
    assert len(made) == 1
    assert made[0].closed is True


def test_recording_pathological_ratio_does_not_open_the_device(monkeypatch):
    """Once the ratio check already condemns the device rate, the open probe must not
    run at all -- touching hardware known to be unusable is wasted and could report a
    second, unrelated problem on the same field (ADR-0076)."""
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    opened: list[int] = []
    monkeypatch.setattr(
        audio.sd,
        "RawInputStream",
        lambda **kw: opened.append(1) or _FakeSdStream(**kw),
    )
    cfg = Config(
        recording=RecordingConfig(
            enable=True, input_device_index=0, input_device_rate=44101
        )
    )
    collect_problems(cfg)
    assert opened == []


def test_recording_pathological_ratio_is_rejected_at_preflight(monkeypatch):
    """recording.rate (config.recording.rate) is a fixed pipeline rate, so a device
    rate that cannot resample to it fails every time the worker runs -- reject it at
    startup instead (ADR-0076)."""
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    # An explicit override skips device-table lookup entirely (resolve_device_rate
    # returns it immediately), so this is deterministic regardless of the device table.
    cfg = Config(
        recording=RecordingConfig(
            enable=True, input_device_index=0, input_device_rate=44101
        )
    )
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "recording.input_device_rate"]
    assert len(matches) == 1, problems
    assert "44101" in matches[0].detail
    assert "16000" in matches[0].detail  # config.recording.rate's default


def test_recording_realistic_rate_pair_passes(monkeypatch):
    # The autouse fixture's device resolves to 48000Hz; against recording.rate=16000
    # (the default) that is an ordinary, safe ratio.
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    cfg = Config(recording=RecordingConfig(enable=True, input_device_index=0))
    assert collect_problems(cfg) == []


def test_recording_rate_drift_is_logged_not_a_config_problem(monkeypatch, caplog):
    """Parity with the worker's own open path (lib/audio.open_device_stream step 4): a
    rate PortAudio reports back a hair off from what was requested is not itself a
    config problem (the worker still converts at the requested rate) -- it becomes
    visible at preflight time too, as a warning, instead of only once the real worker
    opens the same device (ADR-0076)."""
    from vspeech.config import RecordingConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )

    class _DriftingStream(_FakeSdStream):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.samplerate += 1.0  # PortAudio reports a hair off the requested rate

    monkeypatch.setattr(audio.sd, "RawInputStream", lambda **kw: _DriftingStream(**kw))
    cfg = Config(recording=RecordingConfig(enable=True, input_device_index=0))
    with caplog.at_level(logging.WARNING):
        problems = collect_problems(cfg)
    assert problems == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("recording.input_device_rate" in m and "48001" in m for m in messages)


# --- playback: device rate resolution + open probe (Task 9, ADR-0076) --------------


def test_playback_rate_unresolved_is_reported(monkeypatch):
    from vspeech.config import PlaybackConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio,
        "get_device_info",
        lambda i: DeviceInfo(
            host_api=0,
            max_input_channels=0,
            max_output_channels=2,
            name="Ghost Speaker",
            index=i,
        ),
    )
    cfg = Config(playback=PlaybackConfig(enable=True, output_device_index=9))
    problems = collect_problems(cfg)
    assert any(p.field == "playback.output_device_rate" for p in problems), problems
    assert not any(p.field == "playback.output_device_index" for p in problems)


def test_playback_device_open_failure_reports_the_rate(monkeypatch):
    from vspeech.config import PlaybackConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )

    def _boom(**kw):
        raise audio.sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(audio.sd, "RawOutputStream", _boom)
    cfg = Config(playback=PlaybackConfig(enable=True, output_device_index=1))
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "playback.output_device_rate"]
    assert len(matches) == 1, problems
    assert "48000" in matches[0].detail
    assert "channels=1" in matches[0].detail
    assert "dtype=int16" in matches[0].detail


def test_playback_pathological_rate_is_not_rejected_at_preflight(monkeypatch):
    """Unlike recording/stream_vc, playback's counterpart rate is whatever TTS/VC/a
    remote worker produced the utterance -- not known until an utterance actually
    arrives -- so preflight must not reject a device rate here even one that would
    explode against every standard pipeline rate. worker/playback.py already handles
    it per-utterance (a warning, ADR-0075/0076)."""
    from vspeech.config import PlaybackConfig
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )
    cfg = Config(
        playback=PlaybackConfig(
            enable=True, output_device_index=1, output_device_rate=44101
        )
    )
    assert collect_problems(cfg) == []


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
    # VR2's real initialization is layer B, so preflight lets it through.
    from vspeech.config import TtsConfig

    preflight(Config(tts=TtsConfig(enable=True)))  # the default worker_type=VR2


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
    # With fcpe selected, a missing rmvpe_model_file is not flagged
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
    # every asset present -> no ConfigError is raised
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
    # Never introduce a new failure into a TK configuration (ADR-0042).
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
    problems = collect_problems(_acp())  # all four ACP fields empty
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
    assert "rvc.rmvpe_model_file" in fields  # f0_extractor_type defaults to rmvpe


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


def test_local_role_rejects_udp_transport():
    cfg = Config.model_validate(
        {"stream_vc": {"enable": True, "role": "local", "transport_type": "udp"}}
    )
    problems = collect_problems(cfg)
    assert "stream_vc.transport_type" in _fields(problems)
    assert any("無視されます" in p.detail for p in problems)


def test_local_role_default_transport_has_no_transport_problem():
    # The defaults (role=local, transport_type=in_process) raise no transport problem.
    cfg = Config.model_validate({"stream_vc": {"enable": True}})
    assert "stream_vc.transport_type" not in _fields(collect_problems(cfg))


# --- stream_vc: device rate resolution + open probe (Task 9, ADR-0076) -------------


def test_stream_vc_input_rate_unresolved_is_reported(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio,
        "get_device_info",
        lambda i: DeviceInfo(
            host_api=0,
            max_input_channels=2,
            max_output_channels=0,
            name="Ghost Stream Mic",
            index=i,
        ),
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "producer",
                "transport_type": "udp",
                "peer_host": "127.0.0.1",
                "peer_port": 9000,
                "input_device_index": 3,
            }
        }
    )
    problems = collect_problems(cfg)
    assert any(p.field == "stream_vc.input_device_rate" for p in problems), problems
    assert not any(p.field == "stream_vc.input_device_index" for p in problems)


def test_stream_vc_output_rate_unresolved_is_reported(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio,
        "get_device_info",
        lambda i: DeviceInfo(
            host_api=0,
            max_input_channels=0,
            max_output_channels=2,
            name="Ghost Stream Speaker",
            index=i,
        ),
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "consumer",
                "transport_type": "udp",
                "bind_port": 9000,
                "output_device_index": 4,
            }
        }
    )
    problems = collect_problems(cfg)
    assert any(p.field == "stream_vc.output_device_rate" for p in problems), problems
    assert not any(p.field == "stream_vc.output_device_index" for p in problems)


def test_stream_vc_output_device_open_failure_reports_the_rate(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )

    def _boom(**kw):
        raise audio.sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(audio.sd, "RawOutputStream", _boom)
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "consumer",
                "transport_type": "udp",
                "bind_port": 9000,
                "output_device_index": 1,
            }
        }
    )
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "stream_vc.output_device_rate"]
    assert len(matches) == 1, problems
    assert "48000" in matches[0].detail
    assert "channels=1" in matches[0].detail
    assert "dtype=int16" in matches[0].detail


def test_stream_vc_output_start_failure_still_closes_the_stream(monkeypatch):
    """Same guarantee as recording (see
    test_recording_start_failure_still_closes_the_stream) for the output boundary."""
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )
    made: list[_FakeSdStream] = []

    def _open(**kw):
        stream = _FakeSdStream(start_error=audio.sd.PortAudioError("device busy"), **kw)
        made.append(stream)
        return stream

    monkeypatch.setattr(audio.sd, "RawOutputStream", _open)
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "consumer",
                "transport_type": "udp",
                "bind_port": 9000,
                "output_device_index": 1,
            }
        }
    )
    problems = collect_problems(cfg)
    assert any(p.field == "stream_vc.output_device_rate" for p in problems), problems
    assert len(made) == 1
    assert made[0].closed is True


def test_stream_vc_input_pathological_ratio_is_rejected(monkeypatch):
    """CAPTURE_RATE (16000) is a fixed pipeline rate for this boundary, so a device
    rate that cannot resample to it fails every time the worker runs -- reject it at
    startup instead (ADR-0076)."""
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "producer",
                "transport_type": "udp",
                "peer_host": "127.0.0.1",
                "peer_port": 9000,
                "input_device_index": 0,
                "input_device_rate": 44101,
            }
        }
    )
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "stream_vc.input_device_rate"]
    assert len(matches) == 1, problems
    assert "44101" in matches[0].detail
    assert "16000" in matches[0].detail  # CAPTURE_RATE


def test_stream_vc_output_pathological_ratio_is_rejected(monkeypatch):
    """stream_vc output has no single known counterpart rate (it comes from whichever
    RVC model is loaded), so the check is against the whole standard rate family
    instead (ADR-0076) -- one exploding member is enough to prove the device rate
    itself is the problem."""
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "consumer",
                "transport_type": "udp",
                "bind_port": 9000,
                "output_device_index": 1,
                "output_device_rate": 44101,
            }
        }
    )
    problems = collect_problems(cfg)
    matches = [p for p in problems if p.field == "stream_vc.output_device_rate"]
    assert len(matches) == 1, problems
    assert "44101" in matches[0].detail


def test_stream_vc_output_realistic_rates_all_pass(monkeypatch):
    """The controller requirement's own example combinations must all pass -- none may
    be rejected as a pathological ratio against the standard rate family."""
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )
    for rate in (16000, 22050, 24000, 32000, 40000, 44100, 48000, 96000, 192000):
        cfg = Config.model_validate(
            {
                "stream_vc": {
                    "enable": True,
                    "role": "consumer",
                    "transport_type": "udp",
                    "bind_port": 9000,
                    "output_device_index": 1,
                    "output_device_rate": rate,
                }
            }
        )
        assert collect_problems(cfg) == [], (rate, collect_problems(cfg))


def test_producer_validates_input_rate_not_output_rate(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[0])
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "producer",
                "transport_type": "udp",
                "peer_host": "127.0.0.1",
                "peer_port": 9000,
                "input_device_index": 0,
                "input_device_rate": 44101,  # pathological, must be reported
                "output_device_rate": 44101,  # also pathological, must NOT even be
                # checked -- producer never touches the output boundary at all
            }
        }
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.input_device_rate" in fields
    assert "stream_vc.output_device_rate" not in fields


def test_consumer_validates_output_rate_not_input_rate(monkeypatch):
    from vspeech.lib import audio

    monkeypatch.setattr(
        audio, "get_device_info", lambda i: DeviceInfo.model_validate(_SD_DEVICES[1])
    )
    cfg = Config.model_validate(
        {
            "stream_vc": {
                "enable": True,
                "role": "consumer",
                "transport_type": "udp",
                "bind_port": 9000,
                "output_device_index": 1,
                "output_device_rate": 44101,  # pathological, must be reported
                "input_device_rate": 44101,  # also pathological, must NOT even be
                # checked -- consumer never touches the input boundary at all
            }
        }
    )
    fields = _fields(collect_problems(cfg))
    assert "stream_vc.output_device_rate" in fields
    assert "stream_vc.input_device_rate" not in fields


def test_disabled_recording_playback_stream_vc_skip_rate_checks():
    """`input_device_rate=1` / `output_device_rate=1` are deliberately pathological
    (1Hz cannot resample to any real pipeline rate) on all three, including stream_vc:
    with the enable guard removed, role=local's default would check *both*
    input_device_rate against CAPTURE_RATE and output_device_rate against the standard
    rate family, and either one would explode -- so this test only passes if `enable`
    genuinely stops the rate check specifically, not just incidentally because some
    other stream_vc check (e.g. missing RVC assets) also happens to fire."""
    from vspeech.config import PlaybackConfig
    from vspeech.config import RecordingConfig
    from vspeech.config import StreamVcConfig

    cfg = Config(
        recording=RecordingConfig(enable=False, input_device_rate=1),
        playback=PlaybackConfig(enable=False, output_device_rate=1),
        stream_vc=StreamVcConfig(
            enable=False, input_device_rate=1, output_device_rate=1
        ),
    )
    assert collect_problems(cfg) == []


def test_standard_sample_rate_family_never_needs_a_pathological_resampler():
    """Regression guard for the proxy set stream_vc's output check uses (ADR-0076):
    every pair drawn from _STANDARD_SAMPLE_RATES must resample without exceeding
    MAX_PROTOTYPE_TAPS (ADR-0075), or the output check would reject legitimate
    device-rate/model-rate combinations that must pass."""
    from vspeech.lib.resample import make_resampler
    from vspeech.preflight import _STANDARD_SAMPLE_RATES

    for a in _STANDARD_SAMPLE_RATES:
        for b in _STANDARD_SAMPLE_RATES:
            make_resampler(a, b)  # must not raise ValueError
