from pathlib import Path

from scripts import startup_capture as sc


def test_venv_python_win32():
    assert sc.venv_python(Path("/proj"), "win32") == Path(
        "/proj/.venv/Scripts/python.exe"
    )


def test_venv_python_posix():
    assert sc.venv_python(Path("/proj"), "linux") == Path("/proj/.venv/bin/python")


def test_build_vspeech_cmd():
    assert sc.build_vspeech_cmd("py", "cfg.toml") == [
        "py",
        "-m",
        "vspeech",
        "--config",
        "cfg.toml",
    ]


def test_build_pyspy_cmd_spawns_vspeech_with_subprocesses_idle():
    cmd = sc.build_pyspy_cmd(
        out="o.json", python="py", config="cfg.toml", duration=30, rate=100
    )
    assert cmd[:3] == ["uvx", "py-spy", "record"]
    assert "speedscope" in cmd
    assert "-o" in cmd and "o.json" in cmd
    # follow the uv venv trampoline into the real interpreter it launches
    assert "--subprocesses" in cmd
    # sample threads even while blocked on I/O (Python releases the GIL there)
    assert "--idle" in cmd
    assert "--duration" in cmd and "30" in cmd
    # the vspeech command sits after the `--` separator
    sep = cmd.index("--")
    assert cmd[sep + 1 :] == ["py", "-m", "vspeech", "--config", "cfg.toml"]
    assert "--native" not in cmd


def test_build_pyspy_cmd_native_flag():
    cmd = sc.build_pyspy_cmd(
        out="o.json", python="py", config="c", duration=5, rate=50, native=True
    )
    assert "--native" in cmd


def test_default_config_is_baseline_fixture():
    cfg = sc.default_config()
    assert cfg.name == "baseline_startup.toml"
    assert cfg.parent.name == "fixtures"
    assert cfg.exists()


def test_sweep_fixtures_cover_baseline_plus_six_workers():
    fixtures = sc.sweep_fixtures()
    labels = [label for label, _ in fixtures]
    # baseline first (the reference floor), then the six non-vc workers; vc is
    # intentionally excluded.
    assert labels == [
        "baseline",
        "recording",
        "transcription",
        "subtitle",
        "translation",
        "tts",
        "playback",
    ]
    assert "vc" not in labels
    # every bundled fixture must exist on disk
    for _label, path in fixtures:
        assert path.exists(), path


def test_dominant_bucket_picks_largest_active_bucket():
    row = {
        "active": 3.0,
        "blocking-io": 2.0,
        "import": 1.0,
        "compute": 0.0,
        "other": 0.0,
    }
    assert sc._dominant_bucket(row) == "blocking-io"


def test_dominant_bucket_dash_when_no_active_time():
    row = {
        "active": 0.0,
        "blocking-io": 0.0,
        "import": 0.0,
        "compute": 0.0,
        "other": 0.0,
    }
    assert sc._dominant_bucket(row) == "-"


def test_render_comparison_sorts_by_active_and_lists_configs():
    rows = [
        {
            "label": "baseline",
            "active": 0.9,
            "unit": "seconds",
            "blocking-io": 0.0,
            "import": 0.7,
            "compute": 0.0,
            "other": 0.2,
        },
        {
            "label": "translation",
            "active": 4.1,
            "unit": "seconds",
            "blocking-io": 1.6,
            "import": 1.9,
            "compute": 0.0,
            "other": 0.6,
        },
    ]
    out = sc.render_comparison(rows)
    assert "translation" in out and "baseline" in out
    # heaviest config (translation) is listed before the lighter baseline
    assert out.index("translation") < out.index("baseline")


def test_render_comparison_handles_empty():
    assert "no profiles" in sc.render_comparison([])
