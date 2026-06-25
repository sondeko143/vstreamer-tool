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


def test_default_config_is_bundled_fixture():
    cfg = sc.default_config()
    assert cfg.name == "minimal_startup.toml"
    assert cfg.parent.name == "fixtures"
    assert cfg.exists()
