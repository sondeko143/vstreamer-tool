"""Verification that launches the entry point as a process.

CliRunner receives output into a UTF-8 BytesIO, so it cannot reproduce the
UnicodeEncodeError that only appears on a Windows cp932/cp1252 stdout. Since both the help
and the errors are in Japanese, stepping on that breaks even `vsctl --help`, so it is
checked in a real process.
"""

import os
import subprocess
import sys

import pytest

# cp1252 cannot represent a single Japanese character. This creates a condition stricter
# than the Windows default (cp932) so that any missed re-encoding always fails.
NARROW_ENV = {"PYTHONIOENCODING": "cp1252"}


def run(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("args", [["--help"], ["reload", "--help"]])
def test_help_survives_a_narrow_stdout_encoding(args):
    result = run(args, NARROW_ENV)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


def test_usage_error_survives_a_narrow_stdout_encoding():
    # A usage error (no target) goes to stderr with Japanese mixed in.
    result = run(["ping"], {**NARROW_ENV, "VSPEECH_TARGET": ""})
    assert result.returncode == 2
    assert "UnicodeEncodeError" not in result.stderr
