"""entry point をプロセスとして起動する検証。

CliRunner は出力を UTF-8 の BytesIO に受けるので、Windows の cp932/cp1252 な
stdout でだけ出る UnicodeEncodeError を再現できない。help もエラーも日本語な
以上、そこを踏むと `vsctl --help` すら通らないので、実プロセスで確かめる。
"""

import os
import subprocess
import sys

import pytest

# cp1252 は日本語を 1 文字も表現できない。Windows の既定 (cp932) より厳しい
# 条件を作って、再エンコード漏れがあれば必ず落ちるようにする。
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
    # 使い方エラー (宛先なし) は日本語混じりで stderr に出る。
    result = run(["ping"], {**NARROW_ENV, "VSPEECH_TARGET": ""})
    assert result.returncode == 2
    assert "UnicodeEncodeError" not in result.stderr
