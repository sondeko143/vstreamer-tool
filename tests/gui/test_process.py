import sys
from pathlib import Path

from gui.process import build_argv
from gui.process import build_text_command


def test_build_argv():
    argv = build_argv(Path("/x/pipe.toml"))
    assert argv[:3] == [sys.executable, "-m", "vspeech"]
    assert argv[-2:] == ["--config", str(Path("/x/pipe.toml"))]


def test_build_text_command_trims_and_chains():
    command = build_text_command("  hi  ", [["tts", "playback"]])
    assert command.operand.text == "hi"
    assert len(command.chains) == 1
    assert len(command.chains[0].operations) == 2


def test_build_text_command_drops_empty_chains():
    command = build_text_command("x", [["tts"], []])
    assert len(command.chains) == 1


def test_build_text_command_drops_chain_left_empty_by_op_filter():
    command = build_text_command("x", [[""]])
    assert len(command.chains) == 0
