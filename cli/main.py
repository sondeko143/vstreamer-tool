"""The CLI that sends control commands to a running vspeech pipeline (ADR-0061).

One shot per invocation: `vsctl <operation> --to <host:port>`. The target is never
stored -- the environment variable `VSPEECH_TARGET` is read as the default, so put it
there when invoking repeatedly.

The exit code is the outcome of the operation itself (0 = the peer accepted it,
1 = failure), so it chains with `&&` from a script.

Note: the docstrings of the click group and its commands are the `--help` text the user
reads, so they stay in Japanese; every other docstring and comment here is English.
"""

from collections.abc import Callable
from sys import stderr
from sys import stdout
from typing import Any

import click

from cli.client import DEFAULT_TIMEOUT
from cli.client import SendResult
from cli.client import send
from vspeech.config import EventType

TARGET_ENVVAR = "VSPEECH_TARGET"


def normalize_address(address: str) -> str:
    """Validate `host:port` and return it.

    A forgotten port is rejected here. gRPC accepts a target with no port as-is and waits
    until name resolution fails, so letting it through means "no response until the
    deadline".
    """
    text = address.strip()
    host, separator, port_text = text.rpartition(":")
    if not separator or not host:
        raise click.BadParameter(f"host:port の形で指定してください: {address!r}")
    try:
        port = int(port_text)
    except ValueError:
        raise click.BadParameter(f"port が数値ではありません: {port_text!r}") from None
    if not 1 <= port <= 65535:
        raise click.BadParameter(f"port が範囲外です: {port}")
    # An IPv6 literal needs square brackets. Without them the first ":" is read as the
    # port separator and `::1` would pass as host "::" with port 1.
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        raise click.BadParameter(f"IPv6 は [..] で囲んでください: {address!r}")
    return f"{host}:{port}"


def target_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """The destination options shared by every subcommand."""
    command = click.option(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        show_default=True,
        help="RPC の deadline (秒)。",
    )(command)
    return click.option(
        "--to",
        "address",
        envvar=TARGET_ENVVAR,
        required=True,
        metavar="HOST:PORT",
        help=f"宛先 pipeline の listen アドレス (環境変数 {TARGET_ENVVAR} が既定値)。",
    )(command)


def report(address: str, event: EventType, result: SendResult) -> None:
    """Print the result on one line and exit with code 1 on failure."""
    line = (
        f"{'OK' if result.ok else 'NG'}  {address}  "
        f"{event.value}  {result.elapsed_ms:.0f}ms"
    )
    if result.detail:
        line = f"{line}  {result.detail}"
    click.echo(line, err=not result.ok)
    if not result.ok:
        raise SystemExit(1)


def run(address: str, event: EventType, timeout: float, config_path: str = "") -> None:
    target = normalize_address(address)
    report(target, event, send(target, event, config_path=config_path, timeout=timeout))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cmd() -> None:
    """走っている vspeech pipeline へ制御コマンドを送る。

    pipeline の起動・設定編集はしない。起動は `vspeech --config <file>`。
    """


@cmd.command()
@target_options
def ping(address: str, timeout: float) -> None:
    """疎通確認。相手が Command を処理して応答するところまで見る。"""
    run(address, EventType.ping, timeout)


@cmd.command()
@target_options
def pause(address: str, timeout: float) -> None:
    """pipeline を一時停止する (全 worker のゲートを閉じる)。"""
    run(address, EventType.pause, timeout)


@cmd.command()
@target_options
def resume(address: str, timeout: float) -> None:
    """一時停止中の pipeline を再開する。"""
    run(address, EventType.resume, timeout)


@cmd.command()
@target_options
@click.option(
    "--config-path",
    required=True,
    metavar="PATH",
    help="読み直させる config のパス。**対象マシン上の**パスで、受け側が自分で開く。",
)
def reload(address: str, timeout: float, config_path: str) -> None:
    """config を読み直させる。

    パスは対象マシンのファイルシステム上で解決される。こちらに同じファイルが
    あるかは無関係なので、存在確認もしない。
    """
    # An empty string would only be rejected by the peer's validation, where the reason is
    # hard to see, so reject it earlier in click (required=True lets an empty string
    # through).
    if not config_path.strip():
        raise click.BadParameter("--config-path が空です")
    run(address, EventType.reload, timeout, config_path=config_path.strip())


def main() -> None:
    """The entry point (`vsctl`).

    Switches stdout/stderr to UTF-8 before letting click print anything. The Windows
    default is cp932/cp1252, and this CLI's help and errors are in Japanese, so left as-is
    `vsctl --help` dies with UnicodeEncodeError (the same trap as vspeech.logger).
    backslashreplace gives both "readable UTF-8 out" and "never crashes".
    """
    for stream in (stdout, stderr):
        try:
            stream.reconfigure(  # ty: ignore[unresolved-attribute]
                encoding="utf-8", errors="backslashreplace"
            )
        except AttributeError, ValueError:
            pass
    cmd()
