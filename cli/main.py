"""走っている vspeech pipeline へ制御コマンドを送る CLI (ADR-0061)。

`vsctl <操作> --to <host:port>` の 1 発ずつ。宛先は保存しない — 環境変数
`VSPEECH_TARGET` を既定値として読むので、繰り返し叩くときはそちらへ置く。

終了コードは操作の成否そのもの (0 = 相手が受け取った / 1 = 失敗) なので、
スクリプトから `&&` で繋げられる。
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
    """`host:port` を検証して返す。

    port の付け忘れをここで弾く。gRPC は port 無しの target をそのまま受けて
    名前解決に失敗するまで待つので、通すと「deadline まで無反応」になる。
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
    # IPv6 リテラルは角括弧が要る。無いと最初の ":" が port 区切りとして読まれ、
    # `::1` が host "::" port 1 として通ってしまう。
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        raise click.BadParameter(f"IPv6 は [..] で囲んでください: {address!r}")
    return f"{host}:{port}"


def target_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """全サブコマンド共通の宛先オプション。"""
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
    """結果を 1 行で出し、失敗なら終了コード 1 で抜ける。"""
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
    # 空文字は受け側の validation で弾かれるだけで理由が分かりにくいので、
    # click 側で先に落とす (required=True では空文字列を通してしまう)。
    if not config_path.strip():
        raise click.BadParameter("--config-path が空です")
    run(address, EventType.reload, timeout, config_path=config_path.strip())


def main() -> None:
    """entry point (`vsctl`)。

    click に何か出させる前に stdout/stderr を UTF-8 へ差し替える。Windows の
    既定は cp932/cp1252 で、この CLI の help もエラーも日本語なので、素のままだと
    `vsctl --help` が UnicodeEncodeError で落ちる (vspeech.logger と同じ罠)。
    backslashreplace で「読める UTF-8 を出す」かつ「絶対に落ちない」。
    """
    for stream in (stdout, stderr):
        try:
            stream.reconfigure(  # ty: ignore[unresolved-attribute]
                encoding="utf-8", errors="backslashreplace"
            )
        except AttributeError, ValueError:
            pass
    cmd()
