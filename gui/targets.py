"""操作対象 (すでに走っている pipeline) の宛先一覧とその永続化。

このパネルは pipeline を起動しないので、ここに保存するのは「どこへ繋ぐか」
だけ — Config そのものは一切持たない。壊れたファイルは退避してから空の一覧
で起動する (ADR-0034): 起動できないより、宛先を入れ直せる方がまだよい。
"""

from datetime import datetime
from pathlib import Path

import toml
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from gui.paths import GuiPaths
from vspeech.logger import logger

CURRENT_VERSION = 1

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class Target(BaseModel):
    name: str = "new target"
    host: str = DEFAULT_HOST
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    config_path: str = ""
    """reload で読ませる config のパス。**対象マシン上の**パスであることに注意 —
    reload を受けた側が自分で open する (vspeech.lib.command.process_command)。
    こちらのファイルシステムでは解決しないので存在検査もしない。"""

    @property
    def address(self) -> str:
        # host が IPv6 リテラル (":" を含む) のときは grpc の target 記法に合わせて
        # 角括弧で囲む。囲まないと最初の ":" が port 区切りとして読まれる。
        host = self.host.strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"{host}:{self.port}"

    @property
    def label(self) -> str:
        return f"{self.name}  {self.address}"


class TargetList(BaseModel):
    version: int = CURRENT_VERSION
    targets: list[Target] = Field(default_factory=list)


def quarantine(path: Path) -> Path:
    """壊れたファイルを日時付きで退避し、退避先を返す。上書き削除はしない。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f"{path.suffix}.{stamp}.bak")
    backup.write_bytes(path.read_bytes())
    return backup


def load_targets(paths: GuiPaths) -> TargetList:
    path = paths.targets
    if not path.exists():
        return TargetList()
    text = path.read_text(encoding="utf-8")
    try:
        return TargetList.model_validate(toml.loads(text))
    except (toml.TomlDecodeError, ValidationError) as e:
        backup = quarantine(path)
        logger.warning("targets.toml 破損: %s に退避し空の一覧へ (%s)", backup, e)
        return TargetList()


def save_targets(paths: GuiPaths, targets: TargetList) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.targets.write_text(toml.dumps(targets.model_dump()), encoding="utf-8")
