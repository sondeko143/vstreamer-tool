"""マシン共通の素材パス (ADR-0046)。

マシンに 1 セットしかない重い資産のパス。default.toml で一度編集し、明示
操作で既存の全 pipeline へ propagate する。pipeline config は自己完結のまま
なので (ADR-0032)、値は各 pipeline へ実際に書き込む — 起動時合成はしない。
"""

from typing import Any

from gui.config_paths import get_value
from gui.config_paths import set_value
from vspeech.config import Config

SHARED_ASSET_FIELDS: tuple[str, ...] = (
    "rvc.model_file",
    "rvc.hubert_model_file",
    "rvc.rmvpe_model_file",
    "voicevox.openjtalk_dir",
    "voicevox.model_dir",
    "voicevox.onnxruntime_path",
)


def shared_values(config: Config) -> dict[str, Any]:
    return {path: get_value(config, path) for path in SHARED_ASSET_FIELDS}


def apply_shared(source: Config, target: Config) -> list[str]:
    """`source` の共有素材パスを `target` へ写し、実際に変わった path を返す。

    共有指定のフィールドだけ触る。pipeline 固有の調整つまみは保つ。
    """
    changed: list[str] = []
    for path in SHARED_ASSET_FIELDS:
        value = get_value(source, path)
        if get_value(target, path) == value:
            continue
        set_value(target, path, value)
        changed.append(path)
    return changed
