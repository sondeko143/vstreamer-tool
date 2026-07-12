from tkinter import END
from typing import Any

import toml
from pydantic import ValidationError
from ttkbootstrap import Frame

from gui.migration import CONFIG_MIGRATIONS
from gui.migration import CURRENT_CONFIG_VERSION
from gui.migration import migrate_dict
from gui.widgets import ScrolledText
from vspeech.config import Config


class RawTomlEditor(Frame):
    def __init__(self, master: Any):
        super().__init__(master)
        self.text = ScrolledText(self)
        self.text.pack(fill="both", expand=True)

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", END)
        self.text.insert("1.0", text)

    def set_config(self, config: Config) -> None:
        self.set_text(config.export_to_toml())

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def parse(self) -> tuple[Config | None, str | None]:
        try:
            data, _ = migrate_dict(
                toml.loads(self.get_text()),
                from_version=0,
                migrations=CONFIG_MIGRATIONS,
                current=CURRENT_CONFIG_VERSION,
            )
            return Config.model_validate(data), None
        except (toml.TomlDecodeError, ValidationError) as e:
            return None, str(e)
