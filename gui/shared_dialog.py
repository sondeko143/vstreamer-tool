from pathlib import Path
from tkinter import W
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Toplevel

from gui.config_paths import get_value
from gui.config_paths import set_value
from gui.paths import ProfilePaths
from gui.profile import save_default_config
from gui.shared_paths import SHARED_ASSET_FIELDS
from gui.widgets import Textbox
from vspeech.config import Config


class SharedPathsDialog(Toplevel):
    """マシン共通の素材パスを default.toml 上で一度だけ編集する (ADR-0046)。

    `propagate_requested` が True で閉じたら、呼び出し側が既存の全 pipeline へ
    反映する。`propagate` という名前は tkinter の `Misc.propagate`
    (= `pack_propagate` の別名) と衝突するので避けている。
    """

    def __init__(self, master: Any, paths: ProfilePaths, default_config: Config):
        super().__init__(master)
        self.title("共有 (既定) の素材パス")
        self.transient(master)
        self.paths = paths
        self.default_config = default_config
        self.propagate_requested = False
        self.saved = False

        Label(
            self,
            text="マシンに 1 セットしかない資産のパス。ここで編集すると新規 pipeline が継承します。",
        ).pack(anchor=W, padx=12, pady=(12, 6))

        self.entries: dict[str, Textbox] = {}
        for path in SHARED_ASSET_FIELDS:
            row = Frame(self)
            row.pack(fill="x", padx=12, pady=2)
            Label(row, text=path, width=28).pack(side="left", anchor=W)
            entry = Textbox(row)
            entry.set(get_value(default_config, path))
            entry.pack(side="left", fill="x", expand=True)
            self.entries[path] = entry

        buttons = Frame(self)
        buttons.pack(anchor="e", padx=12, pady=12)
        Button(buttons, text="保存", command=self._save, bootstyle="primary").pack(
            side="left", padx=4
        )
        Button(
            buttons,
            text="保存して全 pipeline へ反映",
            command=self._save_and_propagate,
            bootstyle="warning",
        ).pack(side="left", padx=4)
        Button(
            buttons, text="Cancel", command=self.destroy, bootstyle="secondary"
        ).pack(side="left", padx=4)

        self.grab_set()
        self.wait_window()

    def _write_back(self) -> None:
        for path, entry in self.entries.items():
            text = entry.get_value().strip()
            # 空欄は「未設定」。Optional (voicevox.onnxruntime_path) は None、
            # それ以外の Path は Path() 番兵に戻す — preflight がそれを不在と
            # して報告する。
            if not text:
                set_value(
                    self.default_config,
                    path,
                    None if path == "voicevox.onnxruntime_path" else Path(),
                )
            else:
                set_value(self.default_config, path, text)
        save_default_config(self.paths, self.default_config)
        self.saved = True

    def _save(self) -> None:
        self._write_back()
        self.destroy()

    def _save_and_propagate(self) -> None:
        self._write_back()
        self.propagate_requested = True
        self.destroy()
