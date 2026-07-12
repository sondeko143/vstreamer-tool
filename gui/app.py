from pathlib import Path
from tkinter import BOTH
from tkinter import END
from tkinter import LEFT
from tkinter import RIGHT
from tkinter import Listbox
from tkinter import Y
from typing import Any
from uuid import uuid4

import click
from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Window
from ttkbootstrap.dialogs import Querybox
from ttkbootstrap.themes.standard import STANDARD_THEMES

from gui.migration import quarantine
from gui.paths import resolve_paths
from gui.pipeline_editor import PipelineEditor
from gui.ports import allocate_free_port
from gui.profile import PipelineEntry
from gui.profile import load_default_config
from gui.profile import load_profile
from gui.profile import save_pipeline_config
from gui.profile import save_profile
from gui.recipes import RECIPES
from gui.recipes import RECIPES_BY_KEY
from vspeech.logger import logger


class App(Frame):
    def __init__(self, master: Any, profile_dir: Path | None):
        super().__init__(master)
        self.pack(fill=BOTH, expand=True)
        self.paths = resolve_paths(profile_dir)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.default_config = load_default_config(self.paths)
        self.profile = load_profile(self.paths)

        left = Frame(self)
        left.pack(side=LEFT, fill=Y)
        self.listbox = Listbox(left, width=32)
        self.listbox.pack(fill=Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        Button(left, text="+ new", command=self.new_pipeline).pack(fill="x")
        Button(left, text="del", command=self.delete_pipeline).pack(fill="x")

        self.editor = PipelineEditor(self, self.paths, on_dirty=lambda: None)
        self.editor.pack(side=RIGHT, fill=BOTH, expand=True)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, END)
        for entry in self.profile.pipelines:
            self.listbox.insert(END, f"{entry.name}  :{entry.port}")

    def _on_select(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.profile.pipelines[selection[0]]
        self.editor.load_entry(entry)

    def new_pipeline(self) -> None:
        labels = [recipe.label for recipe in RECIPES]
        chosen = Querybox.get_string(
            prompt="レシピを選んでください:\n" + "\n".join(labels),
            title="new pipeline",
        )
        recipe = next(
            (r for r in RECIPES if r.label == chosen or r.key == chosen),
            RECIPES_BY_KEY["blank"],
        )
        claimed = {entry.port for entry in self.profile.pipelines}
        port = allocate_free_port(claimed)
        entry = PipelineEntry(
            id=uuid4().hex[:8],
            name=recipe.label,
            port=port,
            recipe=recipe.key,
        )
        config = recipe.apply(self.default_config)
        save_pipeline_config(self.paths, entry, config)
        self.profile.pipelines.append(entry)
        save_profile(self.paths, self.profile)
        self._refresh_list()

    def delete_pipeline(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.profile.pipelines.pop(selection[0])
        config_path = self.paths.pipeline_config(entry.id)
        if config_path.exists():
            quarantine(config_path)
            config_path.unlink()
        save_profile(self.paths, self.profile)
        self._refresh_list()
        logger.info("deleted pipeline %s", entry.id)


@click.command()
@click.option(
    "--profile-dir", "profile_dir", type=click.Path(path_type=Path), default=None
)
@click.option(
    "-t", "--theme", default="cosmo", type=click.Choice(list(STANDARD_THEMES.keys()))
)
def main(profile_dir: Path | None, theme: str):
    root = Window(themename=theme)
    root.title("vspeech pipelines")
    root.geometry("900x760")
    App(root, profile_dir)
    root.mainloop()
