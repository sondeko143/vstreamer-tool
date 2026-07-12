from tkinter import StringVar
from tkinter import W
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Radiobutton
from ttkbootstrap import Toplevel

from gui.recipes import RECIPES


class RecipeDialog(Toplevel):
    """Modal recipe picker: a clickable radio list of the available recipes.

    After the constructor returns (it blocks via `wait_window`), `result` holds
    the chosen recipe key, or None if the user cancelled or closed the dialog.
    Replaces the old free-text `Querybox.get_string` where a typo silently fell
    through to the "blank" recipe.
    """

    def __init__(self, master: Any):
        super().__init__(master)
        self.title("new pipeline")
        self.result: str | None = None
        self.transient(master)
        self.resizable(False, False)

        Label(self, text="レシピを選んでください:").pack(
            anchor=W, padx=12, pady=(12, 4)
        )
        self._choice = StringVar(value=RECIPES[0].key)
        for recipe in RECIPES:
            Radiobutton(
                self,
                text=recipe.label,
                value=recipe.key,
                variable=self._choice,
            ).pack(anchor=W, padx=16, pady=2)

        buttons = Frame(self)
        buttons.pack(anchor="e", padx=12, pady=12)
        Button(buttons, text="OK", command=self._ok, bootstyle="primary").pack(
            side="left", padx=4
        )
        Button(
            buttons, text="Cancel", command=self._cancel, bootstyle="secondary"
        ).pack(side="left", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.wait_window()

    def _ok(self) -> None:
        self.result = self._choice.get()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()
