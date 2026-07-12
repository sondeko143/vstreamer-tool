"""A combobox whose dropdown filters by a case-insensitive SUBSTRING of what is
typed (not just a prefix), and that opens on a click anywhere in the field, not
only on the small arrow.

Note on behaviour: ttk's native dropdown takes keyboard focus while it is open,
so you cannot keep typing to narrow a *currently-open* list. The flow is: type
to filter the candidates (the entry keeps focus), then click (anywhere) to open
the filtered list and pick one.
"""

import tkinter
from tkinter import Event
from typing import Any

import ttkbootstrap as ttk


class AutocompleteCombobox[T](ttk.Combobox):
    _completion_list: list[str]
    _label_value_map: dict[str, T]

    def get_value(self) -> T | None:
        # None when the current text isn't a known label (blank, or a partial
        # filter the user didn't resolve to a pick). Callers treat None as "no
        # selection" rather than an error.
        return self._label_value_map.get(self.get())

    def get_label_for_item_value(self, value: T) -> str | None:
        for label, _value in self._label_value_map.items():
            if _value == value:
                return label

    def set_completion_list(self, label_value_map: dict[str, T]) -> None:
        self._label_value_map = label_value_map.copy()
        self._completion_list = sorted(label_value_map.keys(), key=str.lower)
        self["values"] = self._completion_list
        self.bind("<KeyRelease>", self._on_keyrelease)
        self.bind("<Button-1>", self._on_click, add="+")

    def _matches(self, text: str) -> list[str]:
        needle = text.strip().lower()
        if not needle:
            return self._completion_list
        return [item for item in self._completion_list if needle in item.lower()]

    def _post(self) -> None:
        """Open the dropdown (the same action as clicking the arrow)."""
        try:
            self.tk.call("ttk::combobox::Post", self)
        except tkinter.TclError:
            pass

    def _on_keyrelease(self, event: Event[Any]) -> None:
        # Navigation keys belong to the entry cursor / open dropdown — don't
        # treat them as filter input.
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab", "Left", "Right"):
            return
        # Narrow the candidates by substring; the entry keeps focus so the user
        # can keep typing.
        self["values"] = self._matches(self.get())

    def _on_click(self, _event: Event[Any]) -> None:
        # Click anywhere in the field opens the (filtered) dropdown, not just the
        # arrow. Post after Tk finishes the click that places the cursor.
        self["values"] = self._matches(self.get())
        self.after(1, self._post)
