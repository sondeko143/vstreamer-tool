"""
tkentrycomplete.py

A tkinter widget that features autocompletion.

Created by Mitja Martini on 2008-11-29.
Updated by Russell Adams, 2011/01/24 to support Python 3 and Combobox.
   Licensed same as original (not specified?), or public domain, whichever is less restrictive.
"""
import tkinter
from tkinter import Event
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar

import ttkbootstrap as ttk

T = TypeVar("T")


class AutocompleteCombobox(ttk.Combobox, Generic[T]):
    _completion_list: List[str]
    _hit_index: int
    _hits: List[str]
    _label_value_map: Dict[str, T]

    def get_value(self) -> T:
        return self._label_value_map[self.get()]

    def get_label_for_item_value(self, value: T) -> Optional[str]:
        for label, _value in self._label_value_map.items():
            if _value == value:
                return label

    def set_completion_list(self, label_value_map: Dict[str, T]):
        """Use our completion list as our drop down selection menu, arrows move through menu."""
        completion_list = list(label_value_map.keys())
        self._completion_list = sorted(
            completion_list, key=str.lower
        )  # Work with a sorted list
        self._hits = []
        self._hit_index = 0
        self.position = 0
        self.bind("<KeyRelease>", self.handle_keyrelease)
        self["values"] = self._completion_list  # Setup our popup menu
        self._label_value_map = label_value_map.copy()

    def autocomplete(self, delta: int = 0):
        """autocomplete the Combobox, delta may be 0/1/-1 to cycle through possible hits"""
        if (
            delta
        ):  # need to delete selection otherwise we would fix the current position
            self.delete(self.position, tkinter.END)
        else:  # set position to end so selection starts where textentry ended
            self.position = len(self.get())
        # collect hits
        _hits = []
        for element in self._completion_list:
            if element.lower().startswith(
                self.get().lower()
            ):  # Match case insensitively
                _hits.append(element)
        # if we have a new hit list, keep this in mind
        if _hits != self._hits:
            self._hit_index = 0
            self._hits = _hits
        # only allow cycling if we are in a known hit list
        if _hits == self._hits and self._hits:
            self._hit_index = (self._hit_index + delta) % len(self._hits)
        # now finally perform the auto completion
        if self._hits:
            self.delete(0, tkinter.END)
            self.insert(0, self._hits[self._hit_index])
            self.select_range(self.position, tkinter.END)

    def handle_keyrelease(self, event: "Event[Any]"):
        """event handler for the keyrelease event on this widget"""
        if event.keysym == "BackSpace":
            self.delete(self.index(tkinter.INSERT), tkinter.END)
            self.position = self.index(tkinter.END)
        if event.keysym == "Left":
            if self.position < self.index(tkinter.END):  # delete the selection
                self.delete(self.position, tkinter.END)
            else:
                self.position = self.position - 1  # delete one character
                self.delete(self.position, tkinter.END)
        if event.keysym == "Right":
            self.position = self.index(tkinter.END)  # go to end (no selection)
        if len(event.keysym) == 1:
            self.autocomplete()
        # No need for up/down, we'll jump to the popup
        # list at the position of the autocompletion
