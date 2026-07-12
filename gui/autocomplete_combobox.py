"""A searchable dropdown: an entry with a focus-preserving popup listbox that
filters its candidates by a case-insensitive substring, live, as you type.

Unlike a native ttk combobox (whose dropdown grabs keyboard focus, so you can't
keep typing to narrow an open list), the popup here never takes focus — the
entry keeps it — so the list filters as you type while staying open.

Drop-in replacement for the old prefix-autocomplete combobox: same public
interface (`set_completion_list` / `get_value` / `get_label_for_item_value` /
`set` / `get`) and it fires a `<<ComboboxSelected>>` virtual event on a pick,
so `gui/form.py` needs no changes.
"""

import tkinter
from tkinter import END
from tkinter import Event
from tkinter import Listbox
from tkinter import Toplevel
from typing import Any

import ttkbootstrap as ttk

_MAX_VISIBLE_ROWS = 8


class AutocompleteCombobox[T](ttk.Entry):
    _label_value_map: dict[str, T]
    _completion_list: list[str]

    def __init__(self, master: Any = None, **kwargs: Any):
        super().__init__(master, **kwargs)
        self._label_value_map = {}
        self._completion_list = []
        self._popup: Toplevel | None = None
        self._listbox: Listbox | None = None
        self._root_click_bind: str | None = None
        self._reposition_job: str | None = None
        self._last_pos: tuple[int, int] | None = None

    # --- public interface (drop-in for the old combobox) ----------------

    def get_value(self) -> T | None:
        # None when the current text isn't a known label (blank, or a partial
        # filter the user didn't resolve to a pick). Callers treat None as "no
        # selection" rather than an error.
        return self._label_value_map.get(self.get())

    def get_label_for_item_value(self, value: T) -> str | None:
        for label, _value in self._label_value_map.items():
            if _value == value:
                return label

    def set(self, label: Any) -> None:
        self.delete(0, END)
        self.insert(0, "" if label is None else str(label))

    def set_completion_list(self, label_value_map: dict[str, T]) -> None:
        self._label_value_map = label_value_map.copy()
        self._completion_list = sorted(label_value_map.keys(), key=str.lower)
        self.bind("<KeyRelease>", self._on_keyrelease)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Down>", self._on_down)
        self.bind("<Up>", self._on_up)
        self.bind("<Return>", self._on_return)
        self.bind("<Escape>", self._on_escape)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Destroy>", lambda _e: self._close())

    # --- filtering ------------------------------------------------------

    def _matches(self) -> list[str]:
        needle = self.get().strip().lower()
        if not needle:
            return self._completion_list
        return [item for item in self._completion_list if needle in item.lower()]

    # --- popup lifecycle ------------------------------------------------

    def _open(self) -> None:
        matches = self._matches()
        if not matches:
            self._close()
            return
        if self._popup is None:
            self._popup = Toplevel(self)
            self._popup.wm_overrideredirect(True)
            self._listbox = Listbox(
                self._popup, activestyle="none", exportselection=False
            )
            self._listbox.pack(fill="both", expand=True)
            self._listbox.bind("<ButtonRelease-1>", self._on_listbox_click)
            self._listbox.bind("<Motion>", self._on_listbox_motion)
            self._root_click_bind = self.winfo_toplevel().bind(
                "<Button-1>", self._on_global_click, add="+"
            )
        self._populate(matches)
        self._place_popup(len(matches))
        if self._reposition_job is None:
            self._reposition_job = self.after(40, self._reposition)

    def _reposition(self) -> None:
        # Keep the popup glued to the entry so it follows when the form is
        # scrolled (the popup is a separate top-level at a fixed screen point).
        self._reposition_job = None
        if self._popup is None:
            return
        # If the entry has scrolled out of its visible viewport (clipped by the
        # ScrolledFrame) or off-window, its own centre point is no longer the
        # entry — hide the popup instead of trailing it off past the window edge.
        centre_x = self.winfo_rootx() + self.winfo_width() // 2
        centre_y = self.winfo_rooty() + self.winfo_height() // 2
        if self.winfo_containing(centre_x, centre_y) is not self:
            self._close()
            return
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        if (x, y) != self._last_pos:
            self._last_pos = (x, y)
            width = self.winfo_width()
            height = self._popup.winfo_height() or self._popup.winfo_reqheight()
            self._popup.wm_geometry(f"{width}x{height}+{x}+{y}")
        self._reposition_job = self.after(40, self._reposition)

    def _populate(self, matches: list[str]) -> None:
        lb = self._listbox
        if lb is None:
            return
        lb.delete(0, END)
        for item in matches:
            lb.insert(END, item)
        lb.selection_clear(0, END)
        lb.selection_set(0)
        lb.activate(0)

    def _place_popup(self, count: int) -> None:
        if self._popup is None or self._listbox is None:
            return
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = self.winfo_width()
        self._listbox.configure(height=min(count, _MAX_VISIBLE_ROWS))
        self._popup.update_idletasks()
        height = self._popup.winfo_reqheight()
        self._popup.wm_geometry(f"{width}x{height}+{x}+{y}")
        self._popup.lift()

    def _close(self) -> None:
        if self._reposition_job is not None:
            self.after_cancel(self._reposition_job)
            self._reposition_job = None
        self._last_pos = None
        if self._root_click_bind is not None:
            try:
                self.winfo_toplevel().unbind("<Button-1>", self._root_click_bind)
            except tkinter.TclError:
                pass
            self._root_click_bind = None
        if self._popup is not None:
            try:
                self._popup.destroy()
            except tkinter.TclError:
                pass
            self._popup = None
            self._listbox = None

    # --- selection ------------------------------------------------------

    def _commit(self, label: str) -> None:
        self.set(label)
        self._close()
        self.focus_set()
        self.icursor(END)
        self.event_generate("<<ComboboxSelected>>")

    def _highlighted_label(self) -> str | None:
        lb = self._listbox
        if lb is None:
            return None
        selection = lb.curselection()
        return lb.get(selection[0]) if selection else None

    def _move(self, delta: int) -> None:
        lb = self._listbox
        if lb is None or lb.size() == 0:
            return
        selection = lb.curselection()
        current = selection[0] if selection else 0
        nxt = max(0, min(lb.size() - 1, current + delta))
        lb.selection_clear(0, END)
        lb.selection_set(nxt)
        lb.activate(nxt)
        lb.see(nxt)

    # --- event handlers -------------------------------------------------

    def _on_keyrelease(self, event: Event[Any]) -> None:
        if event.keysym in ("Up", "Down", "Return", "Escape", "Left", "Right", "Tab"):
            return
        # Re-filter and (re)open; the entry keeps focus so typing continues to
        # narrow the still-open list.
        self._open()

    def _on_click(self, _event: Event[Any]) -> None:
        # Click anywhere in the field opens the (filtered) list. after() lets Tk
        # finish placing the cursor first.
        self.after(1, self._open)

    def _on_down(self, _event: Event[Any]) -> str:
        if self._popup is None:
            self._open()
        else:
            self._move(1)
        return "break"

    def _on_up(self, _event: Event[Any]) -> str:
        if self._popup is not None:
            self._move(-1)
        return "break"

    def _on_return(self, _event: Event[Any]) -> str:
        label = self._highlighted_label()
        if label is not None:
            self._commit(label)
        return "break"

    def _on_escape(self, _event: Event[Any]) -> str:
        self._close()
        return "break"

    def _on_listbox_click(self, event: Event[Any]) -> str:
        lb = self._listbox
        if lb is not None:
            self._commit(lb.get(lb.nearest(event.y)))
        return "break"

    def _on_listbox_motion(self, event: Event[Any]) -> None:
        lb = self._listbox
        if lb is None:
            return
        index = lb.nearest(event.y)
        lb.selection_clear(0, END)
        lb.selection_set(index)
        lb.activate(index)

    def _on_global_click(self, event: Event[Any]) -> None:
        # Close when clicking outside the entry and its popup listbox.
        if event.widget is self or event.widget is self._listbox:
            return
        self._close()

    def _on_focus_out(self, _event: Event[Any]) -> None:
        self.after(1, self._close_if_focus_left)

    def _close_if_focus_left(self) -> None:
        if self._popup is None:
            return
        focused = self.focus_get()
        if focused is not self and focused is not self._listbox:
            self._close()
