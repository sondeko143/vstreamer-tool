from logging import Handler
from logging import LogRecord
from tkinter import END
from tkinter import BooleanVar
from tkinter import StringVar
from typing import Any

from ttkbootstrap import Checkbutton as ttkCheckbutton
from ttkbootstrap import Entry
from ttkbootstrap import Spinbox as ttkSpinbox
from ttkbootstrap import Text
from ttkbootstrap.widgets.scrolled import ScrolledText as ttkScrolledText


class ScrolledText(ttkScrolledText):
    pass


class TextHandler(Handler):
    """Log to a Tkinter Text/ScrolledText widget from any thread."""

    def __init__(self, text: Text | ScrolledText):
        Handler.__init__(self)
        self.text = text

    def emit(self, record: LogRecord):
        msg = self.format(record)

        def append():
            self.text.configure(state="normal")
            self.text.insert(END, msg + "\n")
            self.text.configure(state="disabled")
            self.text.yview(END)

        self.text.after(0, append)


class Spinbox(ttkSpinbox):
    def get_value(self) -> str:
        return super().get()

    def set(self, value: Any):
        self.delete(0, END)
        self.insert(0, str(value))


class Checkbutton(ttkCheckbutton):
    var: BooleanVar

    def __init__(self, master: Any, **kw: Any):
        self.var = BooleanVar()
        super().__init__(master, variable=self.var, onvalue=True, offvalue=False, **kw)

    def get_value(self) -> bool:
        return self.var.get()

    def set(self, value: Any):
        self.var.set(bool(value))


class Textbox(Entry):
    var: StringVar

    def __init__(self, master: Any, **kw: Any):
        self.var = StringVar()
        super().__init__(master, textvariable=self.var, **kw)

    def get_value(self) -> str:
        return self.var.get()

    def set(self, value: Any):
        self.var.set("" if value is None else str(value))
