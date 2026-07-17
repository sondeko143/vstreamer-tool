from collections.abc import Callable
from tkinter import BOTH
from tkinter import LEFT
from tkinter import W
from tkinter import X
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label

from gui.readiness import Readiness


class ReadinessPanel(Frame):
    """「この pipeline は起動できるか」を起動前に見せるパネル (ADR-0045)。

    行の中身は vspeech.preflight が出した問題そのもので、この widget は
    判断をしない — 描くだけ。
    """

    def __init__(self, master: Any, on_fix: Callable[[str], None]):
        super().__init__(master, padding=(0, 0, 0, 6))
        self.on_fix = on_fix
        self.flow = Label(self, text="", bootstyle="secondary")
        self.flow.pack(fill=X, anchor=W)
        self.rows = Frame(self)
        self.rows.pack(fill=BOTH, expand=True)

    def clear(self) -> None:
        self.flow.configure(text="")
        for child in list(self.rows.children.values()):
            child.destroy()

    def show(self, readiness: Readiness) -> None:
        self.clear()
        self.flow.configure(text=self._flow_text(readiness))
        if readiness.error is not None:
            Label(self.rows, text=f"⚠ {readiness.error}", bootstyle="warning").pack(
                fill=X, anchor=W
            )
            return
        if not readiness.workers:
            Label(
                self.rows,
                text="有効な worker がありません (この pipeline は何もしません)",
                bootstyle="warning",
            ).pack(fill=X, anchor=W)
            return
        for worker in readiness.workers:
            if worker.ok:
                Label(self.rows, text=f"✓ {worker.worker}", bootstyle="success").pack(
                    fill=X, anchor=W
                )
                continue
            for problem in worker.problems:
                self._problem_row(worker.worker, problem.detail, problem.field)

    def _problem_row(self, worker_name: str, detail: str, field: str | None) -> None:
        row = Frame(self.rows)
        row.pack(fill=X, anchor=W)
        Label(row, text=f"✗ {worker_name}  {detail}", bootstyle="danger").pack(
            side=LEFT, anchor=W
        )
        if field is not None:
            Button(
                row,
                text="→修正",
                bootstyle="link-danger",
                command=lambda: self.on_fix(field),
            ).pack(side=LEFT)

    def _flow_text(self, readiness: Readiness) -> str:
        if not readiness.flow:
            return ""
        return "  |  ".join(" → ".join(chain) for chain in readiness.flow)
