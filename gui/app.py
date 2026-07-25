"""走っている pipeline へ ping / pause / resume / reload を送るだけの操作パネル。

pipeline の起動・設定編集は持たない (ADR-0060)。ここが知っているのは宛先
(host:port と reload 用の config パス) だけで、pipeline がどう構成されている
かは一切知らない。
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Thread
from tkinter import BOTH
from tkinter import END
from tkinter import LEFT
from tkinter import RIGHT
from tkinter import Listbox
from tkinter import StringVar
from tkinter import TclError
from tkinter import W
from tkinter import X
from tkinter import Y
from typing import Any

import click
from ttkbootstrap import Button
from ttkbootstrap import Entry
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Labelframe
from ttkbootstrap import Window
from ttkbootstrap.themes.standard import STANDARD_THEMES
from ttkbootstrap.widgets.scrolled import ScrolledText

from gui.client import DEFAULT_TIMEOUT
from gui.client import SendResult
from gui.client import send
from gui.paths import resolve_paths
from gui.targets import Target
from gui.targets import load_targets
from gui.targets import save_targets
from vspeech.config import EventType

LOG_MAX_LINES = 500

# 一覧の左端に出す直近の疎通結果。宛先 (host:port) 単位で覚える — 疎通は名前
# ではなくアドレスの性質なので、同じアドレスを指す 2 エントリは同じ印になる。
MARK_UNKNOWN = "・"
MARK_OK = "○"
MARK_NG = "×"

OPERATION_LABELS: list[tuple[str, EventType]] = [
    ("疎通確認", EventType.ping),
    ("pause", EventType.pause),
    ("resume", EventType.resume),
    ("reload", EventType.reload),
]


class App(Frame):
    def __init__(self, master: Any, config_dir: Path | None):
        super().__init__(master)
        self.pack(fill=BOTH, expand=True)
        self.paths = resolve_paths(config_dir)
        self.targets = load_targets(self.paths)
        self.status: dict[str, str] = {}
        self.index: int | None = None
        self.sending = False

        left = Frame(self)
        left.pack(side=LEFT, fill=Y, padx=(8, 4), pady=8)
        Label(left, text="宛先").pack(anchor=W)
        self.listbox = Listbox(left, width=34, exportselection=False)
        self.listbox.pack(fill=Y, expand=True, pady=(2, 4))
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        list_buttons = Frame(left)
        list_buttons.pack(fill=X)
        Button(list_buttons, text="+ 追加", command=self.add_target).pack(
            side=LEFT, expand=True, fill=X, padx=(0, 2)
        )
        Button(list_buttons, text="削除", command=self.delete_target).pack(
            side=LEFT, expand=True, fill=X, padx=(2, 0)
        )

        right = Frame(self)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(4, 8), pady=8)
        self._build_detail(right)
        self._build_operations(right)
        self._build_log(right)

        self._refresh_list()
        if self.targets.targets:
            self._select(0)
        else:
            self._set_detail_enabled(False)
            self._log("宛先がありません。「+ 追加」で登録してください。")

    # --- widgets ---------------------------------------------------------

    def _build_detail(self, master: Any) -> None:
        detail = Labelframe(master, text="宛先の設定")
        detail.pack(fill=X)
        detail.columnconfigure(1, weight=1)
        self.vars: dict[str, StringVar] = {}
        self.entries: list[Entry] = []
        rows = [
            ("name", "名前"),
            ("host", "ホスト"),
            ("port", "ポート"),
            ("config_path", "config パス"),
        ]
        for row, (key, label) in enumerate(rows):
            Label(detail, text=label).grid(
                row=row, column=0, sticky=W, padx=(8, 4), pady=3
            )
            var = StringVar()
            self.vars[key] = var
            entry = Entry(detail, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=3)
            self.entries.append(entry)
        # config パスは「対象マシン上の」パス。reload を受けた側が自分で open
        # するので、こちらに同じファイルがあるかは無関係 — 取り違えやすいので明記する。
        Label(
            detail,
            text="config パスは reload でのみ使用。対象マシン上のパスを入れてください。",
            bootstyle="secondary",
        ).grid(row=len(rows), column=0, columnspan=2, sticky=W, padx=8, pady=(0, 4))
        Button(detail, text="保存", command=self.save_detail).grid(
            row=len(rows) + 1, column=1, sticky="e", padx=8, pady=(0, 8)
        )

    def _build_operations(self, master: Any) -> None:
        operations = Frame(master)
        operations.pack(fill=X, pady=(8, 4))
        self.operation_buttons: list[Button] = []
        for label, event in OPERATION_LABELS:
            button = Button(
                operations,
                text=label,
                command=self._operation_command(event),
                bootstyle="primary" if event == EventType.ping else "secondary",
            )
            button.pack(side=LEFT, expand=True, fill=X, padx=2)
            self.operation_buttons.append(button)

    def _operation_command(self, event: EventType) -> Callable[[], None]:
        return lambda: self.send_operation(event)

    def _build_log(self, master: Any) -> None:
        frame = Labelframe(master, text="結果")
        frame.pack(fill=BOTH, expand=True)
        self.logbox = ScrolledText(frame, height=12, autohide=True, state="disabled")
        self.logbox.pack(fill=BOTH, expand=True, padx=4, pady=4)

    # --- target list -----------------------------------------------------

    def _mark(self, target: Target) -> str:
        return self.status.get(target.address, MARK_UNKNOWN)

    def _refresh_list(self) -> None:
        self.listbox.delete(0, END)
        for target in self.targets.targets:
            self.listbox.insert(END, f"{self._mark(target)} {target.label}")
        if self.index is not None and 0 <= self.index < len(self.targets.targets):
            self.listbox.selection_clear(0, END)
            self.listbox.selection_set(self.index)

    def _on_select(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        self._select(selection[0])

    def _select(self, index: int) -> None:
        if not 0 <= index < len(self.targets.targets):
            self.index = None
            self._set_detail_enabled(False)
            return
        self.index = index
        self.listbox.selection_clear(0, END)
        self.listbox.selection_set(index)
        target = self.targets.targets[index]
        self.vars["name"].set(target.name)
        self.vars["host"].set(target.host)
        self.vars["port"].set(str(target.port))
        self.vars["config_path"].set(target.config_path)
        self._set_detail_enabled(True)

    def _set_detail_enabled(self, enabled: bool) -> None:
        # 選択が無いときは入力欄も閉じる。開けたままだと、どこにも書き戻らない
        # 欄へ打ち込めてしまう (save_detail は選択が無ければ何もしない)。
        if not enabled:
            for var in self.vars.values():
                var.set("")
        for entry in self.entries:
            entry.configure(state="normal" if enabled else "disabled")
        self._set_operations_enabled(enabled)

    def add_target(self) -> None:
        target = Target(name=f"target {len(self.targets.targets) + 1}")
        self.targets.targets.append(target)
        save_targets(self.paths, self.targets)
        # 行を作ってから選ぶ。逆順だと _select が「まだ無い行」を選ぼうとする。
        self._refresh_list()
        self._select(len(self.targets.targets) - 1)

    def delete_target(self) -> None:
        if self.index is None:
            return
        target = self.targets.targets.pop(self.index)
        save_targets(self.paths, self.targets)
        self._log(f"削除しました: {target.label}")
        self._refresh_list()
        # 詰めた後の同じ位置 (末尾を消したら 1 つ前) を選び直す。空になったら
        # index が -1 になり、_select が選択なし + 操作ボタン無効へ落とす。
        self._select(min(self.index, len(self.targets.targets) - 1))

    # --- editing ---------------------------------------------------------

    def _read_form(self) -> Target | None:
        """フォームの内容を Target にする。不正なら理由を出して None。"""
        port_text = self.vars["port"].get().strip()
        try:
            port = int(port_text)
        except ValueError:
            self._log(f"ポートが数値ではありません: {port_text!r}")
            return None
        host = self.vars["host"].get().strip()
        if not host:
            self._log("ホストが空です")
            return None
        try:
            return Target(
                name=self.vars["name"].get().strip() or "(no name)",
                host=host,
                port=port,
                config_path=self.vars["config_path"].get().strip(),
            )
        except ValueError as e:
            self._log(f"設定が不正です: {e}")
            return None

    def save_detail(self) -> Target | None:
        """フォームを選択中のエントリへ書き戻して永続化し、その Target を返す。

        操作ボタンもこれを通す。押す直前の編集が黙って捨てられ「直したはずの
        ホストへ送られない」状態を作らないため。
        """
        if self.index is None:
            return None
        target = self._read_form()
        if target is None:
            return None
        if self.targets.targets[self.index] != target:
            self.targets.targets[self.index] = target
            save_targets(self.paths, self.targets)
            self._refresh_list()
        return target

    # --- operations ------------------------------------------------------

    def send_operation(self, event: EventType) -> None:
        if self.sending:
            return
        target = self.save_detail()
        if target is None:
            return
        if event == EventType.reload and not target.config_path:
            # 受け側は file_path 必須 (WorkerInput の validation)。空のまま
            # 送ると相手側の例外として返ってくるだけなので、ここで止める。
            self._log("reload には対象マシン上の config パスが必要です")
            return
        self._set_operations_enabled(False)
        self.sending = True
        self._log(f"{target.name} へ {event.value} を送信中…")
        Thread(target=self._send_blocking, args=(target, event), daemon=True).start()

    def _send_blocking(self, target: Target, event: EventType) -> None:
        try:
            result = send(
                target.address,
                event,
                config_path=target.config_path,
                timeout=DEFAULT_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001 - UI へ必ず結果を返す
            result = SendResult(
                ok=False, elapsed_ms=0.0, detail=f"{type(e).__name__}: {e}"
            )
        self._schedule(self._on_result, target, event, result)

    def _schedule(self, callback: Callable[..., None], *args: Any) -> None:
        # 送信スレッドから呼ばれる。root が消えた後 (送信中に窓を閉じた) の
        # after() は例外になるので握る。
        try:
            self.after(0, callback, *args)
        except TclError, RuntimeError:
            pass

    def _on_result(self, target: Target, event: EventType, result: SendResult) -> None:
        self.status[target.address] = MARK_OK if result.ok else MARK_NG
        outcome = "OK" if result.ok else "NG"
        detail = f" {result.detail}" if result.detail else ""
        self._log(
            f"{target.name} {event.value} {outcome} ({result.elapsed_ms:.0f}ms){detail}"
        )
        self.sending = False
        self._set_operations_enabled(self.index is not None)
        self._refresh_list()

    def _set_operations_enabled(self, enabled: bool) -> None:
        for button in self.operation_buttons:
            button.configure(state="normal" if enabled else "disabled")

    # --- log -------------------------------------------------------------

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logbox.text.configure(state="normal")
        self.logbox.text.insert(END, f"{stamp} {message}\n")
        # 無限に伸ばさない。古い行から落として直近だけ残す。
        excess = int(self.logbox.text.index("end-1c").split(".")[0]) - LOG_MAX_LINES
        if excess > 0:
            self.logbox.text.delete("1.0", f"{excess + 1}.0")
        self.logbox.text.see(END)
        self.logbox.text.configure(state="disabled")


@click.command()
@click.option(
    "--config-dir", "config_dir", type=click.Path(path_type=Path), default=None
)
@click.option(
    "-t", "--theme", default="cosmo", type=click.Choice(list(STANDARD_THEMES.keys()))
)
def main(config_dir: Path | None, theme: str):
    root = Window(themename=theme)
    root.title("vspeech remote")
    root.geometry("720x520")
    root.minsize(640, 460)
    App(root, config_dir)
    root.mainloop()
