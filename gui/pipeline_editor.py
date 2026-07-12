from collections.abc import Callable
from tkinter import BOTH
from tkinter import DISABLED
from tkinter import END
from tkinter import LEFT
from tkinter import NORMAL
from tkinter import X
from typing import Any

from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Notebook

from gui.form import PipelineForm
from gui.paths import ProfilePaths
from gui.profile import PipelineEntry
from gui.profile import load_pipeline_config
from gui.profile import save_pipeline_config
from gui.rawedit import RawTomlEditor
from gui.widgets import ScrolledText
from gui.widgets import Textbox
from vspeech.config import Config
from vspeech.logger import logger


class PipelineEditor(Frame):
    # See gui/form.py: tkinter.Misc aliases `config` to `configure`, so the
    # bound-config attribute must be declared at class level to override the
    # inherited method's type for the whole class (not just within __init__).
    config: Config | None

    def __init__(
        self,
        master: Any,
        paths: ProfilePaths,
        on_dirty: Callable[[], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_send: Callable[[str], None],
    ):
        # Internal padding so the editor's content isn't flush against the
        # window edge (esp. left/right).
        super().__init__(master, padding=(8, 6))
        self.paths = paths
        self.on_dirty = on_dirty
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_send = on_send
        self.entry: PipelineEntry | None = None
        self.config = None
        self.broken = False
        self.running = False

        self.banner = Label(self, text="", bootstyle="danger")
        self.banner.pack(fill=X)

        notebook = Notebook(self)
        notebook.pack(fill=BOTH, expand=True)
        self.form = PipelineForm(notebook, on_change=self.on_dirty)
        self.raw = RawTomlEditor(notebook)
        notebook.add(self.form, text="Form")
        notebook.add(self.raw, text="Raw TOML")
        self.notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        controls = Frame(self)
        controls.pack(fill=X)
        self.start_bt = Button(controls, text="Start", command=self._start_click)
        self.start_bt.pack(side=LEFT, padx=4, pady=4)
        self.stop_bt = Button(
            controls, text="Stop", command=self._stop_click, state=DISABLED
        )
        self.stop_bt.pack(side=LEFT, padx=4, pady=4)
        self.status = Label(controls, text="■ stopped")
        self.status.pack(side=LEFT, padx=8)
        self.apply_raw_bt = Button(controls, text="Apply Raw", command=self.apply_raw)
        self.apply_raw_bt.pack(side=LEFT, padx=4)
        self.save_bt = Button(controls, text="Save", command=self.save)
        self.save_bt.pack(side=LEFT, padx=4)

        send_frame = Frame(self)
        send_frame.pack(fill=X)
        self.send_entry = Textbox(send_frame)
        self.send_entry.pack(side=LEFT, fill=X, expand=True, padx=4)
        Button(send_frame, text="send", command=self._send_click).pack(
            side=LEFT, padx=4
        )

        self.log = ScrolledText(self, height=10, state=DISABLED)
        self.log.pack(fill=BOTH, expand=True)
        self._refresh_start_button()

    # --- button clicks delegate to App-provided callbacks ---------------

    def _start_click(self) -> None:
        self.on_start()

    def _stop_click(self) -> None:
        self.on_stop()

    def _send_click(self) -> None:
        self.on_send(self.send_entry.get_value())

    # --- runtime state, driven by App (which owns the runners) ----------

    def set_running(self, running: bool) -> None:
        self.running = running
        self.stop_bt.configure(state=NORMAL if running else DISABLED)
        self.status.configure(text="● running" if running else "■ stopped")
        self._refresh_start_button()

    def _refresh_start_button(self) -> None:
        can_start = self.entry is not None and not self.broken and not self.running
        self.start_bt.configure(state=NORMAL if can_start else DISABLED)

    def set_log(self, lines: list[str]) -> None:
        # `self.log` is a ttkbootstrap ScrolledText (a Frame wrapper); only its
        # inner Text (`self.log.text`) accepts `state`/`insert`/`delete`. Calling
        # `self.log.configure(state=...)` hits the Frame and raises a TclError.
        self.log.text.configure(state=NORMAL)
        self.log.delete("1.0", END)
        for line in lines:
            self.log.insert(END, line + "\n")
        self.log.text.configure(state=DISABLED)
        self.log.yview(END)

    def append_log(self, line: str) -> None:
        self.log.text.configure(state=NORMAL)
        self.log.insert(END, line + "\n")
        self.log.text.configure(state=DISABLED)
        self.log.yview(END)

    def clear(self) -> None:
        self.entry = None
        self.config = None
        self.broken = False
        self.running = False
        self.banner.configure(text="")
        self.stop_bt.configure(state=DISABLED)
        self.status.configure(text="■ stopped")
        self._refresh_start_button()
        self.set_log([])

    # --- loading --------------------------------------------------------

    def load_entry(self, entry: PipelineEntry) -> None:
        self.entry = entry
        result = load_pipeline_config(self.paths, entry)
        if result.ok and result.value is not None:
            self.broken = False
            self.banner.configure(text="")
            self.config = result.value
            self.form.bind_config(self.config)
            self.raw.set_config(self.config)
            if result.migrated:
                save_pipeline_config(self.paths, entry, self.config)
                self.on_dirty()
        else:
            self.broken = True
            self.config = None
            if result.quarantined_path is not None:
                self.banner.configure(
                    text=f"❗ config 読込失敗: {result.quarantined_path} に退避。生TOMLで修正してください — {result.error}"
                )
            else:
                self.banner.configure(text=f"❗ config 読込失敗: {result.error}")
            self.raw.set_text(result.raw_text or "")
            self.notebook.select(self.raw)
        self._refresh_start_button()

    # --- form/raw sync --------------------------------------------------

    def _on_tab_changed(self, _event: Any) -> None:
        if self.notebook.select() == str(self.raw) and self.config is not None:
            failed = self.sync_form_to_config()
            self.raw.set_config(self.config)
            self.banner.configure(
                text=(
                    f"⚠ フォームで反映できなかった項目: {', '.join(failed)}"
                    if failed
                    else ""
                )
            )

    def sync_form_to_config(self) -> list[str]:
        if self.config is not None:
            return self.form.read_into(self.config)
        return []

    def apply_raw(self) -> None:
        config, error = self.raw.parse()
        if config is None:
            self.banner.configure(text=f"❗ TOML エラー: {error}")
            return
        self.broken = False
        self.banner.configure(text="")
        self.config = config
        self.form.bind_config(config)
        self._refresh_start_button()
        self.on_dirty()

    def save(self) -> bool:
        if self.entry is None:
            return False
        if self.broken or self.config is None:
            config, error = self.raw.parse()
            if config is None:
                self.banner.configure(text=f"❗ 保存不可 (TOML エラー): {error}")
                return False
            self.config = config
            self.broken = False
            self.banner.configure(text="")
            self.form.bind_config(self.config)
            self._refresh_start_button()
        else:
            failed = self.sync_form_to_config()
            self.banner.configure(
                text=(
                    f"⚠ 反映できなかった項目: {', '.join(failed)}(値を確認してください)"
                    if failed
                    else ""
                )
            )
        save_pipeline_config(self.paths, self.entry, self.config)
        logger.info("saved pipeline %s", self.entry.id)
        return True
