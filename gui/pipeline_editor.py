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
from gui.ports import is_port_free
from gui.process import PipelineRunner
from gui.profile import PipelineEntry
from gui.profile import load_pipeline_config
from gui.profile import save_pipeline_config
from gui.rawedit import RawTomlEditor
from gui.widgets import ScrolledText
from gui.widgets import Textbox
from vspeech.config import Config
from vspeech.logger import logger


class PipelineEditor(Frame):
    # Explicit class-level annotation: `tkinter.Misc` aliases `config` to
    # `configure`, so this must be declared here (not just inferred from an
    # `__init__` assignment) to override the inherited method's type for the
    # whole class rather than only within `__init__`. (Same fix as
    # `PipelineForm` in gui/form.py.)
    config: Config | None

    def __init__(self, master: Any, paths: ProfilePaths, on_dirty: Callable[[], None]):
        super().__init__(master)
        self.paths = paths
        self.on_dirty = on_dirty
        self.entry: PipelineEntry | None = None
        self.config = None
        self.runner: PipelineRunner | None = None
        self.broken = False

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
        self.start_bt = Button(controls, text="Start", command=self.start)
        self.start_bt.pack(side=LEFT, padx=4, pady=4)
        self.stop_bt = Button(controls, text="Stop", command=self.stop, state=DISABLED)
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
        Button(send_frame, text="send", command=self.send_text).pack(side=LEFT, padx=4)

        self.log = ScrolledText(self, height=10, state=DISABLED)
        self.log.pack(fill=BOTH, expand=True)

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
            self.start_bt.configure(state=NORMAL)
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
            self.start_bt.configure(state=DISABLED)

    # --- form/raw sync --------------------------------------------------

    def _on_tab_changed(self, _event: Any) -> None:
        if self.notebook.select() == str(self.raw) and self.config is not None:
            self.sync_form_to_config()
            self.raw.set_config(self.config)

    def sync_form_to_config(self) -> None:
        if self.config is not None:
            self.form.read_into(self.config)

    def apply_raw(self) -> None:
        config, error = self.raw.parse()
        if config is None:
            self.banner.configure(text=f"❗ TOML エラー: {error}")
            return
        self.broken = False
        self.banner.configure(text="")
        self.config = config
        self.form.bind_config(config)
        self.start_bt.configure(state=NORMAL)
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
            self.start_bt.configure(state=NORMAL)
        else:
            self.sync_form_to_config()
        save_pipeline_config(self.paths, self.entry, self.config)
        logger.info("saved pipeline %s", self.entry.id)
        return True

    # --- runtime --------------------------------------------------------

    def start(self) -> None:
        if self.entry is None or not self.save():
            return
        if not is_port_free(self.entry.port):
            self._append_log(f"port {self.entry.port} is busy; cannot start")
            return
        self.runner = PipelineRunner(
            config_path=self.paths.pipeline_config(self.entry.id),
            port=self.entry.port,
            on_log=self._schedule_log,
            on_exit=self._schedule_exit,
        )
        self.runner.start()
        self.start_bt.configure(state=DISABLED)
        self.stop_bt.configure(state=NORMAL)
        self.status.configure(text="● running")

    def stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def _schedule_log(self, line: str) -> None:
        # `Misc.after` returns a scheduler id (str); wrap it in a `-> None`
        # method (not a lambda) so this matches PipelineRunner's
        # `Callable[[str], None]` on_log type. Called from the runner's
        # reader thread, so the UI update itself is marshaled via `after`.
        self.log.after(0, self._append_log, line)

    def _schedule_exit(self, code: int) -> None:
        self.log.after(0, self._on_exit, code)

    def _on_exit(self, code: int) -> None:
        self._append_log(f"process exited: {code}")
        self.start_bt.configure(state=NORMAL)
        self.stop_bt.configure(state=DISABLED)
        self.status.configure(text="■ stopped")

    def send_text(self) -> None:
        if self.runner and self.runner.is_running() and self.config is not None:
            self.runner.send_text(
                self.send_entry.get_value(), self.config.text_send_operations
            )

    def _append_log(self, line: str) -> None:
        self.log.configure(state=NORMAL)
        self.log.insert(END, line + "\n")
        self.log.configure(state=DISABLED)
        self.log.yview(END)
