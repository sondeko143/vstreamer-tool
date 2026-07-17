from collections import deque
from pathlib import Path
from threading import Thread
from tkinter import BOTH
from tkinter import END
from tkinter import LEFT
from tkinter import RIGHT
from tkinter import Listbox
from tkinter import TclError
from tkinter import Y
from typing import Any
from uuid import uuid4

import click
from ttkbootstrap import Button
from ttkbootstrap import Frame
from ttkbootstrap import Window
from ttkbootstrap.themes.standard import STANDARD_THEMES

from gui.dialogs import RecipeDialog
from gui.migration import quarantine
from gui.paths import resolve_paths
from gui.pipeline_editor import PipelineEditor
from gui.ports import allocate_free_port
from gui.ports import is_port_free
from gui.process import PipelineRunner
from gui.profile import PipelineEntry
from gui.profile import load_default_config
from gui.profile import load_profile
from gui.profile import save_pipeline_config
from gui.profile import save_profile
from gui.recipes import RECIPES_BY_KEY
from vspeech.logger import logger

LOG_BUFFER_MAX = 2000
# これより早く落ちたら「起動に失敗した」とみなす。正常起動なら worker が
# 走り続けるので、この窓で終わるのは起動失敗だけ。
QUICK_EXIT_SEC = 10.0
# 失敗バナーに載せる実出力末尾の行数 (合成の "process exited" 行は含めない)。
# ADR-0038 の preflight 失敗は「起動中止: 設定不備 N 件」ヘッダ + 問題行ずつを
# 吐くので、ヘッダと数件の問題が収まる程度に採る (最終的に 400 字で切る)。
FAILURE_TAIL_LINES = 8


class App(Frame):
    def __init__(self, master: Any, profile_dir: Path | None):
        super().__init__(master)
        self._disable_input_mousewheel()
        self.pack(fill=BOTH, expand=True)
        self.paths = resolve_paths(profile_dir)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.default_config = load_default_config(self.paths)
        self.profile = load_profile(self.paths)
        # Runners live at the App level, keyed by pipeline id, so a pipeline
        # keeps running (and stays stoppable) no matter which pipeline the
        # editor is currently showing. Each pipeline's log is buffered here too
        # and replayed into the editor when its pipeline is selected.
        self.runners: dict[str, PipelineRunner] = {}
        self.logs: dict[str, deque[str]] = {}

        left = Frame(self)
        left.pack(side=LEFT, fill=Y, padx=(6, 2), pady=6)
        self.listbox = Listbox(left, width=32)
        self.listbox.pack(fill=Y, expand=True, pady=(0, 4))
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Button-1>", self._ignore_empty_click)
        Button(left, text="+ new", command=self.new_pipeline).pack(fill="x")
        Button(left, text="del", command=self.delete_pipeline).pack(fill="x")

        self.editor = PipelineEditor(
            self,
            self.paths,
            on_dirty=lambda: None,
            on_start=self._start_current,
            on_stop=self._stop_current,
            on_send=self._send_current,
        )
        self.editor.pack(side=RIGHT, fill=BOTH, expand=True)

        self._refresh_list()

    def _disable_input_mousewheel(self) -> None:
        # ttk Spinbox/Combobox grab the mouse wheel to change their own value,
        # so scrolling the form over one silently changes it. Drop the class-
        # level wheel bindings (app-wide) so the wheel only scrolls the
        # ScrolledFrame form body instead of mutating whatever is under it.
        for widget_class in ("TSpinbox", "TCombobox"):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_class(widget_class, sequence)

    # --- pipeline list --------------------------------------------------

    def _is_running(self, pipeline_id: str) -> bool:
        runner = self.runners.get(pipeline_id)
        return runner is not None and runner.is_running()

    def _refresh_list(self) -> None:
        # Preserves the current selection INDEX. This is correct only for a
        # ramp/status refresh where the list order and length are unchanged
        # (e.g. _start_current / _on_exit). After a STRUCTURAL change
        # (delete/new shifts indices), the caller must follow with
        # _select_index(...), which clears this restore and re-syncs the editor
        # — otherwise a stale index would highlight the wrong pipeline.
        selection = self.listbox.curselection()
        self.listbox.delete(0, END)
        for entry in self.profile.pipelines:
            ramp = "●" if self._is_running(entry.id) else "■"
            self.listbox.insert(END, f"{ramp} {entry.name}  :{entry.port}")
        if selection:
            self.listbox.selection_set(selection[0])

    def _ignore_empty_click(self, event: Any) -> str | None:
        # A Listbox selects the last item when you click the empty space below
        # the items. Swallow clicks that land below the last row so that area is
        # inert (the current selection and editor stay put).
        if not self.profile.pipelines:
            return "break"
        index = self.listbox.nearest(event.y)
        bbox = self.listbox.bbox(index)
        if bbox is None or event.y > bbox[1] + bbox[3]:
            return "break"
        return None

    def _on_select(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.profile.pipelines[selection[0]]
        # <<ListboxSelect>> fires on every click (sometimes twice), so
        # re-clicking the already-shown pipeline would rebuild the whole form
        # and flicker. Skip when the selection hasn't actually changed.
        if self.editor.entry is not None and self.editor.entry.id == entry.id:
            return
        self._load_selected(selection[0])

    def _load_selected(self, index: int) -> None:
        entry = self.profile.pipelines[index]
        self.editor.load_entry(entry)
        self.editor.set_running(self._is_running(entry.id))
        self.editor.set_log(list(self.logs.get(entry.id, [])))

    def _select_index(self, index: int) -> None:
        # Programmatic selection does NOT fire <<ListboxSelect>>, so re-sync the
        # editor explicitly. Used after a structural change (delete/new) where a
        # raw index restore would otherwise leave the editor showing one pipeline
        # while the listbox highlights a different (shifted-up) one.
        self.listbox.selection_clear(0, END)
        if 0 <= index < len(self.profile.pipelines):
            self.listbox.selection_set(index)
            self.listbox.activate(index)
            self._load_selected(index)
        else:
            self.editor.clear()

    # --- runner lifecycle (per pipeline, App-owned) ---------------------

    def _start_current(self) -> None:
        entry = self.editor.entry
        if entry is None or not self.editor.save():
            return
        if self._is_running(entry.id):
            return
        if not is_port_free(entry.port):
            self.editor.append_log(f"port {entry.port} is busy; cannot start")
            return
        pipeline_id = entry.id
        # 新しい run はログを最初から始める。持ち越すと前 run の残骸
        # (前回の "process exited" 行など) が失敗バナーの failure_tail に
        # 混じりうる (_on_exit)。表示中なら pane も空にして揃える。
        self.logs[pipeline_id] = deque(maxlen=LOG_BUFFER_MAX)
        if self.editor.entry is not None and self.editor.entry.id == pipeline_id:
            self.editor.set_log([])
        runner = PipelineRunner(
            config_path=self.paths.pipeline_config(pipeline_id),
            port=entry.port,
            on_log=lambda line: self._schedule_log(pipeline_id, line),
            on_exit=lambda code: self._schedule_exit(pipeline_id, code),
        )
        self.runners[pipeline_id] = runner
        runner.start()
        self.editor.set_running(True)
        self._refresh_list()

    def _stop_current(self) -> None:
        entry = self.editor.entry
        if entry is not None and entry.id in self.runners:
            # stop() blocks (terminate → wait → kill); run it off the Tk thread
            # so the UI stays responsive. The ramp/log update when the process
            # actually dies, via _pump → on_exit → after.
            Thread(target=self.runners[entry.id].stop, daemon=True).start()

    def _send_current(self, text: str) -> None:
        entry = self.editor.entry
        if entry is None:
            return
        runner = self.runners.get(entry.id)
        config = self.editor.config
        if runner is not None and runner.is_running() and config is not None:
            runner.send_text(text, config.text_send_operations)

    def _schedule_log(self, pipeline_id: str, line: str) -> None:
        # Fired from the runner's reader thread. after() raises once the root is
        # gone (window closed while a child is still terminating) — swallow it.
        try:
            self.after(0, self._on_log, pipeline_id, line)
        except TclError, RuntimeError:
            pass

    def _schedule_exit(self, pipeline_id: str, code: int) -> None:
        try:
            self.after(0, self._on_exit, pipeline_id, code)
        except TclError, RuntimeError:
            pass

    def _on_log(self, pipeline_id: str, line: str) -> None:
        if pipeline_id not in self.runners:
            return  # pipeline was deleted; drop its late in-flight output
        self.logs.setdefault(pipeline_id, deque(maxlen=LOG_BUFFER_MAX)).append(line)
        if self.editor.entry is not None and self.editor.entry.id == pipeline_id:
            self.editor.append_log(line)

    def _on_exit(self, pipeline_id: str, code: int) -> None:
        runner = self.runners.get(pipeline_id)
        if runner is None:
            return  # deleted while its process was terminating — nothing to show
        log = self.logs.setdefault(pipeline_id, deque(maxlen=LOG_BUFFER_MAX))
        # 意図的な停止 (runner.stopping) は即死とみなさない。terminate の
        # exit code は非 0 なので、これが無いと Stop 直後に誤って失敗バナーが出る。
        quick = code != 0 and not runner.stopping and runner.ran_for() < QUICK_EXIT_SEC
        # 失敗理由は "process exited" の合成行を足す前の実出力末尾から採る。
        # 先に足すと、合成行が末尾枠を 1 つ食って preflight の件数ヘッダ
        # (「起動中止: 設定不備 N 件」) を押し出してしまう。
        failure_tail = list(log)[-FAILURE_TAIL_LINES:] if quick else []
        message = f"process exited: {code}"
        log.append(message)
        if self.editor.entry is not None and self.editor.entry.id == pipeline_id:
            self.editor.append_log(message)
            self.editor.set_running(False)
            if quick:
                self.editor.show_launch_failure(failure_tail)
        self._refresh_list()

    # --- new / delete ---------------------------------------------------

    def new_pipeline(self) -> None:
        dialog = RecipeDialog(self)
        if dialog.result is None:
            return
        recipe = RECIPES_BY_KEY[dialog.result]
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
        self._select_index(len(self.profile.pipelines) - 1)

    def delete_pipeline(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        entry = self.profile.pipelines.pop(index)
        if entry.id in self.runners:
            # Stop off-thread (same as _stop_current) so deleting a RUNNING
            # pipeline doesn't freeze the UI for stop()'s terminate→wait→kill.
            # The thread keeps its own reference, so we can drop it from the
            # dict immediately.
            runner = self.runners.pop(entry.id)
            Thread(target=runner.stop, daemon=True).start()
        self.logs.pop(entry.id, None)
        config_path = self.paths.pipeline_config(entry.id)
        if config_path.exists():
            quarantine(config_path)
            config_path.unlink()
        save_profile(self.paths, self.profile)
        if self.editor.entry is not None and self.editor.entry.id == entry.id:
            self.editor.clear()
        self._refresh_list()
        # Re-select by position (clamped) and re-sync the editor, so a stale
        # old-list index can never leave a shifted-up pipeline highlighted.
        self._select_index(min(index, len(self.profile.pipelines) - 1))
        logger.info("deleted pipeline %s", entry.id)

    def on_close(self) -> None:
        # Signal every runner to terminate WITHOUT waiting, then close
        # immediately — waiting serially on each Popen.wait() would freeze the
        # window for the sum of all shutdown times. The children get the
        # terminate signal synchronously (fast) before the process exits.
        for runner in self.runners.values():
            runner.request_stop()
        self.master.destroy()


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
    app = App(root, profile_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
