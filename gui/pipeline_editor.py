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
from gui.readiness import Readiness
from gui.readiness import evaluate
from gui.readiness_panel import ReadinessPanel
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

        self.readiness: Readiness | None = None
        # 「preflight は不足と言うが承知の上で起動する」を 1 回だけ通すフラグ。
        self.force_start = False
        self.panel = ReadinessPanel(self, on_fix=self._focus_field)
        self.panel.pack(fill=X)

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
        self.start_hint = Label(controls, text="", bootstyle="danger")
        self.start_hint.pack(side=LEFT, padx=(0, 4))
        # 未充足でも起動する escape hatch。技術者が「preflight は不足と言うが
        # 承知の上で起動する」を選べるように小さく残す（既定は無効化側）。
        self.force_bt = Button(
            controls,
            text="未充足でも起動",
            bootstyle="link-danger",
            command=self._force_start_click,
        )
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

    def _force_start_click(self) -> None:
        self.force_start = True
        try:
            self.on_start()
        finally:
            self.force_start = False
        self._refresh_start_button()

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

    @property
    def readiness_ok(self) -> bool:
        return self.readiness is None or self.readiness.ok

    def refresh_readiness(self) -> None:
        """フォームの現在値を config へ書き戻してから再評価する。

        判断は vspeech.preflight が持つ (ADR-0045) ので、ここは呼ぶだけ。
        preflight はデバイス列挙などの I/O を伴うため、呼ぶのは
        load / save / raw 適用の時だけにする（毎キーストロークでは呼ばない）。
        """
        if self.config is None:
            self.readiness = None
            self.panel.clear()
            # 壊れ config を選ぶとフォームは前 pipeline のまま残る(bind_config を
            # 呼ばない)。その ✗ 印が居座らないよう消す。
            self.form.mark_problems(set())
        else:
            self.sync_form_to_config()
            self.readiness = evaluate(self.config)
            self.panel.show(self.readiness)
            self.form.mark_problems(
                {
                    problem.field
                    for worker in self.readiness.workers
                    for problem in worker.problems
                    if problem.field is not None
                }
            )
        self._refresh_start_button()

    def _focus_field(self, path: str) -> None:
        self.notebook.select(self.form)
        if not self.form.focus_field(path):
            self.banner.configure(
                text=f"⚠ {path} はフォームに出ていません（Raw TOML で編集してください）"
            )

    def _refresh_start_button(self) -> None:
        can_start = self.entry is not None and not self.broken and not self.running
        gated = can_start and not self.readiness_ok and not self.force_start
        self.start_bt.configure(state=DISABLED if gated or not can_start else NORMAL)
        if gated and self.readiness is not None:
            self.start_hint.configure(text=f"{self.readiness.problem_count} 件未充足")
            self.force_bt.pack(side=LEFT, padx=4)
        else:
            self.start_hint.configure(text="")
            self.force_bt.pack_forget()

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

    def show_launch_failure(self, lines: list[str]) -> None:
        """起動直後に落ちた理由をログを開かずに見せる。

        vspeech は設定不備なら「起動中止: 設定不備 N 件」+ 各問題 を整形して
        吐いて exit する (ADR-0038) ので、末尾数行がそのまま理由になる。
        """
        reason = " / ".join(line for line in lines if line.strip()) or "(ログを参照)"
        self.banner.configure(text=f"❗ 起動に失敗しました: {reason}"[:400])

    def clear(self) -> None:
        self.entry = None
        self.config = None
        self.broken = False
        self.running = False
        self.banner.configure(text="")
        self.stop_bt.configure(state=DISABLED)
        self.status.configure(text="■ stopped")
        self.readiness = None
        self.panel.clear()
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
        self.refresh_readiness()

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
        self.refresh_readiness()
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
            # ボタン状態は末尾の refresh_readiness() が更新するのでここでは呼ばない。
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
        self.refresh_readiness()
        logger.info("saved pipeline %s", self.entry.id)
        return True
