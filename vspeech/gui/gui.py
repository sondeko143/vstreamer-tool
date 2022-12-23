import json
import sys
from codecs import encode
from functools import partial
from locale import getpreferredencoding
from logging import Handler
from logging import LogRecord
from pathlib import Path
from socket import AF_INET
from socket import SOCK_STREAM
from socket import socket
from subprocess import Popen
from subprocess import TimeoutExpired
from tempfile import NamedTemporaryFile
from tkinter import BOTTOM
from tkinter import END
from tkinter import EW
from tkinter import INSERT
from tkinter import LEFT
from tkinter import SEL
from tkinter import BooleanVar
from tkinter import Listbox
from tkinter import Menu
from tkinter import StringVar
from tkinter import Tk
from tkinter import Variable
from tkinter import W
from tkinter import X
from tkinter import filedialog
from tkinter import font
from tkinter import messagebox
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import get_type_hints

from humps import pascalize
from pyvcroid2 import VcRoid2
from ttkbootstrap import Button
from ttkbootstrap import Checkbutton as ttkCheckbutton
from ttkbootstrap import Entry
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Notebook
from ttkbootstrap import ScrolledText as ttkScrolledText
from ttkbootstrap import Spinbox as ttkSpinbox
from ttkbootstrap import Text
from ttkbootstrap import Window

from vspeech.audio import list_all_devices
from vspeech.config import Config
from vspeech.config import SpeechWorkerType
from vspeech.config import TranscriptionWorkerType
from vspeech.config import VR2Param
from vspeech.gui.autocomplete_combobox import AutocompleteCombobox
from vspeech.gui.dummy_param import create_dummy_param
from vspeech.logger import logger


class ScrolledText(ttkScrolledText):
    def select_range(self, start: Union[str, int], end: Union[str, int]):
        start_index = start
        if isinstance(start, int) or start.isnumeric():
            start_index = str(1.0 + float(start))
        end_index = end
        if isinstance(end, int) or end.isnumeric():
            end_index = str(1.0 + float(end))
        self.tag_add(SEL, start_index, end_index)
        self.mark_set(INSERT, start_index)
        self.see(INSERT)
        return "break"

    def icursor(self, pos: Union[str, int]):
        pos_index = pos
        if isinstance(pos, int) or pos.isnumeric():
            pos_index = str(1.0 + float(pos))
        self.see(pos_index)


class TextHandler(Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text: Text):
        # run the regular Handler __init__
        Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record: LogRecord):
        msg = self.format(record)

        def append():
            self.text.configure(state="normal")
            self.text.insert(END, msg + "\n")
            self.text.configure(state="disabled")
            # Autoscroll to the bottom
            self.text.yview(END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)


class Spinbox(ttkSpinbox):
    def get_value(self):
        return super().get()

    def get_label_for_item_value(self, value: Any):
        return value


class Checkbutton(ttkCheckbutton):
    var: BooleanVar

    def __init__(self, master: Any, **kw: Any):
        self.var = BooleanVar()
        super().__init__(master, variable=self.var, onvalue=True, offvalue=False, **kw)

    def get_value(self):
        return self.var.get()

    def get_label_for_item_value(self, value: Any):
        return value

    def set(self, value: Any):
        self.var.set(value)


class Textbox(Entry):
    var: StringVar

    def __init__(self, master: Any, **kw: Any):
        self.var = StringVar()
        super().__init__(master, textvariable=self.var, **kw)

    def get_value(self):
        return self.var.get()

    def get_label_for_item_value(self, value: Any):
        return value

    def set(self, value: Any):
        self.var.set(value)


T = TypeVar("T")

Widgets = Union[AutocompleteCombobox[Any], Spinbox, Checkbutton, Textbox]


class VspeechGUI(Frame):
    master: Tk
    config: Config
    config_entry_map: Dict[Widgets, str]
    thread: Optional[Popen[bytes]]
    config_file_path: Path
    temp_config_file_path: Path
    paused: bool = False
    pr_toggle_bt: Button
    send_bt: Button
    reload_bt: Button
    templates: Variable

    def save_config_as(self):
        file = filedialog.asksaveasfile("w", defaultextension="json")
        if not file:
            return
        file_name = file.name
        file.write(self.config.json())
        file.close()
        logger.info("保存しました: %s", file_name)
        self.config_file_path = Path(file_name)

    def save_config(self):
        with self.config_file_path.open("w") as file:
            file.write(self.config.json())
            logger.info("保存しました: %s", file.name)

    def load_file(self):
        file = filedialog.askopenfile("r", defaultextension="json")
        if not file:
            return
        file_name = file.name
        config_obj = json.loads(file.read())
        file.close()
        self.config = Config.parse_obj(config_obj)
        self.load_config()
        logger.info("ロードしました: %s", file_name)
        self.config_file_path = Path(file_name)

    def __init__(self, master: Tk):
        super().__init__(master)
        self.master = master
        self.thread = None
        self.config_entry_map = {}
        if not Config.Config.CLI_JSON_CONFIG_PATH:
            raise Exception()
        self.config_file_path = Path(Config.Config.CLI_JSON_CONFIG_PATH)
        self.config = Config.parse_file(self.config_file_path)

        menubar = Menu(master)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_config)
        file_menu.add_command(label="Save as ...", command=self.save_config_as)
        file_menu.add_command(label="Load", command=self.load_file)
        menubar.add_cascade(label="File", menu=file_menu)
        master.configure(menu=menubar)

        self.pack(fill=X)

        parameter_frame = Frame(self)
        parameter_frame.pack(fill=X)
        bt_frame = Frame(self)
        bt_frame.pack(fill=X)
        text_frame = Frame(self)
        text_frame.pack(fill=X)
        log_frame = Frame(self)
        log_frame.pack(fill=X, side=BOTTOM)

        notebook = Notebook(parameter_frame)
        notebook.pack(fill=X)
        self.draw_rec_tab(notebook=notebook)
        self.draw_speech_tab(notebook=notebook)
        self.draw_subtitle_tab(notebook=notebook)
        self.draw_transcription_tab(notebook=notebook)
        self.draw_translation_tab(notebook=notebook)
        self.draw_vr2_tab(notebook=notebook)
        self.draw_voicevox_tab(notebook=notebook)
        self.draw_ami_tab(notebook=notebook)
        self.draw_google_tab(notebook=notebook)
        self.draw_template_text_tab(notebook=notebook)

        run_bt = Button(bt_frame, text="run", command=self.run_vspeech)
        run_bt.pack(padx=5, pady=5, side=LEFT)
        self.pr_toggle_bt = Button(
            bt_frame, text="pause", command=self.pause_or_resume, state="disabled"
        )
        self.pr_toggle_bt.pack(padx=5, pady=5, side=LEFT)
        stop_bt = Button(bt_frame, text="stop", command=self.terminate_main)
        stop_bt.pack(padx=5, pady=5, side=LEFT)
        self.reload_bt = Button(
            bt_frame,
            text="reload",
            command=self.reload_config,
            state="disabled",
        )
        self.reload_bt.pack(padx=5, pady=5, side=LEFT)

        text = ScrolledText(text_frame, height=1)
        text.pack(padx=5, pady=5, fill=X)
        self.send_bt = Button(
            text_frame,
            text="send",
            command=partial(self.send_text, text),
            state="disabled",
        )
        self.send_bt.pack(padx=5, pady=5, side=LEFT)

        log_box = ScrolledText(log_frame, state="disabled")
        log_box.configure(font="TkFixedFont")
        log_box.pack(padx=5, pady=5, side=BOTTOM)
        logger.addHandler(TextHandler(log_box))
        logger.setLevel(self.config.log_level)

        self.load_config()

    def draw_rec_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        input_devices = list_all_devices(input=True)
        self.draw_cb(
            frame=tab_frame,
            candidates=input_devices,
            config_name="input_device_index",
        ).grid(column=0, row=0, columnspan=max_columns, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="silence_threshold",
            from_=0,
            to=100,
            increment=0.1,
        ).grid(column=0, row=1, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="record_interval_sec",
            from_=0,
            to=10,
            increment=0.1,
        ).grid(column=1, row=1, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="max_recording_sec",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=2, row=1, sticky=EW)
        notebook.add(tab_frame, text="rec")

    def draw_speech_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        output_devices = list_all_devices(output=True)
        self.draw_cb(
            frame=tab_frame,
            candidates=output_devices,
            config_name="output_device_index",
        ).grid(column=0, row=0, columnspan=max_columns, sticky=EW)

        worker_types = {
            worker_type.name: worker_type for worker_type in SpeechWorkerType
        }
        self.draw_cb(
            frame=tab_frame,
            candidates=worker_types,
            config_name="speech_worker_type",
        ).grid(column=0, row=1, columnspan=max_columns, sticky=EW)

        notebook.add(tab_frame, text="vc")

    def draw_subtitle_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        font_cb_list = sorted(font.families())
        self.draw_cb(
            frame=tab_frame,
            candidates={font: font for font in font_cb_list},
            config_name="subtitle_font_family",
        ).grid(column=0, row=1, columnspan=max_columns, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="min_subtitle_display_sec",
            from_=0,
            to=10,
            increment=0.1,
        ).grid(column=0, row=1, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="max_subtitle_text_len",
            from_=0,
            to=100,
            increment=1,
        ).grid(column=1, row=1, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name="max_subtitle_translated_len",
            from_=0,
            to=100,
            increment=1,
        ).grid(column=2, row=1, sticky=EW)
        notebook.add(tab_frame, text="sub")

    def draw_transcription_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 2
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        worker_types = {
            worker_type.name: worker_type for worker_type in TranscriptionWorkerType
        }
        self.draw_cb(
            frame=tab_frame,
            candidates=worker_types,
            config_name="transcription_worker_type",
        ).grid(column=0, row=0, columnspan=max_columns, sticky=EW)
        notebook.add(tab_frame, text="transc")

    def draw_ami_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 2
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        self.draw_tb(tab_frame, config_name="ami_appkey").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name="ami_engine_uri").grid(
            column=0, row=1, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name="ami_engine_name").grid(
            column=0, row=2, sticky=EW
        )
        self.draw_tb(tab_frame, config_name="ami_service_id").grid(
            column=1, row=2, sticky=EW
        )
        notebook.add(tab_frame, text="ami")

    def draw_translation_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        self.draw_checkbutton(frame=tab_frame, config_name="enable_translation").grid(
            column=0, row=0, columnspan=max_columns, sticky=W
        )
        notebook.add(tab_frame, text="transl")

    def draw_google_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        self.draw_tb(tab_frame, config_name="gcp_project_id").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name="gcp_credentials_file_path").grid(
            column=0, row=1, columnspan=max_columns, sticky=EW
        )
        notebook.add(tab_frame, text="google")

    def draw_vr2_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        with VcRoid2() as vr2:
            voice_lists = vr2.listVoices()
            self.draw_cb(
                frame=tab_frame,
                candidates={v: v for v in voice_lists},
                config_name="vr2_voice_name",
            ).grid(column=0, row=1, columnspan=max_columns, sticky=EW)
            params = list(get_type_hints(VR2Param).keys())
            chunked_list: List[List[str]] = list()
            for i in range(0, len(params), max_columns):
                chunked_list.append(params[i : i + max_columns])
            dummy_param = create_dummy_param()
            for row, params_chunk in enumerate(chunked_list):
                for column, param_name in enumerate(params_chunk):
                    min_value = getattr(dummy_param, f"min{pascalize(param_name)}", 0)
                    max_value = getattr(
                        dummy_param, f"max{pascalize(param_name)}", 1000
                    )
                    self.draw_sb(
                        frame=tab_frame,
                        config_name=f"vr2_params.{param_name}",
                        from_=min_value,
                        to=max_value,
                        increment=0.01,
                    ).grid(column=column, row=row + 2, sticky=EW)

        notebook.add(tab_frame, text="vr2")

    def draw_voicevox_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        self.draw_tb(tab_frame, config_name="voicevox_core_dir").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name="openjtalk_dir").grid(
            column=0, row=1, columnspan=max_columns, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name="voicevox_speaker_id",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=0, row=2, sticky=EW)

        notebook.add(tab_frame, text="vvox")

    def draw_template_text_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        text_candidate = Entry(tab_frame)
        text_candidate.pack(padx=5, pady=5, fill=X)
        button_frame = Frame(tab_frame)
        button_frame.pack(fill=X)
        add_bt = Button(
            button_frame,
            text="add",
            command=partial(self.add_text_to_template, text_candidate),
        )
        add_bt.pack(padx=5, pady=5, side=LEFT)
        texts = self.config.template_texts
        self.templates = Variable(value=texts)
        template_lb = Listbox(tab_frame, listvariable=self.templates, height=6)
        template_lb.pack(padx=5, pady=5, fill=X)
        send_bt = Button(
            tab_frame,
            text="send",
            command=partial(self.send_selected_template_texts, template_lb),
        )
        send_bt.pack(padx=5, pady=5, side=LEFT)
        del_bt = Button(
            tab_frame,
            text="del",
            command=partial(self.del_text_to_template, template_lb),
        )
        del_bt.pack(padx=5, pady=5, side=LEFT)
        notebook.add(tab_frame, text="templ")

    def send_selected_template_texts(self, listbox: Listbox):
        selected_indices: Iterable[int] = listbox.curselection()
        for i in selected_indices:
            text = listbox.get(i)
            self.send_message(f"t{text.strip()}\n")

    def add_text_to_template(self, text: Entry):
        templates: List[str] = list(self.templates.get())  # type: ignore
        templates.append(text.get())
        self.templates.set(templates)
        self.config.template_texts.clear()
        for template in templates:
            self.config.template_texts.append(template)

    def del_text_to_template(self, listbox: Listbox):
        selected_indices: Iterable[int] = listbox.curselection()
        templates: List[str] = list(self.templates.get())  # type: ignore
        for i in selected_indices:
            text = listbox.get(i)
            templates = [template for template in templates if template != text]
        self.templates.set(templates)
        self.config.template_texts.clear()
        for template in templates:
            self.config.template_texts.append(template)

    def draw_cb(
        self,
        frame: Frame,
        candidates: Dict[str, Any],
        config_name: str,
    ) -> Frame:
        inner_frame = Frame(frame)
        text = "Select " + self.get_display_name(config_name) + ":"
        label = Label(inner_frame, text=text)
        label.pack(fill=X, padx=5, pady=5)
        cb = AutocompleteCombobox[Any](inner_frame)
        cb.bind("<<ComboboxSelected>>", partial(self.set_config, cb))
        cb.set_completion_list(candidates)
        cb.pack(fill=X, padx=5, pady=5)
        self.config_entry_map[cb] = config_name
        return inner_frame

    def draw_sb(
        self,
        frame: Frame,
        from_: float,
        to: float,
        increment: float,
        config_name: str,
    ) -> Frame:
        inner_frame = Frame(frame)
        text = self.get_display_name(config_name)
        label = Label(inner_frame, text=text)
        label.pack(fill=X, padx=5, pady=5)
        sb = Spinbox(inner_frame, from_=from_, to=to, wrap=True, increment=increment)
        sb.configure(command=partial(self.set_config, sb, None))
        sb.bind("<KeyRelease>", partial(self.set_config, sb))
        sb.pack(fill=X, padx=5, pady=5)
        self.config_entry_map[sb] = config_name
        return inner_frame

    def draw_checkbutton(self, frame: Frame, config_name: str) -> Frame:
        inner_frame = Frame(frame)
        text = self.get_display_name(config_name)
        cb = Checkbutton(inner_frame, text=text)
        cb.configure(command=partial(self.set_config, cb, None))
        cb.pack(fill=X, padx=5, pady=5)
        self.config_entry_map[cb] = config_name
        return inner_frame

    def draw_tb(self, frame: Frame, config_name: str) -> Frame:
        inner_frame = Frame(frame)
        text = self.get_display_name(config_name) + ":"
        label = Label(inner_frame, text=text)
        label.pack(fill=X, padx=5, pady=5)
        tb = Textbox(inner_frame)
        tb.bind("<KeyRelease>", partial(self.set_config, tb))
        tb.pack(fill=X, padx=5, pady=5)
        self.config_entry_map[tb] = config_name
        return inner_frame

    def get_display_name(self, config_name: str) -> str:
        return f'{config_name.split(".")[-1].replace("_", " ").capitalize()}'

    def load_config(self):
        for entry, name in self.config_entry_map.items():
            *attributes, child = name.split(".")
            nest = self.config
            for attribute in attributes:
                nest = getattr(nest, attribute)
            selected_item_value = getattr(nest, child)
            selected_item_label = entry.get_label_for_item_value(selected_item_value)
            logger.info(f"{name}: {selected_item_value}")
            if selected_item_label:
                entry.set(selected_item_label)
        self.templates.set(self.config.template_texts)

    def set_config(self, widget: "Widgets", event: Any):
        value = widget.get_value()
        name = self.config_entry_map[widget]
        logger.info(f"set {name}: {value}")
        *attributes, child = name.split(".")
        nest = self.config
        for attribute in attributes:
            nest = getattr(nest, attribute)
        setattr(nest, child, value)

    def check_process_running(self):
        try:
            if not self.thread:
                return
            self.thread.wait(0.005)
            polling = self.thread.poll()
            if polling:
                if self.thread.returncode > 0:
                    messagebox.showinfo(
                        "note",
                        f"recording process stopped with returncode: {self.thread.returncode}",
                    )
                self.on_terminated()
        except TimeoutExpired:
            self.after(500, self.check_process_running)

    def run_vspeech(self):
        if self.thread:
            logger.warning(
                "process already running %s. terminating last one.", self.thread.pid
            )
            self.terminate_main()
        with NamedTemporaryFile("w", delete=False) as temp_config_file:
            temp_config_file.write(self.config.json())
            temp_config_file.flush()
            self.temp_config_file_path = Path(temp_config_file.name)
        self.thread = Popen(
            [
                sys.executable,
                "-m",
                "vspeech",
                "--json-config",
                self.temp_config_file_path,
            ],
        )
        logger.info("process started %s", self.thread.pid)
        self.on_start_process()
        self.check_process_running()

    def send_text(self, text: Text):
        if self.thread:
            lines = text.get("1.0", "end-1c").splitlines()
            for line in lines:
                self.send_message(f"t{line.strip()}\n")

    def pause_or_resume(self):
        if self.thread:
            if self.paused:
                text_var = "r\n"
            else:
                text_var = "p\n"
            self.send_message(text_var)
            self.paused = not self.paused
            self.pr_toggle_bt.configure(text="resume" if self.paused else "pause")

    def send_message(self, message: str):
        if self.thread:
            output_bytes = encode(message, getpreferredencoding(), errors="replace")
            with socket(AF_INET, SOCK_STREAM) as sock:
                sock.connect(("localhost", self.config.port))
                sock.sendall(output_bytes)
            logger.info("send: %s", message)

    def terminate_main(self):
        if self.thread:
            self.thread.terminate()
            self.thread.wait()
            logger.info("terminate %s", self.thread.pid)
            self.on_terminated()

    def reload_config(self):
        self.temp_config_file_path.unlink(missing_ok=True)
        with NamedTemporaryFile("w", delete=False) as temp_config_file:
            temp_config_file.write(self.config.json())
            temp_config_file.flush()
            self.temp_config_file_path = Path(temp_config_file.name)
        self.send_message(f"l{self.temp_config_file_path.resolve()}")

    def on_terminated(self):
        self.thread = None
        self.temp_config_file_path.unlink(missing_ok=True)
        self.paused = False
        self.pr_toggle_bt.configure(text="pause", state="disabled")
        self.send_bt.configure(state="disabled")
        self.reload_bt.configure(state="disabled")

    def on_start_process(self):
        self.pr_toggle_bt.configure(text="pause", state="active")
        self.send_bt.configure(state="active")
        self.reload_bt.configure(state="active")

    def on_close(self):
        self.terminate_main()
        self.master.destroy()


def gui():
    root = Window()
    root.title("vspeech")
    root.geometry("500x600")
    root.resizable(width=True, height=True)
    app = VspeechGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        app.mainloop()
    except Exception:
        app.terminate_main()
