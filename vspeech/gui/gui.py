import sys
from enum import Enum
from functools import partial
from logging import Handler
from logging import LogRecord
from math import floor
from pathlib import Path
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
from tkinter import StringVar
from tkinter import Tk
from tkinter import Variable
from tkinter import W
from tkinter import X
from tkinter import filedialog
from tkinter import font
from tkinter import messagebox
from typing import IO
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from typing import cast
from typing import get_type_hints

import click
import grpc
import toml
from humps import pascalize
from toml.encoder import TomlArraySeparatorEncoder
from ttkbootstrap import Button
from ttkbootstrap import Checkbutton as ttkCheckbutton
from ttkbootstrap import Entry
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Menu
from ttkbootstrap import Notebook
from ttkbootstrap import ScrolledText as ttkScrolledText
from ttkbootstrap import Spinbox as ttkSpinbox
from ttkbootstrap import Text
from ttkbootstrap import Window
from ttkbootstrap.themes.standard import STANDARD_THEMES
from vstreamer_protos.commander.commander_pb2 import PAUSE
from vstreamer_protos.commander.commander_pb2 import RELOAD
from vstreamer_protos.commander.commander_pb2 import RESUME
from vstreamer_protos.commander.commander_pb2 import SET_FILTERS
from vstreamer_protos.commander.commander_pb2 import SPEECH
from vstreamer_protos.commander.commander_pb2 import SUBTITLE
from vstreamer_protos.commander.commander_pb2 import TRANSLATE
from vstreamer_protos.commander.commander_pb2 import Command
from vstreamer_protos.commander.commander_pb2_grpc import CommanderStub

from vspeech.audio import list_all_devices
from vspeech.config import Config
from vspeech.config import ReplaceFilter
from vspeech.config import SpeechWorkerType
from vspeech.config import TranscriptionWorkerType
from vspeech.config import VoicevoxParam
from vspeech.config import VR2Param
from vspeech.gui.autocomplete_combobox import AutocompleteCombobox
from vspeech.logger import logger

try:
    from pyvcroid2 import VcRoid2

    from vspeech.gui.dummy_param import create_dummy_param

    use_vroid2 = True
except ModuleNotFoundError:
    use_vroid2 = False


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


class CommaSeparatedTextbox(Entry):
    var: StringVar

    def __init__(self, master: Any, **kw: Any):
        self.var = StringVar()
        super().__init__(master, textvariable=self.var, **kw)

    def get_value(self):
        cs_text = self.var.get()
        return [t.strip() for t in cs_text.split(",")]

    def get_label_for_item_value(self, value: Any):
        return value

    def set(self, value: Any):
        if isinstance(value, Iterable):
            value = cast(Iterable[Any], value)
            self.var.set(",".join([str(v) for v in value]))
        else:
            self.var.set(value)


Widgets = Union[
    AutocompleteCombobox[Any], Spinbox, Checkbutton, Textbox, CommaSeparatedTextbox
]


class CustomTomlEncoder(TomlArraySeparatorEncoder):  # type: ignore
    def dump_value(self, v: Any) -> str:
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, Enum):
            v = v.value
        return super().dump_value(v)


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
    stop_bt: Button
    templates: Variable
    filters: Variable

    @staticmethod
    def get_operation_from_str(ope_str: str):
        if ope_str.upper() == "TRANSLATE":
            return TRANSLATE
        if ope_str.upper() == "SUBTITLE":
            return SUBTITLE
        if ope_str.upper() == "SPEECH":
            return SPEECH

    @staticmethod
    def is_file_json(file_path: Union[str, Path]):
        file_name = str(file_path)
        return file_name.endswith(".json")

    def config_to_toml_str(self):
        return toml.dumps(self.config.dict())

    def write_config_to_file(self, file: IO[bytes]):
        file_name = file.name
        if VspeechGUI.is_file_json(file_name):
            file.write(bytes(self.config.json(), encoding="utf-8"))
        else:
            file.write(bytes(toml.dumps(self.config.dict(), encoder=CustomTomlEncoder(dict, separator="\n")), encoding="utf-8"))  # type: ignore

    def read_config_from_file(self, file: IO[bytes]):
        self.config = Config.read_config_from_file(file)

    def save_config_as(self):
        file = filedialog.asksaveasfile(
            "wb",
            filetypes=(("TOML", "*.toml"), ("JSON", "*.json"), ("all files", "*.*")),
            defaultextension=".toml",
        )
        if not file:
            return
        file_name = file.name
        self.write_config_to_file(file)
        file.close()
        logger.info("保存しました: %s", file_name)
        self.config_file_path = Path(file_name)

    def save_config(self):
        with self.config_file_path.open("wb") as file:
            self.write_config_to_file(file)
            logger.info("保存しました: %s", file.name)
            self.master.title(f"vspeech: {self.config_file_path}")

    def load_file(self):
        file = filedialog.askopenfile(
            "rb",
            filetypes=(("TOML", "*.toml"), ("JSON", "*.json"), ("all files", "*.*")),
            defaultextension=".toml",
        )
        if not file:
            return
        file_name = file.name
        self.read_config_from_file(file)
        file.close()
        self.update_gui_value_from_config()
        logger.info("ロードしました: %s", file_name)
        self.config_file_path = Path(file_name)
        self.master.title(f"vspeech: {self.config_file_path}")

    def __init__(self, master: Tk, config_file_path: str):
        super().__init__(master)
        self.master = master
        self.thread = None
        self.config_entry_map = {}

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

        self.config_file_path = Path(config_file_path)
        with open(self.config_file_path, "rb") as file:
            self.read_config_from_file(file)
        self.master.title(f"vspeech: {self.config_file_path}")

        notebook = Notebook(parameter_frame)
        notebook.pack(fill=X)
        self.draw_rec_tab(notebook=notebook)
        self.draw_playback_tab(notebook=notebook)
        self.draw_transcription_tab(notebook=notebook)
        self.draw_speech_tab(notebook=notebook)
        self.draw_subtitle_tab(notebook=notebook)
        self.draw_translation_tab(notebook=notebook)
        if use_vroid2:
            self.draw_vr2_tab(notebook=notebook)
        self.draw_voicevox_tab(notebook=notebook)
        self.draw_ami_tab(notebook=notebook)
        self.draw_gcp_tab(notebook=notebook)
        self.draw_whisper_tab(notebook=notebook)
        self.draw_template_text_tab(notebook=notebook)
        self.draw_filter_tab(notebook=notebook)

        run_bt = Button(bt_frame, text="run", command=self.run_vspeech)
        run_bt.pack(padx=5, pady=5, side=LEFT)
        self.pr_toggle_bt = Button(
            bt_frame,
            text="pause",
            command=self.pause_or_resume,
            state="disabled",
        )
        self.pr_toggle_bt.pack(padx=5, pady=5, side=LEFT)
        self.stop_bt = Button(
            bt_frame, text="stop", command=self.terminate_main, state="disabled"
        )
        self.stop_bt.pack(padx=5, pady=5, side=LEFT)
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

        self.update_gui_value_from_config()

    def draw_rec_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "recording"
        input_devices = list_all_devices(input=True)
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        self.draw_cb(
            frame=tab_frame,
            candidates=input_devices,
            config_name=f"{prefix}.input_device_index",
        ).grid(column=0, row=current_row, columnspan=max_columns, sticky=EW)
        current_row += 1
        cb = cast(
            AutocompleteCombobox[Any],
            self.get_entry_from_config(f"{prefix}.input_device_index"),
        )
        add_bt = Button(
            tab_frame,
            text="reload devices",
            command=partial(self.update_completion_list, cb, True, False),
        )
        add_bt.grid(padx=5, pady=5, column=0, row=current_row, sticky=W)
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.silence_threshold",
            from_=-120,
            to=0,
            increment=1,
        ).grid(column=0, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.record_interval_sec",
            from_=0,
            to=10,
            increment=0.1,
        ).grid(column=1, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.chunk",
            from_=0,
            to=48000,
            increment=1,
        ).grid(column=2, row=current_row, sticky=EW)
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.rate",
            from_=0,
            to=48000,
            increment=1,
        ).grid(column=0, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.max_recording_sec",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=1, row=current_row, sticky=EW)
        self.draw_cs_tb(tab_frame, config_name=f"{prefix}.destinations").grid(
            column=0, row=current_row, sticky=EW, columnspan=max_columns
        )
        notebook.add(tab_frame, text="rec")

    def draw_speech_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "speech"
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        worker_types = {
            worker_type.name: worker_type for worker_type in SpeechWorkerType
        }
        self.draw_cb(
            frame=tab_frame,
            candidates=worker_types,
            config_name=f"{prefix}.speech_worker_type",
        ).grid(column=0, row=current_row, columnspan=max_columns, sticky=EW)
        current_row += 1
        self.draw_cs_tb(tab_frame, config_name=f"{prefix}.destinations").grid(
            column=0, row=current_row, sticky=EW, columnspan=max_columns
        )
        notebook.add(tab_frame, text="tts")

    def draw_playback_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "playback"
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        output_devices = list_all_devices(output=True)
        self.draw_cb(
            frame=tab_frame,
            candidates=output_devices,
            config_name=f"{prefix}.output_device_index",
        ).grid(column=0, row=current_row, columnspan=max_columns, sticky=EW)
        current_row += 1
        cb = cast(
            AutocompleteCombobox[Any],
            self.get_entry_from_config(f"{prefix}.output_device_index"),
        )
        add_bt = Button(
            tab_frame,
            text="reload devices",
            command=partial(self.update_completion_list, cb, False, True),
        )
        add_bt.grid(padx=5, pady=5, column=0, row=current_row, sticky=W)
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.speech_volume",
            from_=0,
            to=100,
            increment=1,
        ).grid(column=0, row=current_row, sticky=EW)
        notebook.add(tab_frame, text="play")

    def draw_subtitle_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 6
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "subtitle"
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        font_cb_list = sorted(font.families())
        self.draw_cb(
            frame=tab_frame,
            candidates={font: font for font in font_cb_list},
            config_name=f"{prefix}.text.font_family",
        ).grid(column=0, row=current_row, columnspan=floor(max_columns / 2), sticky=EW)
        self.draw_cb(
            frame=tab_frame,
            candidates={font: font for font in font_cb_list},
            config_name=f"{prefix}.translated.font_family",
        ).grid(
            column=floor(max_columns / 2),
            row=current_row,
            columnspan=floor(max_columns / 2),
            sticky=EW,
        )
        current_row += 1
        self.draw_tb(tab_frame, config_name=f"{prefix}.text.font_color").grid(
            column=0, row=current_row, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.text.outline_color").grid(
            column=1, row=current_row, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.text.font_size",
            from_=0,
            to=255,
            increment=1,
        ).grid(column=2, row=current_row, sticky=EW)
        self.draw_tb(tab_frame, config_name=f"{prefix}.translated.font_color").grid(
            column=3, row=current_row, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.translated.outline_color").grid(
            column=4, row=current_row, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.translated.font_size",
            from_=0,
            to=255,
            increment=1,
        ).grid(column=5, row=current_row, sticky=EW)
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.text.min_display_sec",
            from_=0,
            to=10,
            increment=0.1,
        ).grid(column=0, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.text.max_text_len",
            from_=0,
            to=100,
            increment=1,
        ).grid(column=1, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.text.display_sec_per_letter",
            from_=0,
            to=1,
            increment=0.01,
        ).grid(column=2, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.translated.min_display_sec",
            from_=0,
            to=10,
            increment=0.1,
        ).grid(column=3, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.translated.max_text_len",
            from_=0,
            to=100,
            increment=1,
        ).grid(column=4, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.translated.display_sec_per_letter",
            from_=0,
            to=1,
            increment=0.01,
        ).grid(column=5, row=current_row, sticky=EW)
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.subtitle_window_width",
            from_=0,
            to=65535,
            increment=1,
        ).grid(column=0, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.subtitle_window_height",
            from_=0,
            to=65535,
            increment=1,
        ).grid(column=1, row=current_row, sticky=EW)
        self.draw_tb(tab_frame, config_name=f"{prefix}.subtitle_bg_color").grid(
            column=2, row=current_row, sticky=EW
        )
        notebook.add(tab_frame, text="sub")

    def draw_transcription_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 2
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "transcription"
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        worker_types = {
            worker_type.name: worker_type for worker_type in TranscriptionWorkerType
        }
        self.draw_cb(
            frame=tab_frame,
            candidates=worker_types,
            config_name=f"{prefix}.transcription_worker_type",
        ).grid(column=0, row=current_row, columnspan=max_columns, sticky=EW)
        current_row += 1
        self.draw_cs_tb(tab_frame, config_name=f"{prefix}.destinations").grid(
            column=0, row=current_row, sticky=EW, columnspan=max_columns
        )
        notebook.add(tab_frame, text="transc")

    def draw_translation_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "translation"
        current_row = 0
        self.draw_checkbutton(frame=tab_frame, config_name=f"{prefix}.enable").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=W
        )
        current_row += 1
        self.draw_tb(tab_frame, config_name=f"{prefix}.source_language_code").grid(
            column=0, row=current_row, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.target_language_code").grid(
            column=1, row=current_row, sticky=EW
        )
        current_row += 1
        self.draw_cs_tb(tab_frame, config_name=f"{prefix}.destinations").grid(
            column=0, row=current_row, sticky=EW, columnspan=max_columns
        )
        notebook.add(tab_frame, text="transl")

    def draw_ami_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 2
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "ami"
        current_row = 0
        self.draw_tb(tab_frame, config_name=f"{prefix}.ami_appkey").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=EW
        )
        current_row += 1
        self.draw_tb(tab_frame, config_name=f"{prefix}.ami_engine_uri").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=EW
        )
        current_row += 1
        self.draw_tb(tab_frame, config_name=f"{prefix}.ami_engine_name").grid(
            column=0, row=current_row, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.ami_service_id").grid(
            column=1, row=current_row, sticky=EW
        )
        current_row += 1
        self.draw_tb(tab_frame, config_name=f"{prefix}.ami_extra_parameters").grid(
            column=0, row=current_row, sticky=EW, columnspan=max_columns
        )
        notebook.add(tab_frame, text="ami")

    def draw_gcp_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "gcp"
        self.draw_tb(tab_frame, config_name=f"{prefix}.gcp_project_id").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_tb(tab_frame, config_name=f"{prefix}.gcp_credentials_file_path").grid(
            column=0, row=1, columnspan=max_columns, sticky=EW
        )
        notebook.add(tab_frame, text="gcp")

    def draw_whisper_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 3
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "whisper"
        current_row = 0
        self.draw_tb(tab_frame, config_name=f"{prefix}.whisper_model").grid(
            column=0, row=current_row, columnspan=max_columns, sticky=EW
        )
        current_row += 1
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.whisper_no_speech_prob_threshold",
            from_=0,
            to=1,
            increment=0.01,
        ).grid(column=0, row=current_row, sticky=EW)
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.whisper_logprob_threshold",
            from_=float("-inf"),
            to=1,
            increment=0.01,
        ).grid(column=1, row=current_row, sticky=EW)
        notebook.add(tab_frame, text="whisper")

    def draw_vr2_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 4
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        prefix = "vr2"
        with VcRoid2() as vr2:
            voice_lists = vr2.listVoices()
            self.draw_cb(
                frame=tab_frame,
                candidates={v: v for v in voice_lists},
                config_name=f"{prefix}.vr2_voice_name",
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
                        config_name=f"{prefix}.vr2_params.{param_name}",
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
        prefix = "voicevox"
        self.draw_tb(tab_frame, config_name=f"{prefix}.openjtalk_dir").grid(
            column=0, row=0, columnspan=max_columns, sticky=EW
        )
        self.draw_sb(
            frame=tab_frame,
            config_name=f"{prefix}.voicevox_speaker_id",
            from_=0,
            to=10,
            increment=1,
        ).grid(column=0, row=1, sticky=EW)
        params = list(get_type_hints(VoicevoxParam).keys())
        chunked_list: List[List[str]] = list()
        for i in range(0, len(params), max_columns):
            chunked_list.append(params[i : i + max_columns])
        for row, params_chunk in enumerate(chunked_list):
            for column, param_name in enumerate(params_chunk):
                self.draw_sb(
                    frame=tab_frame,
                    config_name=f"{prefix}.voicevox_params.{param_name}",
                    from_=-1.0,
                    to=2.0,
                    increment=0.01,
                ).grid(column=column, row=row + 2, sticky=EW)

        notebook.add(tab_frame, text="vvox")

    def draw_template_text_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 1
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        current_row = 0
        text_candidate = Entry(tab_frame)
        text_candidate.grid(padx=5, pady=5, column=0, row=current_row, sticky=EW)
        current_row += 1
        add_bt = Button(
            tab_frame,
            text="add",
            command=partial(self.add_text_to_template, text_candidate),
        )
        add_bt.grid(padx=5, pady=5, column=0, row=current_row, sticky=W)
        current_row += 1
        texts = self.config.template_texts
        self.templates = Variable(value=texts)
        template_lb = Listbox(tab_frame, listvariable=self.templates, height=6)
        template_lb.grid(padx=5, pady=5, column=0, row=current_row, sticky=EW)
        current_row += 1
        button_frame = Frame(tab_frame)
        button_frame.grid(column=0, row=current_row, sticky=EW)
        current_row += 1
        send_bt = Button(
            button_frame,
            text="send",
            command=partial(self.send_selected_template_texts, template_lb),
        )
        send_bt.pack(padx=5, pady=5, side=LEFT)
        del_bt = Button(
            button_frame,
            text="del",
            command=partial(self.del_text_from_template, template_lb),
        )
        del_bt.pack(padx=5, pady=5, side=LEFT)
        self.draw_tb(
            frame=tab_frame,
            config_name=f"listen_address",
        ).grid(column=0, row=current_row, sticky=EW)
        current_row += 1
        self.draw_cs_tb(
            frame=tab_frame,
            config_name=f"text_send_operations",
        ).grid(column=0, row=current_row, sticky=EW)
        notebook.add(tab_frame, text="templ")

    def draw_filter_tab(self, notebook: Notebook):
        tab_frame = Frame(self)
        tab_frame.pack(fill=X)
        max_columns = 1
        for i in range(max_columns):
            tab_frame.columnconfigure(i, weight=1)
        current_row = 0
        text = "Filters:"
        label = Label(tab_frame, text=text)
        label.grid(padx=5, pady=5, column=0, row=current_row, sticky=W)
        current_row += 1
        filter_candidate = Entry(tab_frame)
        filter_candidate.grid(
            padx=5, pady=5, column=0, row=current_row, columnspan=max_columns, sticky=EW
        )
        current_row += 1
        add_bt = Button(
            tab_frame,
            text="add",
            command=partial(self.add_filter, filter_candidate),
        )
        add_bt.grid(padx=5, pady=5, column=0, row=current_row, sticky=W)
        current_row += 1
        filters = self.config.filters
        self.filters = Variable(value=[str(f) for f in filters])
        filter_lb = Listbox(tab_frame, listvariable=self.filters, height=6)
        filter_lb.grid(
            padx=5, pady=5, column=0, row=current_row, columnspan=max_columns, sticky=EW
        )
        current_row += 1
        del_bt = Button(
            tab_frame,
            text="del",
            command=partial(self.del_filter, filter_lb),
        )
        del_bt.grid(column=0, row=current_row, padx=5, pady=5, sticky=W)
        notebook.add(tab_frame, text="filt")

    def operations_for_send_text(self):
        operations = [
            VspeechGUI.get_operation_from_str(o)
            for o in self.config.text_send_operations
        ]
        operations = [o for o in operations if o]
        return operations

    def send_selected_template_texts(self, listbox: Listbox):
        selected_indices = cast(Iterable[int], listbox.curselection())
        for i in selected_indices:
            text = cast(str, listbox.get(i))
            self.send_message(
                Command(
                    operations=self.operations_for_send_text(),
                    text=text.strip(),
                )
            )

    def add_text_to_template(self, text: Entry):
        templates: List[str] = list(self.templates.get())  # type: ignore
        templates.append(text.get())
        self.templates.set(templates)
        self.config.template_texts.clear()
        for template in templates:
            self.config.template_texts.append(template)
        self.master.title(f"vspeech: {self.config_file_path}*")

    def del_text_from_template(self, listbox: Listbox):
        selected_indices = cast(Iterable[int], listbox.curselection())
        templates: List[str] = list(self.templates.get())  # type: ignore
        for i in selected_indices:
            text = listbox.get(i)
            templates = [template for template in templates if template != text]
        self.templates.set(templates)
        self.config.template_texts.clear()
        for template in templates:
            self.config.template_texts.append(template)
        self.master.title(f"vspeech: {self.config_file_path}*")

    def add_filter(self, text: Entry):
        value = text.get()
        try:
            rf = ReplaceFilter.from_str(value)
        except ValueError:
            logger.warning("%s 無効な値です。", value)
            return
        filters: List[str] = list(self.filters.get())  # type: ignore
        filters.append(str(rf))
        self.filters.set(filters)
        self.config.filters.clear()
        for filter in filters:
            self.config.filters.append(ReplaceFilter.from_str(filter))
        self.send_message(Command(operations=[SET_FILTERS], filters=filters))
        self.master.title(f"vspeech: {self.config_file_path}*")

    def del_filter(self, listbox: Listbox):
        selected_indices = cast(Iterable[int], listbox.curselection())
        filters: List[str] = list(self.filters.get())  # type: ignore
        for i in selected_indices:
            text = listbox.get(i)
            filters = [filter for filter in filters if filter != text]
        self.filters.set(filters)
        self.config.filters.clear()
        for filter in filters:
            self.config.filters.append(ReplaceFilter.from_str(filter))
        self.send_message(Command(operations=[SET_FILTERS], filters=filters))
        self.master.title(f"vspeech: {self.config_file_path}*")

    def update_completion_list(
        self, cb: AutocompleteCombobox[Any], input: bool, output: bool
    ):
        devices = list_all_devices(input=input, output=output)
        cb.set_completion_list(devices)

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

    def draw_cs_tb(self, frame: Frame, config_name: str) -> Frame:
        inner_frame = Frame(frame)
        text = self.get_display_name(config_name) + ":"
        label = Label(inner_frame, text=text)
        label.pack(fill=X, padx=5, pady=5)
        tb = CommaSeparatedTextbox(inner_frame)
        tb.bind("<KeyRelease>", partial(self.set_config, tb))
        tb.pack(fill=X, padx=5, pady=5)
        self.config_entry_map[tb] = config_name
        return inner_frame

    def get_display_name(self, config_name: str) -> str:
        return f'{config_name.split(".")[-1].replace("_", " ").capitalize()}'

    def get_entry_from_config(self, config_name: str):
        for entry, name in self.config_entry_map.items():
            if name == config_name:
                return entry
        raise AttributeError(config_name)

    def update_gui_value_from_config(self):
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
        self.filters.set([str(f) for f in self.config.filters])

    def set_config(self, widget: "Widgets", event: Any):
        value = widget.get_value()
        name = self.config_entry_map[widget]
        logger.info(f"set {name}: {value}")
        *attributes, child = name.split(".")
        nest = self.config
        for attribute in attributes:
            nest = getattr(nest, attribute)
        setattr(nest, child, value)
        self.master.title(f"vspeech: {self.config_file_path}*")

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
        with NamedTemporaryFile("w", delete=False, suffix=".json") as temp_config_file:
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
                self.send_message(
                    Command(
                        operations=self.operations_for_send_text(),
                        text=line.strip(),
                    )
                )

    def pause_or_resume(self):
        if self.thread:
            if self.paused:
                self.send_message(Command(operations=[RESUME]))
            else:
                self.send_message(Command(operations=[PAUSE]))
            self.paused = not self.paused
            self.pr_toggle_bt.configure(text="resume" if self.paused else "pause")

    def send_message(self, command: Command):
        if self.thread:
            with grpc.insecure_channel(self.config.listen_address) as channel:
                stub = CommanderStub(channel)
                stub.process_command(command)
                logger.info("send: %s", command)

    def terminate_main(self):
        if self.thread:
            self.thread.terminate()
            self.thread.wait()
            logger.info("terminate %s", self.thread.pid)
            self.on_terminated()

    def reload_config(self):
        self.temp_config_file_path.unlink(missing_ok=True)
        with NamedTemporaryFile("w", delete=False, suffix=".json") as temp_config_file:
            temp_config_file.write(self.config.json())
            temp_config_file.flush()
            self.temp_config_file_path = Path(temp_config_file.name)
        self.send_message(
            Command(
                operations=[RELOAD], file_path=str(self.temp_config_file_path.resolve())
            )
        )

    def on_terminated(self):
        self.thread = None
        self.temp_config_file_path.unlink(missing_ok=True)
        self.paused = False
        self.pr_toggle_bt.configure(text="pause", state="disabled")
        self.send_bt.configure(state="disabled")
        self.reload_bt.configure(state="disabled")
        self.stop_bt.configure(state="disabled")

    def on_start_process(self):
        self.pr_toggle_bt.configure(text="pause", state="active")
        self.send_bt.configure(state="active")
        self.reload_bt.configure(state="active")
        self.stop_bt.configure(state="active")

    def on_close(self):
        self.terminate_main()
        self.master.destroy()


@click.command()
@click.option("-c", "--config", default="config.toml")
@click.option(
    "-t", "--theme", default="cosmo", type=click.Choice(list(STANDARD_THEMES.keys()))
)
def gui(config: str, theme: str):
    root = Window(themename=theme)
    root.title("vspeech")
    root.geometry("500x600")
    root.resizable(width=True, height=True)
    app = VspeechGUI(root, config)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        app.mainloop()
    except Exception:
        app.terminate_main()
