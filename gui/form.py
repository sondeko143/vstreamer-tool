from collections.abc import Callable
from functools import partial
from tkinter import W
from tkinter import X
from typing import Any

from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Labelframe

from gui.autocomplete_combobox import AutocompleteCombobox
from gui.widgets import Checkbutton
from gui.widgets import Spinbox
from gui.widgets import Textbox
from vspeech.config import Config
from vspeech.config import TranscriptionWorkerType
from vspeech.config import TtsWorkerType

try:
    from vspeech.lib.audio import list_all_devices
except Exception:  # audio extra not installed

    def list_all_devices(input: bool = False, output: bool = False) -> dict[str, int]:
        return {}


def _get(config: Config, path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _set(config: Config, path: str, value: Any) -> None:
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, child, value)


class PipelineForm(Frame):
    # Explicit class-level annotation: `tkinter.Misc` aliases `config` to
    # `configure`, so this must be declared here (not just inferred from an
    # `__init__` assignment) to override the inherited method's type for the
    # whole class rather than only within `__init__`.
    config: Config | None

    def __init__(self, master: Any, on_change: Callable[[], None]):
        super().__init__(master)
        self.on_change = on_change
        self.config = None
        # widget -> (config_path, coerce fn from widget value to config value)
        self.bindings: dict[Any, tuple[str, Callable[[Any], Any]]] = {}
        self.body = Frame(self)
        self.body.pack(fill=X)

    def bind_config(self, config: Config) -> None:
        self.config = config
        for child in list(self.body.children.values()):
            child.destroy()
        self.bindings.clear()
        self._section_recording()
        self._section_playback()
        self._section_transcription()
        self._section_tts()
        self._section_vc()

    def read_into(self, config: Config) -> None:
        for widget, (path, coerce) in self.bindings.items():
            try:
                _set(config, path, coerce(widget.get_value()))
            except ValueError, KeyError:
                continue

    # --- field builders -------------------------------------------------

    def _check(self, parent: Any, path: str, label: str) -> Checkbutton:
        assert self.config is not None
        widget = Checkbutton(parent, text=label)
        widget.set(_get(self.config, path))
        widget.configure(command=self.on_change)
        self.bindings[widget] = (path, bool)
        return widget

    def _entry(self, parent: Any, path: str, label: str) -> Frame:
        assert self.config is not None
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        widget = Textbox(frame)
        widget.set(_get(self.config, path))
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, str)
        widget.pack(fill=X)
        return frame

    def _spin(
        self, parent: Any, path: str, label: str, from_: float, to: float, inc: float
    ) -> Frame:
        assert self.config is not None
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        widget = Spinbox(frame, from_=from_, to=to, increment=inc, wrap=True)
        widget.set(_get(self.config, path))
        coerce = int if float(inc).is_integer() else float
        widget.configure(command=self.on_change)
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, coerce)
        widget.pack(fill=X)
        return frame

    def _device_combo(
        self, parent: Any, path: str, label: str, *, input: bool
    ) -> Frame:
        assert self.config is not None
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X)
        combo = AutocompleteCombobox[int](frame)
        combo.set_completion_list(list_all_devices(input=input, output=not input))
        current = _get(self.config, path)
        combo_label = (
            combo.get_label_for_item_value(current) if current is not None else None
        )
        if combo_label:
            combo.set(combo_label)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.on_change())
        self.bindings[combo] = (path, lambda v: v)
        combo.pack(fill=X)
        return frame

    def _enum_combo(
        self, parent: Any, path: str, label: str, enum_cls: Any
    ) -> AutocompleteCombobox[Any]:
        assert self.config is not None
        Label(parent, text=label).pack(fill=X)
        combo = AutocompleteCombobox[Any](parent)
        combo.set_completion_list({member.name: member for member in enum_cls})
        current = _get(self.config, path)
        combo.set(current.name)
        self.bindings[combo] = (path, lambda v: v)
        combo.pack(fill=X)
        return combo

    # --- worker sections ------------------------------------------------

    def _section_recording(self) -> None:
        box = Labelframe(self.body, text="recording")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "recording.enable", "enable recording").pack(anchor=W)
        self._device_combo(
            box, "recording.input_device_index", "input device", input=True
        ).pack(fill=X)
        self._spin(box, "recording.rate", "rate", 8000, 48000, 1).pack(fill=X)
        self._spin(
            box, "recording.silence_threshold", "silence threshold (dBFS)", -120, 0, 1
        ).pack(fill=X)

    def _section_playback(self) -> None:
        box = Labelframe(self.body, text="playback")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "playback.enable", "enable playback").pack(anchor=W)
        self._device_combo(
            box, "playback.output_device_index", "output device", input=False
        ).pack(fill=X)
        self._spin(box, "playback.volume", "volume", 0, 100, 1).pack(fill=X)

    def _section_transcription(self) -> None:
        box = Labelframe(self.body, text="transcription")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "transcription.enable", "enable transcription").pack(anchor=W)
        combo = self._enum_combo(
            box, "transcription.worker_type", "worker_type", TranscriptionWorkerType
        )
        backend = Frame(box)
        backend.pack(fill=X)
        combo.bind(
            "<<ComboboxSelected>>",
            partial(self._rebuild_transcription_backend, backend, combo),
            add="+",
        )
        self._rebuild_transcription_backend(backend, combo, None)

    def _rebuild_transcription_backend(
        self, backend: Frame, combo: Any, _event: Any
    ) -> None:
        for child in list(backend.children.values()):
            self.bindings.pop(child, None)
            child.destroy()
        self.on_change()
        try:
            worker_type = combo.get_value()
        except KeyError:
            return
        if worker_type == TranscriptionWorkerType.WHISPER:
            self._entry(backend, "whisper.model", "whisper model").pack(fill=X)
            self._spin(backend, "whisper.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
        elif worker_type == TranscriptionWorkerType.GCP:
            self._entry(
                backend, "gcp.service_account_file_path", "gcp key.json path"
            ).pack(fill=X)
        elif worker_type == TranscriptionWorkerType.ACP:
            self._entry(backend, "ami.appkey", "ami appkey").pack(fill=X)
            self._entry(backend, "ami.engine_uri", "ami engine_uri").pack(fill=X)
            self._entry(backend, "ami.engine_name", "ami engine_name").pack(fill=X)
            self._entry(backend, "ami.service_id", "ami service_id").pack(fill=X)

    def _section_tts(self) -> None:
        box = Labelframe(self.body, text="tts")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "tts.enable", "enable tts").pack(anchor=W)
        combo = self._enum_combo(box, "tts.worker_type", "worker_type", TtsWorkerType)
        backend = Frame(box)
        backend.pack(fill=X)
        combo.bind(
            "<<ComboboxSelected>>",
            partial(self._rebuild_tts_backend, backend, combo),
            add="+",
        )
        self._rebuild_tts_backend(backend, combo, None)

    def _rebuild_tts_backend(self, backend: Frame, combo: Any, _event: Any) -> None:
        for child in list(backend.children.values()):
            self.bindings.pop(child, None)
            child.destroy()
        self.on_change()
        try:
            worker_type = combo.get_value()
        except KeyError:
            return
        if worker_type == TtsWorkerType.VOICEVOX:
            self._entry(backend, "voicevox.openjtalk_dir", "openjtalk_dir").pack(fill=X)
            self._entry(backend, "voicevox.model_dir", "model_dir").pack(fill=X)
            self._entry(backend, "voicevox.onnxruntime_path", "onnxruntime_path").pack(
                fill=X
            )
            self._spin(backend, "voicevox.speaker_id", "speaker_id", 0, 100, 1).pack(
                fill=X
            )
        elif worker_type == TtsWorkerType.VR2:
            self._entry(backend, "vr2.voice_name", "voice_name").pack(fill=X)

    def _section_vc(self) -> None:
        box = Labelframe(self.body, text="vc")
        box.pack(fill=X, padx=4, pady=4)
        self._check(box, "vc.enable", "enable vc").pack(anchor=W)
        self._entry(box, "rvc.model_file", "rvc model_file").pack(fill=X)
        self._entry(box, "rvc.hubert_model_file", "hubert asset dir").pack(fill=X)
        self._entry(box, "rvc.rmvpe_model_file", "rmvpe model_file").pack(fill=X)
        self._spin(box, "rvc.f0_up_key", "f0_up_key", -64, 64, 1).pack(fill=X)
        self._spin(box, "rvc.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
