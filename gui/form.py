from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import BOTH
from tkinter import TclError
from tkinter import W
from tkinter import X
from typing import Any
from typing import get_args

from pydantic import SecretStr
from ttkbootstrap import Frame
from ttkbootstrap import Label
from ttkbootstrap import Labelframe
from ttkbootstrap.widgets.scrolled import ScrolledFrame

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
    if isinstance(node, SecretStr):
        return node.get_secret_value()
    return node


def _coerce_value(node: Any, child: str, value: Any) -> Any:
    # None passes through cleanly (read_into only sends None for Optional
    # fields). Blank strings never reach here — read_into intercepts them — so
    # Path/SecretStr always wrap a real value.
    if value is None:
        return None
    annotation = type(node).model_fields[child].annotation
    types = get_args(annotation) or (annotation,)
    if SecretStr in types:
        return SecretStr(str(value))
    if Path in types:
        return Path(str(value))
    return value


def _field_types(config: Config, path: str) -> tuple[Any, ...]:
    """The pydantic annotation of `path`'s field, flattened to a tuple of types
    (e.g. `int | None` -> `(int, NoneType)`, a bare `int` -> `(int,)`)."""
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    annotation = type(node).model_fields[child].annotation
    return get_args(annotation) or (annotation,)


def _set(config: Config, path: str, value: Any) -> None:
    *parents, child = path.split(".")
    node: Any = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, child, _coerce_value(node, child, value))


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
        # (box, enable-checkbutton) per worker section, so the section's fields
        # can be enabled/disabled to follow its `enable` toggle.
        self._section_enables: list[tuple[Any, Checkbutton]] = []
        # A scrollable body so the full field list is reachable on a small
        # window; the vertical scrollbar shows whenever the content overflows.
        self.body = ScrolledFrame(self, autohide=False)
        self.body.pack(fill=BOTH, expand=True)

    def bind_config(self, config: Config) -> None:
        self.config = config
        for child in list(self.body.children.values()):
            child.destroy()
        self.bindings.clear()
        self._section_enables = []
        self._section_recording()
        self._section_playback()
        self._section_transcription()
        self._section_tts()
        self._section_vc()
        self._apply_all_enables()

    # --- enable-driven disabling ----------------------------------------

    def _register_section(self, box: Any, enable: Checkbutton) -> None:
        # The enable checkbox drives whether the rest of the section is editable.
        enable.configure(command=lambda: (self.on_change(), self._apply_all_enables()))
        self._section_enables.append((box, enable))

    def _apply_all_enables(self) -> None:
        for box, enable in self._section_enables:
            self._apply_enable(box, enable)

    def _apply_enable(self, box: Any, enable: Checkbutton) -> None:
        state = "normal" if enable.get_value() else "disabled"
        for widget in self._descendants(box):
            if widget is enable:
                continue
            try:
                widget.configure(state=state)
            except TclError:
                # containers (Frame) have no `state` option — skip; their
                # children are handled by the recursion.
                pass

    def _descendants(self, parent: Any) -> list[Any]:
        result: list[Any] = []
        for child in parent.winfo_children():
            result.append(child)
            result.extend(self._descendants(child))
        return result

    def read_into(self, config: Config) -> list[str]:
        # Returns the dotted paths of any fields whose widget value could not be
        # coerced/applied (e.g. a non-numeric spinbox, or a garbled combo). The
        # caller surfaces these instead of the old silent `continue`, so a
        # dropped edit is visible rather than vanishing.
        failed: list[str] = []
        for widget, (path, coerce) in self.bindings.items():
            if not widget.winfo_exists():
                continue
            value = widget.get_value()
            if value is None or (isinstance(value, str) and not value.strip()):
                # A blank widget: an unselected combo (None) or a cleared/
                # whitespace-only field. Classify by the field's type:
                types = _field_types(config, path)
                if type(None) in types:
                    # Optional -> clear to None. Leaves a fresh unset field
                    # unchanged AND lets the user clear a previously-set optional
                    # field (device index, gpu_id, optional path).
                    _set(config, path, None)
                elif str in types or SecretStr in types:
                    # A str/SecretStr field where "" is itself a valid value
                    # (e.g. the ami.* ACP fields default to blank) — apply it,
                    # don't cry wolf.
                    _set(config, path, "")
                else:
                    # Required numeric / enum / path — blank genuinely can't be
                    # coerced, so surface it instead of keeping the stale value.
                    failed.append(path)
                continue
            try:
                _set(config, path, coerce(value))
            except ValueError, KeyError:
                failed.append(path)
        return failed

    # --- field builders -------------------------------------------------

    def _check(self, parent: Any, path: str, label: str) -> Checkbutton:
        assert self.config is not None  # nosec B101
        widget = Checkbutton(parent, text=label)
        widget.set(_get(self.config, path))
        widget.configure(command=self.on_change)
        self.bindings[widget] = (path, bool)
        return widget

    def _entry(self, parent: Any, path: str, label: str) -> Frame:
        assert self.config is not None  # nosec B101
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X, pady=(6, 0))
        widget = Textbox(frame)
        widget.set(_get(self.config, path))
        widget.bind("<KeyRelease>", lambda _e: self.on_change())
        self.bindings[widget] = (path, str)
        widget.pack(fill=X)
        return frame

    def _spin(
        self, parent: Any, path: str, label: str, from_: float, to: float, inc: float
    ) -> Frame:
        assert self.config is not None  # nosec B101
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X, pady=(6, 0))
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
        assert self.config is not None  # nosec B101
        devices = list_all_devices(input=input, output=not input)
        frame = Frame(parent)
        Label(frame, text=label).pack(fill=X, pady=(6, 0))
        current = _get(self.config, path)
        if devices:
            combo = AutocompleteCombobox[int](frame)
            combo.set_completion_list(devices)
            combo_label = (
                combo.get_label_for_item_value(current) if current is not None else None
            )
            if combo_label:
                combo.set(combo_label)
            combo.bind("<<ComboboxSelected>>", lambda _e: self.on_change())
            self.bindings[combo] = (path, lambda v: v)
            combo.pack(fill=X)
        else:
            entry = Textbox(frame)
            entry.set("" if current is None else str(current))
            entry.bind("<KeyRelease>", lambda _e: self.on_change())
            self.bindings[entry] = (
                path,
                lambda v: int(v) if str(v).strip() else None,
            )
            entry.pack(fill=X)
        return frame

    def _enum_combo(
        self, parent: Any, path: str, label: str, enum_cls: Any
    ) -> AutocompleteCombobox[Any]:
        assert self.config is not None  # nosec B101
        Label(parent, text=label).pack(fill=X, pady=(6, 0))
        combo = AutocompleteCombobox[Any](parent)
        combo.set_completion_list({member.name: member for member in enum_cls})
        current = _get(self.config, path)
        combo.set(current.name)
        self.bindings[combo] = (path, lambda v: v)
        combo.pack(fill=X)
        return combo

    # --- worker sections ------------------------------------------------

    def _section_box(self, title: str) -> Labelframe:
        # One padded, clearly-separated box per worker: internal padding keeps
        # fields off the border, and the outer margin (esp. left/right) plus the
        # gap below separates adjacent sections.
        box = Labelframe(self.body, text=title, padding=12)
        box.pack(fill=X, padx=12, pady=(2, 10))
        return box

    def _section_recording(self) -> None:
        box = self._section_box("recording")
        enable = self._check(box, "recording.enable", "enable recording")
        enable.pack(anchor=W)
        self._register_section(box, enable)
        self._device_combo(
            box, "recording.input_device_index", "input device", input=True
        ).pack(fill=X)
        self._spin(box, "recording.rate", "rate", 8000, 48000, 1).pack(fill=X)
        self._spin(
            box, "recording.silence_threshold", "silence threshold (dBFS)", -120, 0, 1
        ).pack(fill=X)

    def _section_playback(self) -> None:
        box = self._section_box("playback")
        enable = self._check(box, "playback.enable", "enable playback")
        enable.pack(anchor=W)
        self._register_section(box, enable)
        self._device_combo(
            box, "playback.output_device_index", "output device", input=False
        ).pack(fill=X)
        self._spin(box, "playback.volume", "volume", 0, 100, 1).pack(fill=X)

    def _section_transcription(self) -> None:
        box = self._section_box("transcription")
        enable = self._check(box, "transcription.enable", "enable transcription")
        enable.pack(anchor=W)
        self._register_section(box, enable)
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
            child.destroy()
        self.bindings = {
            widget: binding
            for widget, binding in self.bindings.items()
            if widget.winfo_exists()
        }
        self.on_change()
        # get_value() returns None (not a raise) for an unmatched label; the
        # if/elif below then simply renders no backend fields.
        worker_type = combo.get_value()
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
        # newly-built backend fields must follow the section's enable state
        self._apply_all_enables()

    def _section_tts(self) -> None:
        box = self._section_box("tts")
        enable = self._check(box, "tts.enable", "enable tts")
        enable.pack(anchor=W)
        self._register_section(box, enable)
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
            child.destroy()
        self.bindings = {
            widget: binding
            for widget, binding in self.bindings.items()
            if widget.winfo_exists()
        }
        self.on_change()
        # get_value() returns None (not a raise) for an unmatched label; the
        # if/elif below then simply renders no backend fields.
        worker_type = combo.get_value()
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
        # newly-built backend fields must follow the section's enable state
        self._apply_all_enables()

    def _section_vc(self) -> None:
        box = self._section_box("vc")
        enable = self._check(box, "vc.enable", "enable vc")
        enable.pack(anchor=W)
        self._register_section(box, enable)
        self._entry(box, "rvc.model_file", "rvc model_file").pack(fill=X)
        self._entry(box, "rvc.hubert_model_file", "hubert asset dir").pack(fill=X)
        self._entry(box, "rvc.rmvpe_model_file", "rmvpe model_file").pack(fill=X)
        self._spin(box, "rvc.f0_up_key", "f0_up_key", -64, 64, 1).pack(fill=X)
        self._spin(box, "rvc.gpu_id", "gpu_id", 0, 16, 1).pack(fill=X)
