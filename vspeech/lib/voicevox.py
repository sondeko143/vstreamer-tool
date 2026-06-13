from pathlib import Path

from voicevox_core import AccelerationMode
from voicevox_core.blocking import Onnxruntime
from voicevox_core.blocking import OpenJtalk
from voicevox_core.blocking import Synthesizer
from voicevox_core.blocking import VoiceModelFile

from vspeech.config import VoicevoxParam


class Voicevox:
    def __init__(
        self,
        open_jtalk_dict_dir: Path,
        model_dir: Path,
        onnxruntime_path: Path | None = None,
    ) -> None:
        if onnxruntime_path is not None:
            onnxruntime = Onnxruntime.load_once(
                filename=str(onnxruntime_path.expanduser())
            )
        else:
            onnxruntime = Onnxruntime.load_once()
        self.synthesizer = Synthesizer(
            onnxruntime,
            OpenJtalk(str(open_jtalk_dict_dir.expanduser())),
            acceleration_mode=AccelerationMode.AUTO,
        )
        self.model_dir = model_dir.expanduser()
        self._style_index: dict[int, Path] = self._build_style_index(self.model_dir)
        self._loaded: set[int] = set()

    @staticmethod
    def _build_style_index(model_dir: Path) -> dict[int, Path]:
        index: dict[int, Path] = {}
        for vvm_path in sorted(model_dir.glob("*.vvm")):
            with VoiceModelFile.open(str(vvm_path)) as model:
                for character in model.metas:
                    for style in character.styles:
                        index[int(style.id)] = vvm_path
        return index

    def load_model(self, style_id: int) -> None:
        if style_id in self._loaded:
            return
        vvm_path = self._style_index.get(style_id)
        if vvm_path is None:
            raise ValueError(f"no voice model found for style_id={style_id}")
        try:
            with VoiceModelFile.open(str(vvm_path)) as model:
                self.synthesizer.load_voice_model(model)
        except Exception as e:
            raise ValueError(e)
        self._loaded.add(style_id)

    def is_model_loaded(self, style_id: int) -> bool:
        return style_id in self._loaded

    def voicevox_tts(self, text: str, speaker_id: int, params: VoicevoxParam) -> bytes:
        try:
            audio_query = self.synthesizer.create_audio_query(text, speaker_id)
            for key, value in params:
                setattr(audio_query, key, value)
            return self.synthesizer.synthesis(audio_query, speaker_id)
        except Exception as e:
            raise ValueError(e)
