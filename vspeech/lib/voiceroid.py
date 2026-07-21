from codecs import encode
from dataclasses import dataclass
from dataclasses import field

from humps import camelize
from pyvcroid2 import VcRoid2

from vspeech.config import VR2Param
from vspeech.logger import logger


@dataclass
class VR2:
    vc_roid2: VcRoid2 = field(default_factory=VcRoid2)
    loaded_voices: set[str] = field(default_factory=set)

    def load_voice(self, voice_name: str, vr2_params: VR2Param):
        if voice_name not in self.loaded_voices:
            self.vc_roid2.loadVoice(voice_name)
            self.loaded_voices.add(voice_name)
        if self.vc_roid2.param:
            # pydantic v2: iterate model_fields, NOT get_type_hints(vr2_params).
            # get_type_hints() on a *model instance* returns {} (an instance's
            # __annotations__ is empty in v2), which would silently apply zero
            # parameters.
            for name in type(vr2_params).model_fields:
                value = getattr(vr2_params, name)
                logger.info("vr2 %s: %s", name, value)
                camel = camelize(name)
                setattr(self.vc_roid2.param, camel, value)
                logger.debug(
                    "vr2 param %s applied: %s",
                    name,
                    getattr(self.vc_roid2.param, camel),
                )

    def list_languages(self):
        return self.vc_roid2.listLanguages()

    def load_language(self, language_name: str):
        return self.vc_roid2.loadLanguage(language_name)

    def list_voices(self):
        return self.vc_roid2.listVoices()

    def text_to_speech(
        self, text: str, timeout: float | None = None, raw: bool = False
    ):
        remove_invalid_shiftjis = encode(text, encoding="shift-jis", errors="ignore")
        remove_invalid = remove_invalid_shiftjis.decode(
            encoding="shift-jis", errors="replace"
        )
        return self.vc_roid2.textToSpeech(remove_invalid, timeout=timeout, raw=raw)
