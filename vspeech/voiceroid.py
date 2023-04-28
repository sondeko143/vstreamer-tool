from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Set
from typing import get_type_hints

from humps import camelize
from pyvcroid2 import VcRoid2

from vspeech.config import VR2Param
from vspeech.logger import logger


@dataclass
class VR2:
    vc_roid2: VcRoid2 = field(default=VcRoid2())
    loaded_voices: Set[str] = field(default_factory=set)

    def load_voice(self, voice_name: str, vr2_params: VR2Param):
        if voice_name not in self.loaded_voices:
            self.vc_roid2.loadVoice(voice_name)
            self.loaded_voices.add(voice_name)
        if self.vc_roid2.param:
            for name in get_type_hints(vr2_params):
                value = getattr(vr2_params, name)
                logger.info("vr2 %s: %s", name, value)
                setattr(self.vc_roid2.param, camelize(name), value)

    def list_languages(self):
        return self.vc_roid2.listLanguages()

    def load_language(self, language_name: str):
        return self.vc_roid2.loadLanguage(language_name)

    def list_voices(self):
        return self.vc_roid2.listVoices()

    def text_to_speech(
        self, text: str, timeout: Optional[float] = None, raw: bool = False
    ):
        return self.vc_roid2.textToSpeech(text, timeout=timeout, raw=raw)
