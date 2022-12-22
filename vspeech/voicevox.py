import json
import platform
from ctypes import *
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import TypedDict

import numpy


class AudioQuery(TypedDict):
    accent_phrases: List["AccentPhrase"]
    speedScale: float
    pitchScale: float
    intonationScale: float
    volumeScale: float
    prePhonemeLength: float
    postPhonemeLength: float
    outputSamplingRate: int
    outputStereo: bool
    kana: Optional[str]


class AccentPhrase(TypedDict):
    moras: List["Mora"]
    accent: int
    pause_mora: Optional["Mora"]
    is_interrogative: bool


class Mora(TypedDict):
    text: str
    consonant: Optional[str]
    consonant_length: Optional[float]
    vowel: str
    vowel_length: float
    pitch: float


class VoiceVox:
    def __init__(self, core_dir: Path) -> None:
        # numpy ndarray types
        int64_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=1)
        float32_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=1)
        int64_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=2)
        float32_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2)

        get_os = platform.system()

        lib_file = ""
        if get_os == "Windows":
            lib_file = "core.dll"
        elif get_os == "Darwin":
            lib_file = "libcore.dylib"
        elif get_os == "Linux":
            lib_file = "libcore.so"

        # ライブラリ読み込み
        core_dll_path = core_dir / lib_file
        if not core_dll_path.exists():
            raise Exception(f"coreライブラリファイルが{core_dll_path}に存在しません")
        lib = cdll.LoadLibrary(str(core_dll_path))

        # 関数型定義
        lib.initialize.argtypes = (c_bool, c_int, c_bool)
        lib.initialize.restype = c_bool

        lib.load_model.argtypes = (c_int64,)
        lib.load_model.restype = c_bool

        lib.is_model_loaded.argtypes = (c_int64,)
        lib.is_model_loaded.restype = c_bool

        lib.finalize.argtypes = ()

        lib.metas.restype = c_char_p

        lib.supported_devices.restype = c_char_p

        lib.yukarin_s_forward.argtypes = (
            c_int64,
            int64_dim1_type,
            int64_dim1_type,
            float32_dim1_type,
        )
        lib.yukarin_s_forward.restype = c_bool

        lib.yukarin_sa_forward.argtypes = (
            c_int64,
            int64_dim2_type,
            int64_dim2_type,
            int64_dim2_type,
            int64_dim2_type,
            int64_dim2_type,
            int64_dim2_type,
            int64_dim1_type,
            float32_dim2_type,
        )
        lib.yukarin_sa_forward.restype = c_bool

        lib.decode_forward.argtypes = (
            c_int64,
            c_int64,
            float32_dim2_type,
            float32_dim2_type,
            int64_dim1_type,
            float32_dim1_type,
        )
        lib.decode_forward.restype = c_bool

        lib.last_error_message.restype = c_char_p

        lib.voicevox_load_openjtalk_dict.argtypes = (c_char_p,)
        lib.voicevox_load_openjtalk_dict.restype = c_int

        lib.voicevox_tts.argtypes = (
            c_char_p,
            c_int64,
            POINTER(c_int),
            POINTER(POINTER(c_uint8)),
        )
        lib.voicevox_tts.restype = c_int

        lib.voicevox_tts_from_kana.argtypes = (
            c_char_p,
            c_int64,
            POINTER(c_int),
            POINTER(POINTER(c_uint8)),
        )
        lib.voicevox_tts_from_kana.restype = c_int

        lib.voicevox_wav_free.argtypes = (POINTER(c_uint8),)

        lib.voicevox_error_result_to_message.argtypes = (c_int,)
        lib.voicevox_load_openjtalk_dict.argtypes = (c_char_p,)

        self.lib = lib

    def __enter__(self) -> "VoiceVox":
        self.initialize(use_gpu=False, cpu_num_threads=0, load_all_models=False)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.finalize()

    # ラッパー関数
    def initialize(self, use_gpu: bool, cpu_num_threads=0, load_all_models=True):
        success = self.lib.initialize(use_gpu, cpu_num_threads, load_all_models)
        if not success:
            raise Exception(self.lib.last_error_message().decode())

    def load_model(self, speaker_id: int):
        success = self.lib.load_model(speaker_id)
        if not success:
            raise Exception(self.lib.last_error_message().decode())

    def is_model_loaded(self, speaker_id: int) -> bool:
        return self.lib.is_model_loaded(speaker_id)

    def metas(
        self,
    ) -> str:
        return self.lib.metas().decode()

    def supported_devices(
        self,
    ) -> str:
        return self.lib.supported_devices().decode()

    def yukarin_s_forward(
        self, length: int, phoneme_list: numpy.ndarray, speaker_id: numpy.ndarray
    ) -> numpy.ndarray:
        output = numpy.zeros((length,), dtype=numpy.float32)
        success = self.lib.yukarin_s_forward(length, phoneme_list, speaker_id, output)
        if not success:
            raise Exception(self.lib.last_error_message().decode())
        return output

    def yukarin_sa_forward(
        self,
        length: int,
        vowel_phoneme_list,
        consonant_phoneme_list,
        start_accent_list,
        end_accent_list,
        start_accent_phrase_list,
        end_accent_phrase_list,
        speaker_id,
    ):
        output = numpy.empty(
            (
                len(speaker_id),
                length,
            ),
            dtype=numpy.float32,
        )
        success = self.lib.yukarin_sa_forward(
            length,
            vowel_phoneme_list,
            consonant_phoneme_list,
            start_accent_list,
            end_accent_list,
            start_accent_phrase_list,
            end_accent_phrase_list,
            speaker_id,
            output,
        )
        if not success:
            raise Exception(self.lib.last_error_message().decode())
        return output

    def decode_forward(self, length: int, phoneme_size: int, f0, phoneme, speaker_id):
        output = numpy.empty((length * 256,), dtype=numpy.float32)
        success = self.lib.decode_forward(
            length, phoneme_size, f0, phoneme, speaker_id, output
        )
        if not success:
            raise Exception(self.lib.last_error_message().decode())
        return output

    def voicevox_load_openjtalk_dict(self, dict_path: str):
        errno = self.lib.voicevox_load_openjtalk_dict(dict_path.encode())
        if errno != 0:
            raise Exception(self.lib.voicevox_error_result_to_message(errno).decode())

    def voicevox_tts(self, text: str, speaker_id: int) -> bytes:
        output_binary_size = c_int()
        output_wav = POINTER(c_uint8)()
        errno = self.lib.voicevox_tts(
            text.encode(), speaker_id, byref(output_binary_size), byref(output_wav)
        )
        if errno != 0:
            raise Exception(self.lib.voicevox_error_result_to_message(errno).decode())
        output = create_string_buffer(output_binary_size.value * sizeof(c_uint8))
        memmove(output, output_wav, output_binary_size.value * sizeof(c_uint8))
        self.lib.voicevox_wav_free(output_wav)
        return output

    def voicevox_tts_from_kana(self, text: str, speaker_id: int) -> bytes:
        output_binary_size = c_int()
        output_wav = POINTER(c_uint8)()
        errno = self.lib.voicevox_tts_from_kana(
            text.encode(), speaker_id, byref(output_binary_size), byref(output_wav)
        )
        if errno != 0:
            raise Exception(self.lib.voicevox_error_result_to_message(errno).decode())
        output = create_string_buffer(output_binary_size.value * sizeof(c_uint8))
        memmove(output, output_wav, output_binary_size.value * sizeof(c_uint8))
        self.lib.voicevox_wav_free(output_wav)
        return output

    def finalize(self):
        self.lib.finalize()
