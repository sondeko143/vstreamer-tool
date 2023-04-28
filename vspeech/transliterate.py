import re
from typing import Optional
from typing import cast

from mozcpy import Converter

from vspeech.ami import FILLER_PATTERN
from vspeech.ami import AmiResponse
from vspeech.ami import AmiToken


def delimiter_token(token: AmiToken) -> Optional[str]:
    match = re.fullmatch(FILLER_PATTERN, token.written)
    if match:
        return match.group(1)
    if token.written in ["、", "。"]:
        return token.written
    return None


def get_transliterated_text(res_body: AmiResponse):
    converter = Converter()
    full_text = ""
    one_sentence = ""
    for result in res_body.results:
        for token in result.tokens:
            delimiter = delimiter_token(token)
            if delimiter:
                if one_sentence:
                    full_text += cast(str, converter.convert(one_sentence))
                full_text += delimiter
                one_sentence = ""
            else:
                one_sentence += token.spoken
    if one_sentence:
        full_text += cast(str, converter.convert(one_sentence))
    return full_text
