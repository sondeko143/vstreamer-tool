import re
from typing import Any
from typing import List
from typing import Optional
from typing import cast

from pydantic import BaseModel

FILLER_PATTERN = re.compile(r"^%(.+)%$")


class AmiToken(BaseModel):
    written: str
    confidence: float
    starttime: int
    endtime: int
    spoken: str


class AmiResult(BaseModel):
    tokens: List[AmiToken]
    confidence: Optional[float]
    starttime: Optional[int]
    endtime: Optional[int]
    tags: List[Any]
    rulename: str
    text: str


class AmiResponse(BaseModel):
    results: List[AmiResult]
    utteranceid: str
    text: str
    code: str
    message: str


def parse_response(res_json: Any, use_mozc: bool):
    res_body = AmiResponse.parse_obj(res_json)
    if use_mozc:
        text = get_transliterated_text(res_body=res_body)
    else:
        text = text_removed_filler_symbol(res_body)
    spoken = "".join(
        [
            "".join([token.spoken for token in result.tokens])
            for result in res_body.results
        ]
    )
    return text, spoken


def text_removed_filler_symbol(res_body: AmiResponse):
    return "".join(
        [
            "".join(
                [
                    token.written.replace("%", "")
                    if re.fullmatch(FILLER_PATTERN, token.written)
                    else token.written
                    for token in result.tokens
                ]
            )
            for result in res_body.results
        ]
    )


def delimiter_token(token: AmiToken) -> Optional[str]:
    match = re.fullmatch(FILLER_PATTERN, token.written)
    if match:
        return match.group(1)
    if token.written in ["、", "。"]:
        return token.written
    return None


def get_transliterated_text(res_body: AmiResponse):
    from mozcpy import Converter

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
