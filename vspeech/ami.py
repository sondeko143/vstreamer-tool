import re
from typing import Any
from typing import List
from typing import Optional

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
