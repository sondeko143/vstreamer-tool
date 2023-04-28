from typing import List
from typing import Union


class Converter(object):
    @staticmethod
    def _convert(sentence: str, n_best: int, mecab_args: str) -> Union[List[str], str]:
        ...

    def convert(self, sentence: str, n_best: int = 1) -> Union[List[str], str]:
        ...

    def convert_wakati(self, sentence: str, n_best: int = 1) -> Union[List[str], str]:
        ...

    def wakati(self, sentence: str, n_best: int = 1) -> Union[List[str], str]:
        ...
