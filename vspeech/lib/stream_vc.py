"""ストリーミング VC の再利用コア(ADR-0053)。

固定長ブロックを rolling 左文脈と連結してステートフル変換する。既存の
`change_voice` の内部部品(HuBERT 特徴量 / f0 / infer / int16化)をそのまま
再利用し、発話系の `change_voice` 経路は無改変で温存する。M1 はこのコアの
per-block 計測(RTF)に集中し、クロスフェード連続性の音質は M2 で足す。

純粋ヘルパ(next_context / slice_block_output)は numpy でも torch tensor でも
動くよう `len(seq)` ベースにしてあり、torch 無し・rvc extra 無しの CPU でも
import できる(重い import は StreamingVc のメソッド内でのみ行う)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Not used by the pure helpers below; reserved for the StreamingVc class
    # that a later task adds to this module (M1 Task 2/3).
    import torch  # noqa: F401
    from numpy.typing import NDArray  # noqa: F401
    from onnxruntime import InferenceSession  # noqa: F401

    from vspeech.config import RvcConfig  # noqa: F401
    from vspeech.lib.rvc import HubertSession  # noqa: F401


def next_context(seq, context_len: int):
    """`seq` の末尾 `context_len` 要素(次 tick の左文脈)。

    `context_len == 0` のとき `seq[-0:]` は全体を返してしまうので、明示的に
    空スライスにする。`len(seq)` ベースなので numpy/torch 双方で同じ挙動。
    `context_len >= len(seq)` のときは全体を返す(clamp — 使える分だけ左文脈を
    渡す防御的ガード。StreamingVc の呼び出し側は文脈を全長まで事前充填する)。
    """
    if context_len <= 0:
        return seq[:0]
    return seq[max(0, len(seq) - context_len) :]


def slice_block_output(out, block_len: int, seq_len: int):
    """`out` のうち、直近ブロック相当(末尾 block_len/seq_len)の区間。

    infer は `[context|block]` 全体の波形を返すので、ブロック相当の末尾だけ
    採用する。正確なシーム整列(等電力クロスフェード)は M2 の担当で、ここは
    比率で切り出す近似(RTF 計測には出力長は影響しない)。
    `block_out >= len(out)` のときは `out` 全体を返す(clamp)。
    """
    if block_len <= 0:
        return out
    block_out = round(len(out) * block_len / seq_len)
    if block_out <= 0:
        return out
    return out[max(0, len(out) - block_out) :]
