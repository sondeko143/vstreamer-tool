"""ストリーミング VC の入力エンベロープ追従 (ADR-0057)。

入力ブロックの相対ラウドネス包絡を、入力平均 RMS の rolling EMA を参照に正規化し、
duck ゲイン (clip(shape^strength, min_gain, max_gain)) として出力ブロックへ掛ける。
バッチ apply_input_envelope (worker/vc.py) と同じダック思想だが、参照を「発話全体の
平均」→「rolling EMA」に置換したストリーミング版 (単一ブロックしか手に入らない)。

判定と適用だけの pure ロジックで、numpy はメソッド内 import (torch/sounddevice を
引かず CPU・モデル無しで単体テストできる。gate.py と同型)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# 入力ブロックのサンプルレート (capture.py CAPTURE_RATE と同じ 16k)。capture.py を
# import すると sounddevice を引くのでここでは定数で持つ (この module を CPU で単体
# テストできるようにするため)。
_INPUT_RATE = 16000


class StreamingEnvelope:
    """rolling-EMA 参照の入力エンベロープ追従 (duck, ADR-0057)。

    状態は参照レベル `_ema_level` (スカラ) のみ。`apply()` が出力ブロックへ現在の
    入力ブロックの相対ラウドネス包絡を掛け、参照 EMA を次ブロック用に更新する。
    """

    def __init__(
        self,
        strength: float,
        min_gain: float,
        max_gain: float,
        window_ms: float,
        ema_ms: float,
        block_ms: float,
    ) -> None:
        self.strength = strength
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.window_ms = window_ms
        # 時定数 ema_ms の per-block EMA 係数 alpha = 1 - exp(-block_ms/ema_ms)。
        self._alpha = 1.0 - math.exp(-block_ms / ema_ms) if ema_ms > 0 else 1.0
        self._ema_level: float | None = None  # 初回 apply で block mean から init

    def reset(self) -> None:
        """参照レベルを未初期化へ戻す (pause/resume・capture 再 open で runner が呼ぶ)。

        実時間が飛んだあと古い参照レベルが次ブロックを妙に duck しないよう、
        次の apply で改めて cold start (block mean で init) させる。
        """
        self._ema_level = None

    def apply(
        self, out_i16: NDArray[np.int16], in_block: NDArray[np.float32]
    ) -> NDArray[np.int16]:
        """出力ブロック out_i16 に、入力ブロック in_block (16k float32) の相対
        ラウドネス包絡を rolling EMA 参照で duck 適用する。

        参照は **過去の** EMA (履歴)。cold start / reset 直後は現ブロックの平均で
        初期化する (初回ブロックが不自然に duck されないため)。参照を更新してから
        返すので、次ブロックはこのブロックを織り込んだ EMA を使う。

        **既知の特性 (ADR-0057, 実機耳確認で調整):** 長い無音では参照 EMA が入力の
        ノイズ床へ寄る (envelope_ema_ms で減衰)。その直後の発話頭 (phrase onset) は
        低い参照に対して全フレームが loud 判定になり duck されにくい = このブロック
        単独では整形が弱い。連続発話中の語間 dip / decay tail は参照が発話レベルに
        あるので正しく整形される。phrase onset は VAD ゲートが受け持つ。ema_ms を
        長くすると参照が無音を跨いで発話レベルを保ち、onset 整形が効きやすくなる。
        """
        import numpy as np

        out_len = int(out_i16.shape[0])
        if out_len == 0 or in_block.shape[0] == 0 or self.strength <= 0.0:
            return out_i16
        # 入力の per-frame RMS (絶対スケールは参照正規化で相殺されるので無関係)。
        frame_len = max(1, round(self.window_ms * _INPUT_RATE / 1000.0))
        n_frames = max(1, in_block.shape[0] // frame_len)
        bounds = np.linspace(0, in_block.shape[0], n_frames + 1).astype(np.int64)
        frame_rms = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            seg = in_block[bounds[i] : bounds[i + 1]].astype(np.float64)
            if seg.shape[0]:
                frame_rms[i] = np.sqrt(np.mean(seg**2))
        block_mean = float(frame_rms.mean())
        if self._ema_level is None:
            self._ema_level = block_mean
        ref = self._ema_level
        self._ema_level = self._alpha * block_mean + (1.0 - self._alpha) * ref
        if ref < 1e-8:  # 実質デジタル無音 (init 直後の完全無音等) → 素通し
            return out_i16
        # 相対形状 (mean~1 ではなく参照相対) を出力サンプル格子へ線形補間。
        src_x = (np.arange(n_frames) + 0.5) / n_frames
        dst_x = (np.arange(out_len) + 0.5) / out_len
        shape = np.interp(dst_x, src_x, frame_rms / ref)
        gain = np.clip(np.power(shape, self.strength), self.min_gain, self.max_gain)
        out_f = out_i16.astype(np.float32)
        return np.clip(out_f * gain, -32768.0, 32767.0).astype(np.int16)
