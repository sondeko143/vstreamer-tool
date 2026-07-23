"""streaming VC のブロック粒度 VAD ノイズゲート(ADR-0053 / ADR-0019)。

streaming 経路は無音でも止まらず回り続けるので、ゲートが無いと**部屋の
ノイズフロアがそのまま RVC を通り、しかも増幅されて**鳴り続ける(実測: 入力
RMS 0.000298 に対し出力 0.0204 = +10〜+37dB、context 100ms 時)。

ここは判定と適用だけを持つモデル非依存の純ロジックで、Silero VAD 本体は
`vspeech/lib/vad.py`(発話系 `[vc]` と共有)をそのまま読み取り専用で再利用する。
そのため CPU・モデル無しで単体テストできる。

設計上の要点:

- **入力ブロックで判定し、出力ブロックへ適用する**。ゲートが閉じていても推論は
  スキップしない。`StreamingVc` は rolling 左文脈とクロスフェード tail を持つ
  ステートフル変換なので、ブロックを飛ばすと文脈に穴が開き、発話が再開した
  ときの seam が壊れる。減衰するのは emit する音だけ。
- **hangover 付きのステートフル判定**。ブロック単位(160ms なら 6.25Hz)で
  素の VAD 判定をそのまま使うと語間の短い無音でゲートがバタつく。
- **ゲインはブロック内で線形に ramp する**。ブロック境界でゲインを階段状に
  変えること自体がクリックを生む — この branch がずっと戦ってきた種類の
  アーティファクトそのものなので、前ブロック終端のゲインから今ブロックの
  目標ゲインへ滑らかに繋ぐ。

numpy は `vspeech/lib/stream_vc.py` と同様にメソッド内 import に留める
(この module を import 軽量に保つ)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class StreamingVadGate:
    """hangover 付きブロック粒度ゲート + ブロック内ゲイン ramp。

    `update()` が「このブロックの目標ゲイン」を返し、`ramp()` がそれを前ブロック
    終端のゲインから繋いで出力へ適用する。状態は hangover の残り時間と直前の
    ゲインの二つだけ。
    """

    def __init__(
        self, threshold: float, hangover_ms: float, min_gain: float, block_ms: float
    ) -> None:
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self.min_gain = min_gain
        self.block_ms = block_ms
        self._hangover_remaining_ms = 0.0
        # fail-open 警告の重複抑止フラグ(runner が使う)。streaming は 6.25Hz で
        # 回るので、VAD が壊れたときに毎ブロック警告するとログが埋まる。
        self.warned = False
        # 直前ブロック終端のゲイン。初期値 1.0 = 開いた状態から始める(最初の
        # ブロックが無音なら 1.0 -> min_gain へ ramp して閉じる。逆にすると
        # 起動直後の発話頭が ramp で欠ける)。
        self._gain = 1.0

    def speech_from_probs(self, probs: NDArray[np.float64]) -> bool:
        """ブロック内の窓確率の **max** を threshold と比較する。

        160ms ブロックは Silero の 32ms 窓で ~5 窓ぶん。mean/ratio ではなく max
        を採るのは、ブロック末尾で始まった発話頭(onset)を落とさないため
        — 1 窓でも speech なら開ける。閉じる側は hangover が受け持つ。
        空(窓なし)は無音扱い。
        """
        return bool(probs.shape[0] and probs.max() >= self.threshold)

    def update(self, is_speech: bool) -> float:
        """このブロックの目標ゲインを返す(hangover ステートマシン)。

        speech なら hangover 予算を `hangover_ms` に戻して 1.0。無音なら予算を
        `block_ms` だけ減らし、まだ残っていれば 1.0、尽きたら `min_gain`。
        """
        if is_speech:
            self._hangover_remaining_ms = self.hangover_ms
            return 1.0
        self._hangover_remaining_ms -= self.block_ms
        if self._hangover_remaining_ms > 0.0:
            return 1.0
        self._hangover_remaining_ms = 0.0
        return self.min_gain

    def ramp(self, out_i16: NDArray[np.int16], target_gain: float) -> NDArray[np.int16]:
        """前ブロック終端のゲインから `target_gain` までブロック内で線形に繋ぐ。

        末端が正確に `target_gain` になる(endpoint 込みの linspace)ので、次
        ブロックの ramp 始点と連続する = ブロック境界に段差が生じない。
        `1.0 -> 1.0` は恒等の高速路: ゲート既定 off / 常時 speech のとき出力は
        ビット単位で無ゲート時と一致する。
        """
        import numpy as np

        start = self._gain
        self._gain = target_gain
        if start == 1.0 and target_gain == 1.0:
            return out_i16
        n = int(out_i16.shape[0])
        if n == 0:
            return out_i16
        gain = np.linspace(start, target_gain, n, dtype=np.float32)
        out_f = out_i16.astype(np.float32) * gain
        return np.clip(np.rint(out_f), -32768.0, 32767.0).astype(np.int16)
