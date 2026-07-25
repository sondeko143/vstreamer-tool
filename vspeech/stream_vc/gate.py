"""streaming VC の窓単位 VAD ノイズゲート(ADR-0059 / ADR-0053 / ADR-0019)。

streaming 経路は無音でも止まらず回り続けるので、ゲートが無いと**部屋のノイズ
フロアがそのまま RVC を通り、しかも増幅されて**鳴り続ける。さらに語頭では
HuBERT 特徴抽出器の第1層 GroupNorm が解析窓全体の時間軸統計で正規化する結果、
発声直前の微小入力(実測 RMS 0.002)が音声レベルまで持ち上げられ、気音として
合成される(実測: 同一音声・同一モデルで batch 経路の **+41dB**)。モデル側の
正規化は動かせないので、**バッチ経路と同じ 32ms 窓の粒度でゲートする**ことで
可聴成分を落とす(実録音の e2e 実測で語頭のブレス -20.5dB / -12.4dB、本物の語頭と
定常は保持)。

ここは判定と適用だけを持つモデル非依存の純ロジックで、Silero VAD 本体は
`vspeech/lib/vad.py`(発話系 `[vc]` と共有)をそのまま読み取り専用で再利用する。
そのため CPU・モデル無しで単体テストできる。

設計上の要点:

- **入力ブロックで判定し、出力ブロックへ適用する**。ゲートが閉じていても推論は
  スキップしない。`StreamingVc` は rolling 左文脈とクロスフェード tail を持つ
  ステートフル変換なので、ブロックを飛ばすと文脈に穴が開き、発話が再開した
  ときの seam が壊れる。減衰するのは emit する音だけ。
- **判定も適用も 32ms 窓の粒度**(発話系 `lib/vad.py` の `speech_gate_mask` /
  `apply_vad_gate` と同じ考え方)。ブロック粒度(160ms)で窓確率の max を採ると、
  語頭の 1 窓のせいでブロック全体が開き、その手前にある発声前のブレスを full
  gain で通してしまう。
- **hangover は前方へは dilate しない**。バッチ側 `speech_gate_mask` は前後対称に
  広げるが、streaming で前方へ広げると語頭直前のブレスをそのまま開けてしまう
  (実録音へマスク単体をかけた実測: 前方 0ms なら -26dB のところ 32ms 足すと -9dB
  まで後退)。語尾・語間の保護に要るのは後方だけなので、`hangover_ms` を
  **後方 dilation** として使う。
- **emit 遅延を補正して重ねる**。emit の内容は入力ブロックより手前から始まる
  (crossfade + SOLA + HuBERT 受容野の切り詰めで実測 ~52ms)。補正しないとマスクが
  ずれた音声に当たり、同じ実測で抑圧が -26dB から -8dB まで落ちる。遅延量は
  `StreamingVc.emit_delay_samples` が tick ごとに公開する。
- **ゲインは窓中心のあいだを線形補間する**。境界でゲインを階段状に変えること自体が
  クリックを生むので、32ms かけて渡す(前ブロックのマスク末尾から連続させる)。

numpy は `vspeech/lib/stream_vc.py` と同様にメソッド内 import に留める
(この module を import 軽量に保つ)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import VAD_WINDOW_SAMPLES

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Silero の 1 窓の長さ(ms)。hangover をこの粒度の窓数へ換算する。
_WINDOW_MS = VAD_WINDOW_SAMPLES * 1000.0 / VAD_SAMPLE_RATE


class StreamingVadGate:
    """32ms 窓単位のゲートマスク + emit 遅延補正付きの適用。

    `window_gains()` がこのブロックの窓ごとのゲインを返し、`apply()` がそれを
    emit のサンプル格子へ(遅延補正して)写して掛ける。状態は「最後に speech を
    見てから何窓経ったか」と「前ブロックのマスク」の二つだけ。
    """

    def __init__(self, threshold: float, hangover_ms: float, min_gain: float) -> None:
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self.min_gain = min_gain
        # fail-open 警告の重複抑止フラグ(runner が使う)。streaming は 6.25Hz で
        # 回るので、VAD が壊れたときに毎ブロック警告するとログが埋まる。
        self.warned = False
        self._hangover_windows = max(0, round(hangover_ms / _WINDOW_MS))
        # 最後の speech からの窓数。予算超えで頭打ちにして単調増加を止める。
        # 初期値は「閉じた状態」: 窓単位なら speech の窓がそのまま開くので、
        # 開いた状態から始めて無音を漏らす必要が無い。
        self._since_speech = self._hangover_windows + 1
        self._prev_gains: NDArray[np.float64] | None = None

    def reset(self) -> None:
        """閉じた状態(hangover 空・前ブロックのマスク無し)へ戻す。

        pause/resume や capture 再 open で実時間が飛んだあと、古い hangover 残量や
        マスクが漏れて直後のブロックを妙に開放/減衰させないため、runner が遷移で
        呼ぶ。`warned`(fail-open 警告の重複抑止)は障害状態なので触らない。
        """
        self._since_speech = self._hangover_windows + 1
        self._prev_gains = None

    def window_gains(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """窓確率列からこのブロックの窓ごとのゲインを返す(後方 dilation のみ)。

        speech 窓は 1.0。無音窓は最後の speech から `hangover_ms` 以内なら 1.0、
        超えたら `min_gain`。予算はブロック境界をまたいで持ち越す(判定はブロック
        ごとに来るが、発話は境界を意識しない)。
        """
        import numpy as np

        gains = np.empty(probs.shape[0], dtype=np.float64)
        for i in range(probs.shape[0]):
            if probs[i] >= self.threshold:
                self._since_speech = 0
            else:
                self._since_speech = min(
                    self._since_speech + 1, self._hangover_windows + 1
                )
            gains[i] = (
                1.0 if self._since_speech <= self._hangover_windows else self.min_gain
            )
        return gains

    def apply(
        self,
        out_i16: NDArray[np.int16],
        gains: NDArray[np.float64],
        delay_samples: int,
        sample_rate: int,
    ) -> NDArray[np.int16]:
        """窓ゲインを emit のサンプル格子へ写して掛ける(遅延補正つき)。

        emit のサンプル j が持つ音は、入力ブロック先頭から見て
        `j - delay_samples`(出力レート)の位置にある。したがってマスクもその分だけ
        ずらして重ねる。emit の先頭 `delay_samples` は**前ブロック**の入力に対応
        するので、前ブロックのマスク(`_prev_gains`)を左へ連結してから補間する
        — これがブロック境界のゲイン連続性(段差=クリック無し)も同時に担保する。

        全窓 1.0(かつ直前も 1.0)は恒等の高速路で、入力オブジェクトをそのまま返す:
        常時 speech / 既定 off のとき出力は無ゲート時とビット単位で一致する。
        """
        import numpy as np

        n = int(out_i16.shape[0])
        if n == 0 or gains.shape[0] == 0:
            return out_i16
        prev = self._prev_gains
        self._prev_gains = gains
        if prev is None:
            # 直前の情報が無い(起動直後 / reset 直後)。頭を今ブロック先頭窓の
            # ゲインで保持する = 余計な遷移を作らない。
            prev = gains[:1]
        if float(gains.min()) == 1.0 and float(prev.min()) == 1.0:
            return out_i16
        # 窓 1 つぶんの出力サンプル数。窓中心をこの格子に並べて線形補間する。
        step = VAD_WINDOW_SAMPLES * sample_rate / VAD_SAMPLE_RATE
        all_gains = np.concatenate([prev, gains])
        n_prev = int(prev.shape[0])
        centers = (
            np.arange(all_gains.shape[0], dtype=np.float64) + 0.5 - n_prev
        ) * step
        gain = np.interp(
            np.arange(n, dtype=np.float64) - delay_samples, centers, all_gains
        )
        out_f = out_i16.astype(np.float32) * gain
        return np.clip(np.rint(out_f), -32768.0, 32767.0).astype(np.int16)
