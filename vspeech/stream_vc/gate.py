"""streaming VC の窓単位 VAD ノイズゲート(ADR-0059 / ADR-0053 / ADR-0019)。

streaming 経路は無音でも止まらず回り続けるので、ゲートが無いと**部屋のノイズ
フロアがそのまま RVC を通り、しかも増幅されて**鳴り続ける。さらに語頭では、
発声直前の微小入力(実測 RMS 0.002)が音として合成されて出る(実測: 同一音声・
同一モデルで batch 経路の **+43dB**)。これは解析窓の中身に依存する現象で、
content encoder が同じ音を左文脈次第で別物として符号化するために起きる
(f0 経路ではないことは実測で確認済み。窓のどの性質が効いているかまでは
切り分けていない — ADR-0059 参照)。モデル側は動かせないので、**バッチ経路と
同じ 32ms 窓の粒度でゲートする**ことで可聴成分を落とす(実録音の e2e 実測で
語頭のブレス -25.4dB / -16.9dB、本物の語頭と定常は保持)。

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
  (crossfade + HuBERT 受容野の切り詰めで既定 50ms)。補正しないとマスクがずれた
  音声に当たり、実測で抑圧が -26dB から -8dB まで落ちる。遅延量は
  `StreamingVc.emit_delay_samples` が公開する(公称位置由来なので tick 間で一定)。
- **ゲインは窓中心のあいだを線形補間する**。境界でゲインを階段状に変えること自体が
  クリックを生むので、32ms かけて渡す(前ブロックのマスク末尾から連続させる)。

numpy は `vspeech/lib/stream_vc.py` と同様にメソッド内 import に留める
(この module を import 軽量に保つ)。
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from vspeech.lib.vad import VAD_SAMPLE_RATE
from vspeech.lib.vad import VAD_WINDOW_SAMPLES
from vspeech.lib.vad import VadCarry

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
        self.min_gain = min_gain
        # fail-open 警告の重複抑止フラグ(runner が使う)。streaming は 6.25Hz で
        # 回るので、VAD が壊れたときに毎ブロック警告するとログが埋まる。
        self.warned = False
        # Silero の再帰状態。ブロックごとに作り直すと RNN が毎回コールドスタートし、
        # 明確な有声窓の確率まで壊れる(lib/vad.py の VadCarry 参照)。runner が
        # speech_probs へ渡す。
        self.vad_carry = VadCarry()
        self._hangover_windows = max(0, round(hangover_ms / _WINDOW_MS))
        # 最後の speech からの窓数。予算超えで頭打ちにして単調増加を止める。
        # 初期値は「閉じた状態」: 窓単位なら speech の窓がそのまま開くので、
        # 開いた状態から始めて無音を漏らす必要が無い。
        self._since_speech = self._hangover_windows + 1
        self._prev_gains: NDArray[np.float64] | None = None

    def reset(self) -> None:
        """閉じた状態(hangover 空・前ブロックのマスク無し・VAD 状態も新品)へ戻す。

        pause/resume や capture 再 open で実時間が飛んだあと、古い hangover 残量や
        マスク、飛ぶ前の音で育った VAD の再帰状態が漏れて直後のブロックを妙に
        開放/減衰させないため、runner が遷移で呼ぶ。`warned`(fail-open 警告の
        重複抑止)は障害状態なので触らない。

        `vad_carry` を残す案は実測して**却下**した(ADR-0059 の Alternatives 参照)。
        発話中に pause して無音へ resume すると、古い「発話中」の状態が最初の窓を
        誤って speech と判定しうる。1 窓でも誤ると `_since_speech` が 0 に戻って
        hangover 予算が満額で再武装されるので、漏れは 1 窓では止まらない
        (実測: 104 通り中 8 回漏れ、最大 320ms)。それはこの ADR が消そうとしている
        「増幅された微小入力が鳴る」そのものなので、精度と引き換えにはできない。
        """
        self._since_speech = self._hangover_windows + 1
        self._prev_gains = None
        self.vad_carry = VadCarry()

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
        ただし連続性が成り立つのは `delay_samples` が tick 間で一定のときだけなので、
        `StreamingVc` は SOLA の lag を含まない**公称**遅延を公開する(ADR-0059)。

        全窓 1.0(かつ直前も 1.0)は恒等の高速路で、入力オブジェクトをそのまま返す:
        常時 speech / 既定 off のとき出力は無ゲート時とビット単位で一致する
        (起動直後・reset 直後の 1 ブロックだけは閉じた状態から開くので例外)。
        """
        import numpy as np

        n = int(out_i16.shape[0])
        if n == 0 or gains.shape[0] == 0:
            return out_i16
        # 窓 1 つぶんの出力サンプル数。窓中心をこの格子に並べて線形補間する。
        step = VAD_WINDOW_SAMPLES * sample_rate / VAD_SAMPLE_RATE
        prev = self._prev_gains
        self._prev_gains = gains
        if prev is None:
            # 直前の情報が無い(起動直後 / reset 直後)。emit の頭は「実時間が飛ぶ前」
            # または zeros 文脈から描かれた音なので、閉じた状態(min_gain)から始める
            # ── `_since_speech` の初期値と揃える。**1 窓ではなく hop ぶんの窓数**を
            # 置くこと: 1 要素だとその中心が hop まるごと手前(既定で -144ms)に来て、
            # 立ち上がりが 32ms でなく 160ms かけて渡り、頭が閉じきらない(実測 -4.6dB)。
            # 窓数は実マスクと同じ数え方(ceil)にする。round だとブロック長が窓長の
            # 倍数でない設定(block_ms=80)で 1 窓少なくなり、最後の seed 中心が
            # 手前へずれて頭が閉じきらない。
            prev = np.full(max(1, ceil(n / step)), self.min_gain, dtype=np.float64)
        if float(gains.min()) == 1.0 and float(prev.min()) == 1.0:
            return out_i16
        n_prev = int(prev.shape[0])
        # 前ブロックの原点は「窓数 x 窓長」ではなく **emit 長(= hop) ぶん手前**。
        # speech_probs は ceil(block_len/512) 窓へゼロパディングするので、block_len が
        # 512 の倍数でないと窓の総長がブロック長を超える(例: block_ms=80 で 96ms)。
        # n_prev*step でずらすとその差だけマスクが早まる(80ms 設定で 16ms)。
        centers = np.concatenate(
            [
                (np.arange(n_prev, dtype=np.float64) + 0.5) * step - n,
                (np.arange(gains.shape[0], dtype=np.float64) + 0.5) * step,
            ]
        )
        all_gains = np.concatenate([prev, gains])
        # `prev` は 1 ブロックぶんしか持たないので、`delay_samples` が hop を超える
        # 設定(例: block_ms=80 かつ crossfade_ms=70)では emit の頭が最初の窓中心より
        # 左に出て `prev[0]` へ clamp される。連続なので click にはならないが、その
        # 区間のマスクは 2 ブロック前の情報を持たない。
        gain = np.interp(
            np.arange(n, dtype=np.float64) - delay_samples, centers, all_gains
        )
        out_f = out_i16.astype(np.float32) * gain
        return np.clip(np.rint(out_f), -32768.0, 32767.0).astype(np.int16)
