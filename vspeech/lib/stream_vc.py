"""ストリーミング VC の再利用コア(ADR-0053)。

固定長ブロックを rolling 左文脈と連結してステートフル変換する。既存の
`change_voice` の内部部品(HuBERT 特徴量 / f0 / infer / int16化)をそのまま
再利用し、発話系の `change_voice` 経路は無改変で温存する。ブロック境界は
クロスフェードで繋ぎ(SOLA on なら振幅保存・和=1、SOLA off なら等電力)、
混ぜる前に SOLA で位相を合わせる(`_emit_with_crossfade`)。

純粋ヘルパ(next_context / crossfade_weights / overlap_add /
sola_offset)は numpy でも torch tensor でも動くよう `len(seq)` ベースにしてあり(ただし
sola_offset は numpy 配列専用)、torch 無し・
rvc extra 無しの CPU でも import できる(重い import は StreamingVc のメソッド内
でのみ行う)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: used by StreamingVc's annotations below. The pure helpers
    # (next_context / crossfade_weights / overlap_add)
    # don't need these, so keeping the imports under TYPE_CHECKING (rather
    # than module-level) still lets this module import on a CPU machine
    # without torch/onnxruntime/the rvc extra.
    import numpy as np
    import torch
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession

    from vspeech.config import RvcConfig
    from vspeech.lib.rvc import HubertSession


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


def crossfade_weights(
    n: int, *, correlated: bool
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """長さ n のクロスフェード重み `(fade_in, fade_out)`。フェード則は隣接描画の
    相関(= SOLA の有無)で切り替える。`correlated` は必須のキーワード専用引数。

    - `correlated=True`(SOLA on): セル中心の sin²/cos²(= `(1-cos)/2` 型)で
      **振幅保存**(`fade_in + fade_out == 1`)。SOLA が位相を合わせて隣接描画を
      意図的に相関させる(整列点相関 ρ≈0.82)ので、相関信号どうしは和=1 で
      混ぜるとユニティ利得になる。ここで等電力(²和=1)を使うと seam が過剰
      加算される(SOLA on 再測定で +1.14dB, ADR-0053)。w-okada VCClient も
      SOLA には sum-to-1 を組む。
    - `correlated=False`(SOLA off, `sola_search_len == 0`): sin/cos で **等電力**
      (`fade_in² + fade_out² == 1`)。SOLA を切ると隣接描画は無相関(ρ≈0)で、
      無相関どうしを和=1 で混ぜるとクロスフェード帯に約 -1.25dB のノッチ(=
      block レートの微小トレモロ)が出る。等電力なら無相関加算でも合成パワーが
      平坦になり、かつ SOLA 導入前の出力に等電力則として float32 の丸め(~1 ULP,
      max|Δ|≈1.19e-7)まで一致する。厳密なビット一致ではない — 重みを
      `theta=0.5πx; sin(theta)` と先に畳むので pre-SOLA の `sin(0.5πx)` 直書きと
      ~1 ULP ずれる — が、int16 量子化より遥かに下で可聴でない(計算は無害なので
      変えない)。

    `n <= 0` は空配列を返す。
    """
    import numpy as np

    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    theta = 0.5 * np.pi * ((np.arange(n, dtype=np.float32) + 0.5) / n)
    if correlated:
        fade_in = (np.sin(theta) ** 2).astype(np.float32)
        fade_out = (np.cos(theta) ** 2).astype(np.float32)
    else:
        fade_in = np.sin(theta).astype(np.float32)
        fade_out = np.cos(theta).astype(np.float32)
    return fade_in, fade_out


def overlap_add(prev_tail, head, fade_in, fade_out):
    """`prev_tail` をフェードアウト・`head` をフェードインして加算する。

    クロスフェード overlap-add の 1 行。要素積なので numpy 配列でも torch tensor でも
    動く(呼び出し側は同じ長さ・同じ域で渡す)。
    """
    return prev_tail * fade_out + head * fade_in


# SOLA を諦める tail の RMS 下限(int16 単位のフルスケール 32768 に対する比)。
# 1e-4 = -80dBFS 相当。従来の絶対値 1e-9 判定は「完全なデジタル無音」でしか発火せず、
# 現実のノイズフロア(実測 RMS 0.000298 * 32768 ≈ 9.8 int16 単位)ではノイズ同士を
# 相関させて argmax が事実上ランダムな lag を選んでいた。位相を合わせる相手が
# そもそも無い領域では探索せず、公称(シフト無し)位置を返すのが正しい。
_SOLA_MIN_RMS = 32768.0 * 1e-4

# 相関がほぼ平坦なときに中央(公称 lag)へ寄せるための微小バイアス。正規化相関は
# [-1, 1] なので、探索半幅で正規化した最大 1e-3 のペナルティは本物のピーク
# (実測 0.89 vs 0.02)には影響せず、真に同点な面(例: DC 一定領域で全 lag の
# 正規化相関が 1.0)でだけ効く。無しだと argmax が常に index 0 = 探索窓の
# 最も手前 = 最大の負シフトを選んでしまう。
_SOLA_CENTER_BIAS = 1e-3


def sola_offset(prev_tail, region):
    """`region` 内で `prev_tail` と最も相関する開始位置(index)を返す。

    RVC デコーダはステートレスなので、同じ入力区間でも tick ごとに位相が
    ずれた波形を返す(実測: lag0 の相関 -0.02 に対し最適 lag では 0.89)。
    固定位置で混ぜると同じ波形を数 ms ずらして足すことになり、櫛形フィルタ
    (comb)になって「扇風機」的な音になる。混ぜる前に位相を合わせる。

    戻り値は `region` 先頭からの index(0 <= i <= len(region)-len(prev_tail))。
    シフト無し(公称)に相当するのは **index 0 ではなく中央** `(len(region)-n)//2`
    ── 呼び出し側は探索半幅ぶん手前から region を切り出すため。したがって
    「探索を諦める」ときは 0 ではなく中央を返す:

    - `prev_tail` の RMS が `_SOLA_MIN_RMS` 未満(実質無音)→ 中央。合わせるべき
      位相が無く、ノイズ同士の相関の argmax は任意の lag になるため。
    - 相関面がほぼ平坦(同点)→ `_SOLA_CENTER_BIAS` で中央寄りが勝つ。

    `region` が `prev_tail` より短いときだけは窓が 1 つも取れないので 0。
    """
    import numpy as np

    n = len(prev_tail)
    if n == 0 or len(region) < n:
        return 0
    center = (len(region) - n) // 2  # シフト無し(公称)に相当する index
    tail_rms = float(
        np.sqrt(np.mean(np.square(np.asarray(prev_tail, dtype=np.float64))))
    )
    if tail_rms < _SOLA_MIN_RMS:
        return center
    tail_norm = float(np.linalg.norm(prev_tail))
    win = np.lib.stride_tricks.sliding_window_view(region, n)
    num = win @ prev_tail
    den = np.linalg.norm(win, axis=1) * tail_norm + 1e-9
    score = num / den
    # 中央からの距離に比例した微小ペナルティ。探索半幅で割ってあるので窓長や
    # サンプルレートに依らず最大 _SOLA_CENTER_BIAS。O(len(region)) で安い。
    lags = np.arange(score.shape[0])
    score = score - _SOLA_CENTER_BIAS * np.abs(lags - center) / max(center, 1)
    return int(np.argmax(score))


class StreamingVc:
    """固定ブロック + rolling 左文脈のステートフル VC(ADR-0053)。

    毎 tick `[context | block]`(16kHz)を組み立て、既存 `change_voice` の内部
    部品で HuBERT 特徴量 -> f0 -> infer -> int16 を通し、ブロック相当の出力だけ
    採用して context を更新する。block_len / context_len を固定するので入力
    shape が固定になり、warmup は 1 回で済む(以後 re-autotune なし)。

    重い依存(torch / rvc の内部部品)はここで初めて import する。`rvc_config`
    の f0_extractor_type は渡す `f0_session` と一致していること。
    """

    def __init__(
        self,
        rvc_config: RvcConfig,
        device: torch.device,
        hubert_model: HubertSession,
        session: InferenceSession,
        f0_session: InferenceSession | None,
        target_sample_rate: int,
        f0_enabled: bool,
        emb_output_layer: int,
        use_final_proj: bool,
        block_len: int,
        context_len: int,
        crossfade_len: int = 0,
        sola_search_len: int = 0,
    ) -> None:
        import torch

        from vspeech.lib.rvc import _is_model_half

        self.rvc_config = rvc_config
        self.device = device
        self.hubert_model = hubert_model
        self.session = session
        self.f0_session = f0_session
        self.target_sample_rate = target_sample_rate
        self.f0_enabled = f0_enabled
        self.emb_output_layer = emb_output_layer
        self.use_final_proj = use_final_proj
        self.block_len = block_len
        self.context_len = context_len
        self._is_half = _is_model_half(session)
        self._sid = torch.tensor(0, device=device).unsqueeze(0).long()
        self._context = torch.zeros(context_len, device=device, dtype=torch.float32)

        self.crossfade_len = crossfade_len
        # SOLA 探索半幅(16kHz 入力サンプル)。0 で SOLA 無効 = 固定位置の従来挙動。
        self.sola_search_len = sola_search_len
        # crossfade の出力域長(hop / crossfade / SOLA 探索半幅)は実時間クロック
        # (`* target_sample_rate / 16000`)から導く。描画長 out.shape[0] からの比率
        # 導出は HuBERT の受容野ぶん(約 320 入力サンプル)短く出て sink を飢えさせる。
        # 読み出し位置だけは out.shape[0] からの逆算(末尾アンカー)を維持し、
        # 切り詰められた末尾を避ける。長さは毎tick一定 → 初回 emit で算出しキャッシュ。
        self._xfade_cache: (
            tuple[int, int, int, NDArray[np.float32], NDArray[np.float32]] | None
        ) = None
        self._output_tail = None  # 初回 crossfade で zeros(out_xf) を遅延生成
        if crossfade_len > 0 and context_len < crossfade_len:
            raise ValueError(
                "context_len must be >= crossfade_len for context-overlap crossfade"
            )
        if crossfade_len > 0 and crossfade_len >= block_len:
            raise ValueError("crossfade_len must be < block_len")

    def warmup(self, n: int = 3) -> None:
        """zeros ブロックで ONNX グラフ / CUDA カーネルを先に構築する。

        block_len は固定なので、実値でなく shape さえ通れば以後 stall しない。
        warmup 後は context を zeros に戻す。
        """
        import numpy as np

        zeros = np.zeros(self.block_len, dtype=np.float32)
        for _ in range(n):
            self.process_block(zeros)
        self._reset_context()

    def _reset_context(self) -> None:
        import torch

        self._context = torch.zeros(
            self.context_len, device=self.device, dtype=torch.float32
        )
        # crossfade tail も rolling 状態。warmup 後の stale tail が最初の実ブロックの
        # seam に漏れないようリセット(次 emit で zeros に再初期化 → 無音から fade-in)。
        self._output_tail = None
        self._xfade_cache = None

    def process_block(self, block: NDArray[np.float32]) -> NDArray[np.int16]:
        """長さ block_len の 16kHz float32 [-1,1] を変換し int16 ブロックを返す。"""
        import numpy as np
        import torch

        from vspeech.lib.rvc import _align_pitch_to_feats
        from vspeech.lib.rvc import _extract_hubert_feats
        from vspeech.lib.rvc import _select_pitch
        from vspeech.lib.rvc import _to_int16
        from vspeech.lib.rvc import infer

        block_t = torch.from_numpy(np.ascontiguousarray(block)).to(
            device=self.device, dtype=torch.float32
        )
        seq = torch.cat([self._context, block_t])  # 固定長 L = context_len + block_len

        feats = _extract_hubert_feats(
            hubert_model=self.hubert_model,
            audio_pad=seq,
            device=self.device,
            emb_output_layer=self.emb_output_layer,
            use_final_proj=self.use_final_proj,
        )

        p_len = seq.shape[0] // self.rvc_config.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        pitch, pitchf = _select_pitch(
            audio_pad=seq,
            rvc_config=self.rvc_config,
            f0_enabled=self.f0_enabled,
            p_len=p_len,
            device=self.device,
            f0_session=self.f0_session,
        )

        feats_len = feats.shape[1]
        pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
        p_len_tensor = torch.tensor([feats_len], device=self.device).long()

        with torch.inference_mode():
            audio_i16 = _to_int16(
                infer(
                    is_half=self._is_half,
                    session=self.session,
                    feats=feats,
                    pitch_length=p_len_tensor,
                    pitch=pitch,
                    pitchf=pitchf,
                    sid=self._sid,
                )[0]
            )

        out = audio_i16.detach().cpu().numpy()
        self._context = next_context(seq, self.context_len).detach()
        if self.crossfade_len > 0:
            return self._emit_with_crossfade(out)
        # crossfade 無効時も長さは実時間クロック由来。描画長からの比率導出は
        # HuBERT の受容野ぶん(約 320 入力サンプル)短く出て sink を飢えさせる。
        # 位置は末尾アンカーのままなので、切り詰められた末尾は避けられる。
        out_hop = round(self.block_len * self.target_sample_rate / 16000)
        return out[-out_hop:] if out.shape[0] >= out_hop else out

    def _emit_with_crossfade(self, out: NDArray[np.int16]) -> NDArray[np.int16]:
        """SOLA で位相を合わせてから overlap-add し、実時間 hop ちょうどを返す。

        出力域の長さ(hop / crossfade / SOLA 探索半幅)は**実時間クロック**から導く
        (`block_len * target_sample_rate / 16000` など)。描画長 out.shape[0] からの
        比率導出は誤りで、HuBERT の受容野が末尾を一定量(約 320 入力サンプル)切り
        詰めるぶんだけ hop が短くなり、sink を永続的に飢えさせる(実測 3.03% =
        30.3ms/s)。長さは shape 固定なら毎tick一定なので初回に算出してキャッシュ。

        一方、**読み出し位置は out_total からの逆算(末尾アンカー)**のまま。これに
        より切り詰められた末尾は自然に避けられる。

        前 tick が emit 末尾 out_xf を `_output_tail` に保持しており、今 tick はそれと
        同じ入力時刻を再描画した区間を振幅保存(和=1)ブレンドして emit 先頭にする(seam の真の
        overlap-add)。ただし RVC デコーダはステートレスで、同じ入力区間でも tick ごとに
        数 ms の位相ずれが乗る(実測: lag0 の相関 -0.02 に対し最適 lag では 0.89)。
        固定位置で混ぜると comb フィルタになるので、混ぜる前に `sola_offset` で
        `_output_tail` と最も相関する読み出し位置 `start` を ±out_sola の窓で探す
        (SOLA = Synchronous OverLap-Add)。フェード則は SOLA の有無で切り替える
        (`crossfade_weights(..., correlated=self.sola_search_len > 0)`)。SOLA on は
        隣接描画を意図的に相関させる(整列点相関 ρ=0.82)ので振幅保存(和=1、
        sin²/cos²)でユニティ利得(等電力だと +1.14dB 過剰加算、sum-to-1 は
        -0.76dB でユニティ寄り。w-okada VCClient も SOLA に sum-to-1)。SOLA off
        (`sola_search_len == 0`)は隣接描画が無相関(ρ≈0)なので等電力(²和=1、
        sin/cos)。無相関に和=1 を使うと帯域に約 -1.25dB のノッチ(block レートの
        トレモロ)が出る。等電力なら pre-SOLA 出力に float32 の丸め(~1 ULP)まで
        一致する(厳密なビット一致ではない; ADR-0053)。

        index 不変量:

        - emit 長は常にちょうど out_hop = `block_len * target_sample_rate / 16000`
          (= 入力 hop を sink のサンプルレートへ写した実時間長。ドリフト無し)。
          lag は「どこから読むか」だけを変え、「どれだけ出すか」は変えない。
        - 触れる最大 index は
          `start + out_hop + out_xf <= (nominal + out_sola) + out_hop + out_xf
          == out_total` なので、描画の外へは決して出ない。
        - `nominal - out_sola >= 0`(探索窓が出力の先頭を割らない)は out_sola の
          clamp `out_sola <= (out_total - out_hop - out_xf) // 2` で保証する。
        - `out_sola == 0`(= `sola_search_len == 0`)のときは `start == nominal ==
          out_total - out_hop - out_xf` で読み出し位置が pre-SOLA と一致し、かつ
          フェード則も等電力(`correlated=False`)に落ちるので、出すサンプルは
          pre-SOLA と float32 の丸め(~1 ULP)まで一致する(重みを先に畳んだぶんの
          ずれで厳密なビット一致ではないが、int16 量子化より下で可聴でない)。
          読み出し位置だけでなく重みも一致させることでこの不変量が成立する。
        - `sola_offset` が探索を諦める(tail が実質無音・相関面が平坦)ときは region
          の**中央** `out_sola` を返すので `start == nominal` になる。すなわち
          「シフト無し」に落ちる(index 0 = 最大の負シフト、ではない)。

        アルゴリズム遅延は out_sola サンプル(読み出し位置を探索半幅だけ手前へ
        ずらすため)に加え、HuBERT の受容野による末尾切り詰めぶん(約 320 入力
        サンプル = 20ms)が乗る。どちらも emit 長には影響しない。
        """
        import numpy as np

        out_total = out.shape[0]
        if self._xfade_cache is None:
            r = self.target_sample_rate
            # 長さは実時間クロックから導く。out_total からの比率で出すと、HuBERT の
            # 受容野で末尾が一定量(約 320 入力サンプル)切り詰められるぶんだけ hop が
            # 短くなり、出力デバイスを永続的に飢えさせる(実測 3.03% = 30.3ms/s)。
            # 読み出し位置は従来どおり out_total からの逆算(末尾アンカー)なので、
            # 切り詰められた末尾は自然に避けられる。
            out_hop = round(self.block_len * r / 16000)
            out_xf = round(self.crossfade_len * r / 16000)
            out_sola = round(self.sola_search_len * r / 16000)
            if out_total < out_hop:
                raise ValueError(
                    f"decoder output ({out_total}) < one hop ({out_hop}): "
                    "context_ms が短すぎる(HuBERT の受容野ぶん実効長が縮む)。"
                    "context_ms を増やすこと。"
                )
            # crossfade 帯は hop 以下 かつ context 区間(out_total-out_hop)以下に抑える
            out_xf = min(out_xf, out_hop, out_total - out_hop)
            # nominal - out_sola >= 0 を保証(探索窓が出力の先頭を割らない)
            out_sola = max(0, min(out_sola, (out_total - out_hop - out_xf) // 2))
            # フェード則は SOLA の有無で決まる。SOLA on は隣接描画が相関する(和=1)、
            # SOLA off は無相関(等電力)。sola_search_len==0 で pre-SOLA と ~1 ULP 一致。
            fade_in, fade_out = crossfade_weights(
                out_xf, correlated=self.sola_search_len > 0
            )
            self._xfade_cache = (out_hop, out_xf, out_sola, fade_in, fade_out)
        out_hop, out_xf, out_sola, fade_in, fade_out = self._xfade_cache
        out_f = out.astype(np.float32)
        if self._output_tail is None:
            self._output_tail = np.zeros(out_xf, dtype=np.float32)
        # 読み出し開始位置。out_sola=0 なら従来と同一 (= out_total-out_hop-out_xf)。
        nominal = out_total - out_hop - out_xf - out_sola
        if out_sola > 0:
            region = out_f[nominal - out_sola : nominal + out_sola + out_xf]
            start = (nominal - out_sola) + sola_offset(self._output_tail, region)
        else:
            start = nominal
        head = out_f[start : start + out_xf]
        blended = overlap_add(self._output_tail, head, fade_in, fade_out)
        middle = out_f[start + out_xf : start + out_hop]
        emit_f = np.concatenate([blended, middle])
        self._output_tail = out_f[start + out_hop : start + out_hop + out_xf].copy()
        return np.clip(np.rint(emit_f), -32768.0, 32767.0).astype(np.int16)
