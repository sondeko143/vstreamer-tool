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
    # Type-only: used by StreamingVc's annotations below. The pure helpers
    # (next_context / slice_block_output) don't need these, so keeping the
    # imports under TYPE_CHECKING (rather than module-level) still lets this
    # module import on a CPU machine without torch/onnxruntime/the rvc extra.
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


def equal_power_weights(n: int):
    """長さ n の等電力クロスフェード重み `(fade_in, fade_out)`。

    セル中心の sin/cos なので `fade_in**2 + fade_out**2 == 1`。独立推論した
    (無相関の)隣接出力を混ぜても総電力が一定に保たれる — RVC デコーダは
    ステートレスで hop ごとに位相非整合なので、線形重みより等電力が正しい。
    `n <= 0` は空配列を返す(crossfade 無効)。
    """
    import numpy as np

    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    x = (np.arange(n, dtype=np.float32) + 0.5) / n
    fade_in = np.sin(0.5 * np.pi * x).astype(np.float32)
    fade_out = np.cos(0.5 * np.pi * x).astype(np.float32)
    return fade_in, fade_out


def overlap_add(prev_tail, head, fade_in, fade_out):
    """`prev_tail` をフェードアウト・`head` をフェードインして加算する。

    等電力 overlap-add の 1 行。要素積なので numpy 配列でも torch tensor でも
    動く(呼び出し側は同じ長さ・同じ域で渡す)。
    """
    return prev_tail * fade_out + head * fade_in


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
        return slice_block_output(out, self.block_len, seq.shape[0])
