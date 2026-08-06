# 0065. ストリーミング入力エンベロープを継ぎ目引き継ぎ + emit 遅延補正にする（0057 の適用部を refine）

- Status: Accepted (refines 0057)
- Date: 2026-08-06
- Related: [0057](0057-streaming-input-envelope-rolling-ema.md), [0059](0059-stream-vc-window-resolution-vad-gate.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0062](0062-log-throttle-time-based-episodes.md)

## Context

[0057](0057-streaming-input-envelope-rolling-ema.md) が入れた `StreamingEnvelope` は、入力ブロックのフレーム RMS 形状を**ブロック内で正規化した 0..1 軸**(`src_x`/`dst_x`)で出力ブロックへ補間していた。この写像はブロックを跨いだ状態を持たないので、ブロック k の最終サンプルのゲイン `frame_rms_k[-1]/ref_k` と、ブロック k+1 の先頭サンプルのゲイン `frame_rms_{k+1}[0]/ref_{k+1}` が無関係になる(参照 EMA 自体も毎ブロック動く)。結果としてブロック境界ごとに**ゲインの段差**が入る。

実測(実機設定 `strength=1.0 / min_gain=0.4 / max_gain=0.9 / ema_ms=4000 / block_ms=160`、音声様信号 20 秒 = 124 継ぎ目):

- 段差 > 0.05 の継ぎ目 **33/124 (27%、約 1.7 回/秒)**、平均 |段差| 0.085
- 最大 |段差| **0.500 = レール間フルスイング = 1 サンプルで +7.0 dB**
- フルスケールの RVC 出力を模した波形では、継ぎ目のサンプル間跳躍が 8613 LSB = ストリーム中央値の 36 倍

emit 自体は SOLA + クロスフェードで連続に作られており([0053](0053-streaming-vc-fixed-block-crossfade.md))、playback はブロックを隙間なく書き出すので、この段差はそのままスピーカへ出るクリックである。同じ問題を VAD ゲートは `_prev_gains` の引き継ぎで明示的に潰していた([0059](0059-stream-vc-window-resolution-vad-gate.md):「Stepping the gain at a boundary is itself a click」)。エンベロープ側だけ対策が無かった。

さらに、継ぎ目を**厳密に**連続にするには emit 遅延補正が必要だと分かった。emit は入力に対して `emit_delay_samples` 遅れる(emit サンプル j はブロック相対時刻 `j - delay` の音)。補正すると、ブロック k のクエリ範囲 `[-d, out_len-1-d]` とブロック k+1 の `[out_len-d, 2*out_len-1-d]` が共有時間軸上で隙間なく連続し、かつ継ぎ目が形状の**内側**に落ちるので、両ブロックが同じ 2 つのフレーム中心を同じ値で補間する。補正しないと継ぎ目が端(片側クランプ)に落ちて半段差が残る。[0057](0057-streaming-input-envelope-rolling-ema.md) はこの補正を「~20ms、160ms ブロックの ~12%、緩慢なゲインには可聴影響が小さい」として v1 で却下していたが、[0059](0059-stream-vc-window-resolution-vad-gate.md) の実測でずれは **50ms (31%)** と判明しており、前提が違っていた。

## Decision

`StreamingEnvelope.apply` を、VAD ゲートのマスク重畳(`StreamingVadGate.apply`)と**同一の構成**にする。

- **前ブロックの形状を引き継ぐ。** `_prev_shape` / `_prev_len` を状態として持ち、前ブロックのフレーム中心を 1 emit 長ぶん手前(負の位置)に置いて現ブロックの中心と連結してから補間する。`prev_len` はブロックごとに持ち回し、emit 長が変わっても前ブロックの原点が黙ってずれないようにする。
- **emit 遅延を補正する。** `apply` は `delay_samples`(= `StreamingVc.emit_delay_samples`)を**必須引数**で受け、`arange(out_len) - delay_samples` で問い合わせる。既定値を置かない: 既定 0 は「渡し忘れ」を黙って元のバグへ戻すため。[0057](0057-streaming-input-envelope-rolling-ema.md) の「時間ずれは v1 では非補償」を本 ADR で上書きする。
- **cold start とリセットは unity から引き継ぐ。** `_prev_shape is None`(起動直後 / `reset()` 直後)のときは 1 emit 分のフレームを 1.0 で埋めて引き継ぐ。`_ema_level` の「最初のブロックはダックしない」cold start と揃う。参照が digital silence でブロックを素通しした場合も、実際に適用したゲイン(= 1.0)を引き継ぐ。
- `reset()` は `_prev_shape` も落とす(実時間ジャンプ後の emit 先頭は zeros 文脈からのレンダなので、跳ぶ前の形状で整形すると別の音を整形することになる)。
- 正規化 0..1 軸は捨てる。フレーム中心は emit の**絶対サンプルグリッド**に置く。

## Alternatives rejected

- **ブロック内正規化のまま据え置き([0057](0057-streaming-input-envelope-rolling-ema.md) v1)** — 上記の実測どおり継ぎ目に最大 +7 dB の段差が入り、ブロックレート(6.25Hz)のクリックになる。`envelope_min_gain`/`max_gain` を狭めれば段差は小さくなるが(実機 0.4/0.9 で 0.5)、それは整形量そのものを削ることであり対策ではない。
- **emit 遅延補正なしで引き継ぎだけ入れる** — 継ぎ目が形状の端に落ち、ブロック k の尾はクランプ・ブロック k+1 の頭は補間になるので半段差が残る(実測 delay=0 で最大 0.32、25/124 継ぎ目が > 0.05)。引き継ぎの効果を半分しか得られない。
- **フレーム中心でなく端点写像にする**(中心を `i/(n-1)*out_len` に置く) — 遅延に依らず連続にできるが、フレーム RMS が代表する時刻(= 中心)から半フレームずれた位置に値を置くことになり、バッチ `apply_input_envelope` とも VAD ゲートとも写像がずれる。物理的に正しい中心写像を保ち、遅延の下限は境界条件として文書化する方を採る。
- **エンベロープ側にもクロスフェードを持たせる**(前ブロックのゲイン曲線と現ブロックのそれを継ぎ目でクロスフェード) — 引き継ぎ補間と等価な効果を、専用の窓と長さという新しいノブを足して実現することになる。ゲートと同型の構成の方が読み手の負担が小さい。

## Consequences

継ぎ目の段差が最大 0.00087 まで下がり、**ブロック内の最大サンプル間変化 (0.00167) を下回る** = 継ぎ目が他のサンプルと区別できなくなる(実機設定・同一プローブ、0.05 超の継ぎ目 33/124 → 0/124)。同時に整形が音に対して 50ms 早く当たるようになり、[0057](0057-streaming-input-envelope-rolling-ema.md) が残した時間ずれが解消する。

`delay_samples` には上下 2 つの境界がある。大きすぎる(emit 長超)と emit 先頭が `prev_shape[0]` にクランプされ、その区間は 2 ブロック前の情報を持たない(ゲートのマスクと同じ境界)。小さすぎる(半フレーム未満)と継ぎ目に半段差が残る。検証済みジオメトリは双方に余裕がある(遅延 50ms 対 半フレーム 13ms、`crossfade_ms=0` でも HuBERT の ~20ms 切り詰めぶんが残る)ので、機構を足さずコード上のコメントで境界を明示するに留める。`envelope_window_ms` を `block_ms` へ近づけると余裕が縮む。

`envelope_follow=false`(既定)のとき `apply` は呼ばれないので、既定構成の出力はビット単位で不変。有効時の音は変わる(段差の除去 + 50ms のずれ解消)ので**実機耳確認が要る**。Status を `Accepted` としたのは、段差の除去が数値で確定し([0053](0053-streaming-vc-fixed-block-crossfade.md) の emit 連続性と `_prev_gains` 引き継ぎという既存の判断と同型)、退行テスト(継ぎ目連続性 / 遅延シフト / carry 落とし)が入ったため。耳確認は整形量(`envelope_strength` / `envelope_ema_ms`)の調整として別に回す。

なお本 ADR は整形の**当て方**だけを変える。参照レベルを rolling EMA とする [0057](0057-streaming-input-envelope-rolling-ema.md) の中核判断はそのまま有効で、supersede ではなく refine である。
