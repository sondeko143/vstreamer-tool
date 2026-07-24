# 0057. ストリーミング VC の入力エンベロープ追従を rolling-EMA 参照で行う

- Status: Accepted
- Date: 2026-07-24
- Related: [spec](../superpowers/specs/2026-07-24-streaming-input-envelope-follow-design.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0017](0017-rvc-input-envelope-shape-transfer.md), [0018](0018-rvc-envelope-duck-only.md), [0054](0054-stream-vc-config-section.md)

## Context

ストリーミング VC の出力が入力音声の音量エンベロープに追従せず、アタック/ディケイが急峻に聞こえる([spec](../superpowers/specs/2026-07-24-streaming-input-envelope-follow-design.md))。発話系(バッチ)は `apply_input_envelope`([0017](0017-rvc-input-envelope-shape-transfer.md)/[0018](0018-rvc-envelope-duck-only.md))で入力の相対ラウドネス包絡を出力へ転写している:入力 RMS を**発話全体の平均**で正規化(平均 1 = マイクゲイン非依存の相対形状)し、`clip(shape^strength, min_gain, max_gain)` のダックゲインとして出力へ掛ける。

これをストリーミングへ持ち込むときの中心的な問題は「何を参照レベルに正規化するか」である。ストリーミングは 160ms 程度の固定ブロックが単独で到着し、バッチが使う「発話全体の平均」が手に入らない。ブロック自身の平均で正規化すると、丸ごと減衰尾(decay tail)であるブロックが自分の平均 1 に正規化されてしまい、直前の loud ブロックに対してダックされず、**ブロック跨ぎのダイナミクスが平坦化**して追従にならない。固定の絶対参照はマイクゲインに依存する。

## Decision

入力ブロックごとの相対ラウドネス包絡を、**入力平均 RMS の rolling EMA**(時定数 `envelope_ema_ms`、既定 ~2s)を参照に正規化し、`clip(shape^strength, min_gain, max_gain)` のダックゲイン(既定 `max_gain=1` = 減衰のみ)として出力ブロックへ掛ける。バッチ `apply_input_envelope` と同じダック思想だが、参照を「発話全体の平均」→「rolling EMA」に置換したストリーミング版とする。rolling EMA は 1 スカラの状態で、ブロック内(フレーム RMS の形状)とブロック跨ぎ(参照が安定に持続)の双方のダイナミクスに追従する。

- 実装はストリーミング runner + 専用の pure コンポーネント(`gate.py` の `StreamingVadGate` と同型、numpy はメソッド内)で行い、**発話系 `worker/vc.py` と StreamingVc コアは無改変**([0053](0053-streaming-vc-fixed-block-crossfade.md) の「内部部品再利用・発話系温存」を踏襲)。
- 参照形状は**素の入力ブロック**(input_boost 適用前)から取る(input_boost は一様ゲインで形状[比]を変えないため raw でも boosted でも同じ。VAD 判定が素の入力を見るのと揃える)。
- 適用順はバッチの `envelope → gate` を踏襲し、**VAD ゲートの前**に掛ける。ゲートとは併用する(役割が別:ゲート=無音時ノイズ抑制、エンベロープ=発話中の attack/decay 整形)。
- 設定は `[stream_vc]`([0054](0054-stream-vc-config-section.md))に `envelope_follow`(既定 false)ほかを足す。**既定 off のとき `apply` は恒等で出力はビット単位で不変**(`input_boost=1.0` / `vad_gate=false` と同じ規律)。
- EMA 状態は `_reset_context()`(warmup / pause-resume / capture 再 open / session 変更)と同じライフサイクルでリセットする。

## Alternatives rejected

- **ブロック自身の平均で正規化(バッチ literal)** — 160ms ブロックでは丸ごと減衰尾のブロックが自分の平均 1 に正規化されて直前の loud ブロックに対してダックされず、ブロック跨ぎのダイナミクスを平坦化する。attack/decay 追従にならない。
- **固定の絶対参照レベル** — マイクゲインに依存し(バッチの mic-invariance を失う)、マイク/セットアップ変更のたびに再調整が要る。
- **出力自身のエンベロープで割る** — バッチが明示的に却下した手法([0017](0017-rvc-input-envelope-shape-transfer.md))。静音部でゲインが飽和してノイズフロアを持ち上げ、loud 部を逆重みする(`max_gain>1` で compressor 化)。ストリーミングでも採らない。
- **VAD ゲートを置き換える** — エンベロープ追従は rolling 参照ゆえ長い無音では参照がノイズ床へ寄り、無音時ノイズ抑制は担えない。ゲートとは役割が別で併用する。
- **出力/入力の時間ずれ(SOLA + 特徴抽出の受容野, ~20ms)を補償** — 緩慢なゲインには可聴影響が小さい(160ms ブロックの ~12%)。measure-first で v1 は非補償とし、耳確認で問題化したら補償を検討する。

## Consequences

attack/decay がバッチ並みに入力へ追従する(opt-in)。rolling EMA は 1 スカラ状態で安く、GPU/モデルを要さない(CPU で単体テスト可能)。参照が rolling ゆえ長い無音では参照がノイズ床へ寄り、ノイズ抑制は担えない(VAD ゲート併用が前提)。`envelope_ema_ms` が短すぎると loud onset で参照が跳ねて追従が過敏になり、長すぎるとレベル変化に鈍い → 実測で調整する。出力ブロックは入力ブロックから ~20ms 時間ずれるが v1 では非補償。既定 off で出力はビット単位で不変。Status は `Accepted`(実装 + 全コミット independent review + 実機で連続発話の attack/decay 整形を確認)。[0053](0053-streaming-vc-fixed-block-crossfade.md) を extend する。
