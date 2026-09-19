# 0092. ストリーミング入力エンベロープに lift 方向（負の `envelope_strength`）を許す

- Status: Accepted (extends [0018](0018-rvc-envelope-duck-only.md))
- Date: 2026-08-29
- Related: [0017](0017-rvc-input-envelope-shape-transfer.md), [0018](0018-rvc-envelope-duck-only.md), [0057](0057-streaming-input-envelope-rolling-ema.md), [0065](0065-streaming-envelope-seam-handover.md), [0093](0093-envelope-reference-follows-speech-only.md)

## Context

[0057](0057-streaming-input-envelope-rolling-ema.md) の入力エンベロープ追従は「入力が静かな所では出力も静かにする」ダックである。配信用途では逆が欲しい: **発話の中で埋もれる小さい所を持ち上げ、上下差を詰めたい**。

ゲイン則は既に `gain = clip(shape^strength, min_gain, max_gain)` であり、dB で書くと `gain_dB = strength * shape_dB` — つまり `strength` は**傾き**で、求めているものはその傾きを負にすることに等しい。しかし実装は `strength <= 0.0` を早期 return の条件にしており、負の値は黙って恒等になっていた。設定側も `ge=0` で拒否していた。

「入力の形状で出力を持ち上げてよいのか」は自明ではない。[0018](0018-rvc-envelope-duck-only.md) は対称ブースト（`max_gain=4.0`）がフルスケール出力をクリップさせる（実測 1864 サンプルが ±32767 に張り付き）ことを理由にゲインをダック（≤1.0）へ限定している。持ち上げが意味を持つのは「入力が静かな区間では RVC 出力も静かで、ヘッドルームがある」場合だけである。これを実測した。

- **持続的なレベル変化には出力は追従しない。** 合成有声信号を 0 → −36dB の階段で 1.9 秒ずつ保持し、実モデル（`Uyu-V4S-16-dynamo`, rmvpe, block 160 / context 500 / lookahead 160）へ通すと `out_dB = 0.065 * in_dB`。入力 36dB の幅が出力では 3.3dB に潰れる。RVC は自前で持続レベルを正規化してしまう。
- **発話内の短期ダイナミクスはほぼ 1:1 で通る。** 実マイク録音（`onset_repro.wav`、16k / 12 秒）を同じ経路へ通し、emit 遅延（10080 サンプル = 210ms）で整列して 25ms フレームで比べると `out_dB = 0.955 * in_dB`、corr **+0.805**。
- **ヘッドルームは 2.3dB**（emit ピーク 25104 = フルスケールの 0.766）。

参照は入力平均 RMS の rolling EMA なので、`shape` が表しているのは**持続レベルを取り除いた短期の偏差**である。つまり `shape` が動く時間スケールは、出力が 0.955 の傾きで追従する側のスケールと一致する。モデルが既に正規化してしまうスケールは参照側が吸収する。

## Decision

`envelope_strength` に**負の値を許し、符号を整形の向きとする**。

- `strength > 0` = **duck**（[0057](0057-streaming-input-envelope-rolling-ema.md) の追従。`max_gain=1.0` 既定で [0018](0018-rvc-envelope-duck-only.md) のダック限定を維持）
- `strength == 0` = 恒等
- `strength < 0` = **lift**（静かな所を参照へ向けて持ち上げ、大きい所を下げる = 圧縮）

- 実装は早期 return の条件を `strength <= 0.0` から `strength == 0.0` へ変えるだけで、整形の当て方（継ぎ目引き継ぎ + emit 遅延補正, [0065](0065-streaming-envelope-seam-handover.md)）は共有する。**専用のコンポーネントは作らない**: 一番難しい部分を二重に持つことになり、片方だけ直る事故を生む。
- `shape` に下限 `_SHAPE_FLOOR = 1e-6` を敷く。デジタル無音のフレームは `0 ** 負` = inf を生み、numpy が warning を出す（無音が続く限りブロックごとに）。clip が値は隠すが warning は残る。1e-6 は −120dB で int16 の全ダイナミックレンジ（約 96dB）より下なので、既に無音のフレームにしか当たらず、duck 方向の挙動は変わらない（0 も下限も `min_gain` へクランプされる）。
- **`min_gain` / `max_gain` の意味は向きによって入れ替わる**（値の意味は「クランプ」のままで変わらない）。lift では `max_gain` が持ち上げの上限で、同時に**ノイズ増幅のガード**になる: ノイズは発話より 30〜40dB 下 = `shape` 0.01〜0.03 なので、`strength=-0.3` なら素で 3.5〜4.6 倍まで持ち上がる。相対値なので [0017](0017-rvc-input-envelope-shape-transfer.md) の mic-gain 非依存は保たれる。
- 既定は据え置き（`strength=1.0` / `max_gain=1.0`）。**既定構成の出力はビット単位で不変**で、lift は明示的に負を書いたときだけ有効になる。

## Alternatives rejected

- **出力自身のエンベロープを検出して圧縮する（本物のレベラ / AGC）** — [0017](0017-rvc-input-envelope-shape-transfer.md) が `÷shape_out` を却下したのと同じ理由で、静音区間でゲインが飽和しノイズフロアを持ち上げる。加えて上の実測どおり入力側の形状が出力レベルの妥当な代理になっており、検出器を増やす理由が無い。
- **`envelope_mode` のようなモードフラグを足して strength は正のまま** — ノブが 1 つ増えるだけで表現力は同じ。dB 上で傾きである以上、符号がそのまま向きであるほうが読み手の負担が小さい。
- **プリ減衰 + メイクアップでヘッドルームを作る** — 全体のピークを犠牲にする。lift では `gain>1` になるのは `shape<1` の所（= 出力も静かな所）だけで loud 部は `gain<1` に下がるため、ピークは伸びない。実測（`onset_repro.wav` 再生でのオフライン適用）でもクリップは 0 サンプルだった。
- **時定数（attack/release）を同時に入れる** — 「上下差を詰める」意味の平滑化は指数だけで効く。ポンピングを均す時定数は別の問題で、必要になってから別の決定として足す。

## Consequences

発話内で埋もれる小さい部分が持ち上がり、`|strength|` が圧縮の強さになる。既定 off・既定符号のままなら出力は不変。lift を使うときは `max_gain > 1` が必須で（そうしないと持ち上がらない）、その値がノイズ増幅の上限を決める。ヘッドルームは 2.3dB しかないので `max_gain` を大きく取りすぎると loud 部がクリップし得る。

lift 方向は**参照レベルの誤差に対して duck 方向より脆い**: `shape` が 1 より大きい側へ張り付くと、duck では「整形しない」（`max_gain=1.0` で頭打ち）で済むのに対し、lift では**フレーズ全体が一律に減衰する**。この脆さが顕在化した実測と、その対処は [0093](0093-envelope-reference-follows-speech-only.md) にある。本 ADR 単独では sparse な発話で使い物にならない。

Status を `Accepted` としたのは、向きの妥当性（0.955 の追従）とクリップしないことが実測で確定し、退行テスト（lift / attenuate / 無音フレームの inf ガード）が入ったため。整形量（`strength` / `max_gain`）の詰めは実機耳確認として別に回す。
