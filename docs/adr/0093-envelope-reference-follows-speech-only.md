# 0093. 入力エンベロープの参照レベルを「VAD が開けたブロックだけ」で更新し、既定時定数を 1000ms にする

- Status: Accepted (refines [0057](0057-streaming-input-envelope-rolling-ema.md))
- Date: 2026-08-29
- Related: [0057](0057-streaming-input-envelope-rolling-ema.md), [0059](0059-stream-vc-window-resolution-vad-gate.md), [0065](0065-streaming-envelope-seam-handover.md), [0092](0092-streaming-envelope-lift-direction.md)

## Context

[0057](0057-streaming-input-envelope-rolling-ema.md) は参照レベルを「入力平均 RMS の rolling EMA」と決め、その帰結として「長い無音では参照がノイズ床へ寄る」ことを既知の特性として記録していた。[0092](0092-streaming-envelope-lift-direction.md) の lift 方向を実装する過程で、これが**特性ではなく機能を殺す欠陥**であることが実測で分かった。

実マイク録音（`onset_repro.wav`、12 秒、発話デューティ 13%）で `shape = frame_rms / ref` を測ると、発話フレームで **p5/p50/p95 = 1.63 / 8.01 / 197.8**。参照は発話レベルより約 18dB 下に居座っている。`shape` は 1 を中心にしていない。

帰結は 2 つある。

- **duck 方向（現行）は実質何もしていなかった。** `strength=1.0 / max_gain=1.0` では発話フレームの **100%** が `max_gain` に張り付き、ゲインは恒等になる。録音済みの RVC 出力を実際に `StreamingEnvelope` へ通す端から端の再生でも、生出力と**ビット一致**した。実機 config の値（`strength=0.1 / min=0.4 / max=0.9 / ema=4000`）ではゲインは発話フレームの 100% で**定数 0.9** = 平坦な −0.9dB のみ（連続発話でも振れ幅 1.75dB）。
- **lift 方向は逆に有害だった。** `shape` がほぼ常に 1 より大きいので負の指数は全体を下げる方に働く。実測で 39% が `min_gain` に張り付き、ゲイン中央値 0.54、ピーク 25104 → 12552（ちょうど半分）。`max_gain` は 1.3 でも 4.0 でも結果が 1 ビットも変わらない（一度も 1 を超えないため）。**リフトではなく平坦な −5dB の減衰**である。

参照が低い理由は「無音で下がる」ことではなく「**そもそも発話レベルまで上がりきらない**」ことだった。減衰側の時定数だけ伸ばす案（up 4000ms / down 120000ms）では p50 は 8.01 → 7.09 にしか動かない。

なお連続発話（デューティ 76%、120 秒）では現行のままでも p50 = 1.15 / 張り付き 3.5% で健全である。問題は**間を置いて話す**使い方に固有で、それは配信そのものの使い方である。

## Decision

参照レベルの更新を **VAD ゲートが開けたブロックに限定**し、`envelope_ema_ms` の既定を **2000ms → 1000ms** にする。

- `StreamingEnvelope.apply` は `update_reference: bool` を**必須引数**で受ける。既定値は置かない: 引数を落とすと黙って「無音にも追従する参照」へ戻るため（[0065](0065-streaming-envelope-seam-handover.md) が `delay_samples` に既定を置かなかったのと同じ規律）。
- **cold start も `update_reference` に従う。** 参照が未確立のまま `update_reference=False` のブロックが来たら、参照を作らずに素通しする（実際に適用したゲイン = 1.0 を継ぎ目へ引き継ぐ。`ref < 1e-8` の素通しと同型）。無音中に起動・resume したときに参照をノイズ床へ固定してしまうのを防ぐ。`reset()` も同じ寿命に従う。
- **判定は runner が持つ。** `runner.reference_may_follow(gains)` が VAD ゲートの窓ゲイン（`StreamingVadGate.window_gains`: 開 = 1.0 / 閉 = `vad_min_gain`）を読み、どれか 1 窓でも全開なら speech とみなす。`gains is None`（`vad_gate=false`）なら常に更新 = 本 ADR 以前の挙動。エンベロープ側は VAD を知らないままにする。
- VAD 判定は既に**エンベロープの 1 つ手前**（`runner.py` の `gate_window_gains`）で計算済みなので、新しい推論も新しい依存も増えない。

実測（2 本の録音、lift `-0.3 / min 0.5 / max 2.0` での張り付き率）:

| 参照の方針 | sparse (デューティ 13%) | continuous (デューティ 76%) |
|---|---|---|
| 現行 EMA 4000ms・常時更新 | 38.6% | 3.5% |
| VAD ゲートのみ（4000ms） | 31.6% | 3.5% |
| EMA 1000ms のみ | 19.3% | 3.0% |
| **VAD ゲート + 1000ms** | **12.3%** | **3.0%** |

両方に効果があり、片方だけでは足りない。時定数は 1000ms が knee で、2000/1500ms は 15.8%、750ms も 12.3%、500ms は 14.0% と再び悪化する（参照がピーク追従に寄る）。連続発話側はどの値でも劣化しない。

## Alternatives rejected

- **減衰側の時定数だけ伸ばす（up 4000ms / down 30000〜120000ms）** — 実測で p50 8.01 → 7.09、張り付き 38.6% → 28.1% にしかならない。参照は無音で下がるのではなく上がりきらないので、下げ方を遅くしても届かない。
- **上昇を速くする非対称 EMA（up 500ms / down 30000ms）** — sparse は p50 1.43 まで改善するが、参照が平均でなくピークの追従になり、連続発話側が p50 0.54 / 張り付き 10.2% へ悪化する（現行 3.5%）。片方のデューティを他方の犠牲で直している。
- **絶対 RMS 閾値で「発話」を判定する** — [0017](0017-rvc-input-envelope-shape-transfer.md) の中核であるマイクゲイン非依存を壊す。マイクやセットアップを変えるたびに再調整が要る。VAD の判定なら相対性を保ったまま同じことができ、しかも既に計算されている。
- **`min_gain` / `max_gain` を広げて張り付きを避ける** — [0065](0065-streaming-envelope-seam-handover.md) が同じ形の案を却下したのと同じで、整形量そのものを削るだけで参照の誤差は残る。
- **エンベロープ自身に VAD を持たせる** — 推論が二重になり、ゲートとエンベロープが別の判定で動く可能性を作る。判定は 1 箇所に置く。

## Consequences

参照が発話レベルに留まるようになり、[0092](0092-streaming-envelope-lift-direction.md) の lift が sparse な発話でも整形として働く（張り付き 38.6% → 12.3%）。duck 方向も初めて実際に整形するようになる（`strength=+1.0` の張り付き 100% → 91.2%、連続発話で 58.4% → 57.5%）。

**これは既存の音を変える。** `envelope_follow=true` の config は、これまで実質恒等（または平坦な減衰）だったものが本当に整形を始める。`envelope_follow=false`（既定）なら `apply` は呼ばれないので出力は不変。`envelope_ema_ms` の既定変更も `envelope_follow=true` の構成にだけ効く。**実機耳確認が要る。**

参照の遅れは減っても消えない: フレーズ頭では参照が発話レベルへ登り切る前なので、最初のフレームは実際より大きく判定される（sparse で 12.3% が残る）。フレーズ頭は [0059](0059-stream-vc-window-resolution-vad-gate.md) の VAD ゲートの担当という [0057](0057-streaming-input-envelope-rolling-ema.md) の切り分けはそのまま有効。

`vad_gate=false` で `envelope_follow=true` を使う構成では `reference_may_follow` が常に True を返すので、本 ADR の改善は得られない（時定数の変更だけが効く）。lift を使うなら VAD ゲートとの併用が前提になる。

Status を `Accepted` としたのは、欠陥と改善の双方が数値で確定し（端から端の再生でのビット一致という形で欠陥が再現し）、退行テスト（非 speech ブロックで参照が動かない / 無音での cold start / runner の受け渡し）が入ったため。本 ADR は参照の**決め方**だけを変える。rolling EMA を参照とする [0057](0057-streaming-input-envelope-rolling-ema.md) の中核判断はそのまま有効で、supersede ではなく refine である。
