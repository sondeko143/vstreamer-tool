# 0053. ストリーミング VC を固定ブロック+左文脈+クロスフェードのステートフル変換にする

- Status: Accepted
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0016](0016-change-voice-decompose-seeded-golden.md), [0017](0017-rvc-input-envelope-shape-transfer.md), [0019](0019-vc-silero-vad-gate.md), [0050](0050-streaming-vc-separate-subsystem.md)

## Context

既存の `change_voice` は発話全体を reflect-pad して全区間を一括推論するステートレス変換。これを短いブロックにそのまま適用すると、reflect-pad の文脈がゴミで境界が壊れ、独立推論した出力の単純連結でクリックが出る。ONNX デコーダはステートレスで隠れ状態を持ち越せない。f0(RMVPE/FCPE)や HuBERT は左右文脈がないと端が不安定になる。onnxruntime のグラフ構築は shape 依存で、可変長入力だと毎回オートチューンのコストがかかる(発話単位では 145s の初回 stall 既往)。

## Decision

ストリーミング VC を**固定長ブロック単位のステートフル変換**にする。

- rolling context buffer で直近の実音声を左文脈として保持し、毎 tick `[context | 新ブロック]` を推論して、出力は新ブロック相当だけ採用する(reflect-pad ではなく実音声の左文脈を与える)。
- 隣接出力の overlap 区間を等電力クロスフェードで overlap-add してクリックを消す(crossfade tail をブロック間状態として保持)。
- **混ぜる前に SOLA(Synchronous OverLap-Add)で位相を合わせる**。前 tick の emit 末尾(`_output_tail`)と、今 tick の描画のうち同じ入力時刻にあたる区間を、小さな探索窓(`sola_search_ms`、既定 ±5ms)内で正規化相互相関にかけ、最も相関する読み出し位置から等電力クロスフェードする。lag は「どこから読むか」だけを変え、emit 長は常に hop 相当ちょうど — すなわち**サンプルレートのクロックから導く実時間長 `block_len * target_sample_rate / 16000`** であって、デコーダの描画長からの比率ではない — なので、レートロックは保たれる(ドリフト無し・sink の飢餓無し)。読み出し**位置**だけは描画長からの逆算(末尾アンカー)にして、HuBERT の受容野が末尾を一定量切り詰めるぶんを避ける。
- 入力 shape を固定し、warmup を 1 回で済ませる(以後 re-autotune なし)。
- 既存 `change_voice` の内部部品(HuBERT 特徴量抽出 / f0 抽出 / infer / int16 化)は再利用するが、発話系の `change_voice` 経路自体は無改変で温存する。
- envelope 整合は発話全体平均に依存するため、streaming では rolling(EMA)基準へ置換するか既定 off とする。VAD はブロック粒度のバタつきを避けるため hangover 付きステートフルゲートとする(既定 off、fail-open)。

### 実測(SOLA と等電力の裏づけ)

実モデルの診断で、同じ入力区間の二つの描画は「同じ波形が時間シフトしているだけ」だと確認できた: 正規化相互相関は **lag 0 で -0.017、最適 lag では 0.891**(33/33 tick、最適 lag はいずれも ±10ms 以内、平均 |lag| 3ms)。したがって固定位置でのブレンドは同一波形を数 ms ずらして足す櫛形フィルタ(comb)であり、聴感上の「扇風機」ノイズの正体だった。位置合わせしてから混ぜるのが正しい。

フェード則は別途測定して**等電力が正しい**と確認済み(blend/rest の RMS 比 **1.0007**)。sum-to-1 の線形重みは **0.873 = -1.16dB のディップ**になるため却下した。よって SOLA は探索位置だけを変え、`equal_power_weights` は変更しない。

代償は `sola_search_ms` 分のアルゴリズム遅延(読み出し位置を探索半幅だけ手前へずらす)と、HuBERT の受容野による末尾切り詰めぶん(約 320 入力サンプル = 20ms)。どちらも emit 長には影響しない(長さは実時間クロック由来の固定値)ので、ドリフトは発生しない。`sola_search_ms = 0` は SOLA 無効 = 導入前と完全に同一のサンプルを出す。

### 検証(実機耳確認)

実機(RTX 4060 Laptop / f0 抽出器 fcpe / 実声)での耳確認により、`block_ms = 160` / `context_ms = 500` / `crossfade_ms = 25` + SOLA(`sola_search_ms = 5.0`)の構成が clean であることを確認した。context を 500ms 未満にすると seam のガタつきが常時聞こえ(100ms は「ガタゴト」が乗って使い物にならない)、逆に 500ms を超えて 2000ms にしても改善しなかった。block を 80ms に縮めると片道遅延は ~199ms → ~119ms に下がるが、seam のプチプチ(クリック)が可聴になる。SOLA は上で測定した位相ずれ(lag 0 の相関 -0.02、最適 lag で 0.89)を解消するためのもので、耳確認した構成にはこれが含まれている — ただし SOLA 単独の可聴寄与は A/B で分離測定していないため、「SOLA on の構成で clean だった」という以上のことは主張しない。

## Alternatives rejected

- **固定位置(lag 0)のまま等電力クロスフェードする** — 上の実測どおり lag 0 の相関は -0.02(ほぼ無相関〜逆相関)で、混ぜると comb になる。クロスフェード長を伸ばしても位相不整合は消えないので解決しない。
- **クロスフェード則を sum-to-1(線形)に変える** — 実測 RMS 比 0.873(-1.16dB のディップ)。等電力(1.0007)より悪く、位相ずれの本質的な対策にもならない。
- **発話全体 reflect-pad の `change_voice` をブロックにそのまま呼ぶ(ステートレス流用)** — 短ブロックの reflect-pad 文脈がゴミで境界破綻、クロスフェード無しでクリック、f0 端不安定、可変長で毎回グラフ再構築。streaming では成立しない。
- **発話単位の envelope/VAD をそのままブロックへ適用** — 発話全体平均 RMS 正規化やブロック粒度 VAD は、基準/ゲートがブロックごとに飛んで pumping/choppy になる。

## Consequences

クリック無し・ピッチ連続の連続変換が、固定 shape で安定して回る(warmup 1 回)。既存 `change_voice` を壊さず内部部品を再利用するので差分を局所化できる。反面、context 分の余剰推論で per-block RTF が増える(RTF 実測の主対象)。クロスフェード/f0 連続性の正しさは seeded golden で担保が要る([0016](0016-change-voice-decompose-seeded-golden.md) の決定的シード基盤を流用)。
