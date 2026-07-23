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

### 実測(SOLA と等電力の裏づけ)

実モデルの診断で、同じ入力区間の二つの描画は「同じ波形が時間シフトしているだけ」だと確認できた: 正規化相互相関は **lag 0 で -0.017、最適 lag では 0.891**(33/33 tick、最適 lag はいずれも ±10ms 以内、平均 |lag| 3ms)。したがって固定位置でのブレンドは同一波形を数 ms ずらして足す櫛形フィルタ(comb)であり、聴感上の「扇風機」ノイズの正体だった。位置合わせしてから混ぜるのが正しい。

フェード則は **SOLA 導入後に再測定して等電力を維持**した。当初は SOLA 無しの無相関状態(lag 0 相関 -0.02)で測り「等電力=RMS 比 1.0007 / sum-to-1=0.873(-1.16dB ディップ)」だったが、**SOLA は意図的に信号を相関させる**(整列点相関 ρ=0.82)ので、この前提は void になった。相関した信号では等電力は seam を過剰加算しうる。そこで SOLA on で 3 則を同一 seed(=デコーダ出力・SOLA 整列がビット同一、重みだけ差)で再測定した:

| フェード則 | blend/rest RMS 比 | 6.25Hz 変調指数 |
|---|---|---|
| 等電力(現行) | +1.14 dB | 0.0671 |
| sum-to-1 (sin²/cos²) | -0.76 dB | 0.0656 |
| linear | -0.90 dB | 0.0652 |

静的な blend/rest 比では sum-to-1 が僅かに優(等電力は予測どおり +1.14dB の seam bump)。だが**実際に聴こえる周期変調(6.25Hz)は 3 則でほぼ同一**(0.067 vs 0.066、差 ~0.01dB)、seam 連続性も 3 則とも mid-block より良好。3 則の WAV を実機で A/B したが**聴き分け不能**だった(2026-07-24)。→ 可聴差が無いので実装は変えず**等電力を維持**。correctness としては sum-to-1 が僅かに優だが、SOLA の整列は短窓で crossfade 帯全体では相関がやや薄れるため +1.14dB の静的 bump が周期変調としては現れない、というのが解釈。将来 seam の周期性が問題化したら sum-to-1 への切替が第一候補。

代償は `sola_search_ms` 分のアルゴリズム遅延(読み出し位置を探索半幅だけ手前へずらす)と、HuBERT の受容野による末尾切り詰めぶん(約 320 入力サンプル = 20ms)。どちらも emit 長には影響しない(長さは実時間クロック由来の固定値)ので、ドリフトは発生しない。`sola_search_ms = 0` は SOLA 無効 = 導入前と完全に同一のサンプルを出す。

### 検証(実機耳確認)

実機(RTX 4060 Laptop / f0 抽出器 fcpe / 実声)での耳確認により、`block_ms = 160` / `context_ms = 500` / `crossfade_ms = 25` + SOLA(`sola_search_ms = 5.0`)の構成が clean であることを確認した。context を 500ms 未満にすると seam のガタつきが常時聞こえ(100ms は「ガタゴト」が乗って使い物にならない)、逆に 500ms を超えて 2000ms にしても改善しなかった。block を 80ms に縮めると片道遅延は ~199ms → ~119ms に下がるが、seam のプチプチ(クリック)が可聴になる。SOLA は上で測定した位相ずれ(lag 0 の相関 -0.02、最適 lag で 0.89)を解消するためのもので、耳確認した構成にはこれが含まれている — ただし SOLA 単独の可聴寄与は A/B で分離測定していないため、「SOLA on の構成で clean だった」という以上のことは主張しない。

### VAD ノイズゲート(実装済み)

ゲート無しの streaming 経路は無音でも回り続け、**部屋のノイズフロアをそのまま変換したうえで増幅して**鳴らしていた(実測: 入力 RMS 0.000298 に対し出力 0.0204 = **+10〜+37dB**、context 100ms 時)。[0019](0019-vc-silero-vad-gate.md) の VAD ゲートは発話系 `[vc]` の chunk 単位ゲートで streaming には効かないため、streaming 専用のゲートを入れた(`vspeech/stream_vc/gate.py`、設定は `[stream_vc]` の `vad_gate` / `vad_model_file` / `vad_threshold` / `vad_hangover_ms` / `vad_min_gain`)。

- **判定は入力ブロック、適用は出力ブロック**。Silero VAD 本体は `vspeech/lib/vad.py`(発話系と同じ `silero_vad.onnx`)を読み取り専用で再利用する。streaming のキャプチャは 16kHz なので `VAD_SAMPLE_RATE` と一致し、リサンプルは要らない。判定は `input_boost` を**かける前**の素のブロックで行う(見かけのレベルではなく実際のマイクレベルで判定する)。
- **ゲートが閉じていても推論はスキップしない**。`StreamingVc` は rolling 左文脈とクロスフェード tail を持つステートフル変換なので、ブロックを飛ばすと文脈に穴が開き、発話が再開したときの seam が壊れる。減衰するのは emit する音だけ(GPU 余力は実測 RTF 0.24 で足りる)。
- **hangover 付きのステートフル判定**。ブロック粒度(160ms なら 6.25Hz)の素の判定をそのまま使うと語間の短い無音でゲートがバタつく。ブロック内は窓確率の **max** で判定して発話頭(onset)を落とさず、閉じる側を hangover が受け持つ。
- **ゲインはブロック内で線形に ramp する**。ブロック境界でゲインを階段状に変えること自体がクリックを生む(この ADR がずっと消してきたアーティファクトと同種)ので、前ブロック終端のゲインから今ブロックの目標ゲインへ繋ぐ。`1.0 -> 1.0` は恒等の高速路なので、**既定 off / 常時 speech のときの出力はゲート導入前とビット単位で同一**。
- **既定 off、fail-open**。実行時に VAD が失敗したら音は素通しし、警告は 1 回だけ出す(6.25Hz でログを埋めない)。`vad_gate=true` のときのモデル実在は preflight(層A)で fail-loud に検査する。

### 未実装(この ADR の決定に含まれないもの)

envelope 整合(rolling/EMA 基準への置換)は、当初この Decision に書いていたが**実装していない**ので決定から外した。`StreamVcConfig` に envelope のフィールドは無く、`StreamingVc` / `stream_vc/runner.py` のどちらにも入力エンベロープ転写は無い(発話系 `[vc]` の [0017](0017-rvc-input-envelope-shape-transfer.md) は発話全体を基準にするので streaming にはそのまま持ち込めない)。導入するならブロック粒度で基準が飛ぶことによる pumping をどう抑えるかを含めて別の ADR で決めること。

## Alternatives rejected

- **固定位置(lag 0)のまま等電力クロスフェードする** — 上の実測どおり lag 0 の相関は -0.02(ほぼ無相関〜逆相関)で、混ぜると comb になる。クロスフェード長を伸ばしても位相不整合は消えないので解決しない。
- **クロスフェード則を sum-to-1(sin²/cos²)に変える** — SOLA 導入後に再測定したところ、静的 blend/rest 比では等電力より僅かに良い(-0.76dB vs +1.14dB)が、可聴に効く 6.25Hz 変調指数は等電力とほぼ同一(~0.01dB 差)で、実機 A/B でも聴き分け不能だった(上「実測」節参照)。可聴差が無いので現行の等電力を維持し、実装は変えない。位相ずれの本質的な対策は SOLA であってフェード則ではない。
- **発話全体 reflect-pad の `change_voice` をブロックにそのまま呼ぶ(ステートレス流用)** — 短ブロックの reflect-pad 文脈がゴミで境界破綻、クロスフェード無しでクリック、f0 端不安定、可変長で毎回グラフ再構築。streaming では成立しない。
- **発話単位の envelope/VAD をそのままブロックへ適用** — 発話全体平均 RMS 正規化や、hangover もゲイン ramp も無い素のブロック粒度 VAD は、基準/ゲートがブロックごとに飛んで pumping/choppy になる(境界でのゲイン階段はクリックそのもの)。上のゲートが hangover とブロック内 ramp を持つのはこれを避けるため。

## Consequences

クリック無し・ピッチ連続の連続変換が、固定 shape で安定して回る(warmup 1 回)。既存 `change_voice` を壊さず内部部品を再利用するので差分を局所化できる。反面、context 分の余剰推論で per-block RTF が増える(RTF 実測の主対象)。クロスフェード/f0 連続性の正しさは seeded golden で担保が要る([0016](0016-change-voice-decompose-seeded-golden.md) の決定的シード基盤を流用)。
