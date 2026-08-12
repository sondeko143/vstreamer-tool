# 0082. RVC 経路のサンプルレート変換を自前ポリフェーズ実装に一本化する（torchaudio を廃止、0073 を変換経路へ拡張）

- Status: Accepted
- Date: 2026-08-12
- 効力: 既定
- Related: extends [ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md); [ADR-0080](0080-torch-free-rvc-runtime.md)（この決定の動機）; [ADR-0069](0069-torch-213-and-terminal-torchaudio.md)（廃止する依存のピン）; [ADR-0075](0075-wire-sample-rate-validation.md)（比の上限）; [ADR-0036](0036-whisper-resample-via-pyav.md)（下記のとおり、その記述の片側を無効化する）

> [ADR-0036](0036-whisper-resample-via-pyav.md) は「VC/RVC パスは torchaudio、whisper パスは av」という当時の分担を記述している。本 ADR はその **VC/RVC 側だけ**を置き換える。0036 の決定そのもの（whisper のリサンプルを PyAV で行う）は有効なので supersede しない。

## Context

変換経路のリサンプルは torchaudio の `T.Resample`（既定の `sinc_interp_hann`、`lowpass_filter_width=6`、`rolloff=0.99`）を `lru_cache` して使っている。呼ばれるのは 2 箇所だけで、発話単位 VC の入力（`voice_sample_rate` → 16kHz）と、VC ワーカーの VAD 前処理である。

[ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md) はデバイス境界のレート変換を自前のポリフェーズ FIR（Kaiser 窓、阻止域 80dB、numpy のみ）へ移すことを決めた。ストリーミング経路は capture 側でその実装が既に 16kHz へ落としているため、torchaudio を通らない。つまり torchaudio は発話経路にだけ残った重複実装である。

torchaudio は torch 本体に依存するので、**torchaudio を残すことは torch を残すことと同義**であり、[ADR-0080](0080-torch-free-rvc-runtime.md) の目的と両立しない。

数値回帰の観点では、既存の numeric golden の入力が 48kHz であることが効く。リサンプラは実際に通る経路にあり、判定は相関 ≥0.999 / 波形 SNR ≥35dB である。フィルタを差し替えれば出力は数値的に変わる。

## Decision

`get_resampler`（torchaudio）を廃し、`PolyphaseResampler` の一括変換エントリに一本化する。VAD 前処理も同じ実装を通す。

golden は実機で回して実測する。判定を通れば据え置き、落ちた場合は**実測値を記録した上で** golden を再取得し、実機での聴感確認を経る。「落ちたから閾値を緩める」ことはしない。

## Alternatives rejected

- **torchaudio のカーネル（`sinc_interp_hann`、width 6、rolloff 0.99）を numpy へ移植する** — golden の数値をほぼ保てる見込みはあるが、repo にリサンプラ実装が 2 つ並ぶ。[ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md) が「境界のレート変換は 1 つの実装に寄せる」と決めた直後にそれを崩すことになる。しかも移植先はフィルタとして劣る（width 6 の Hann 窓 sinc の阻止域は Kaiser 80dB より浅い）。数値の据え置きのために品質と単一実装の両方を捨てる取引になる。
- **`change_voice` を 16kHz 入力限定にしてリサンプル自体を廃止する** — コードは最も単純になるが、既存の設定ファイルの互換を壊す。移行の手当てが要るうえ、その手当ては結局どこかでレート変換を行うことになる。
- **torchaudio だけ残す** — 依存の実体は torch 本体（476.7MB）なので、削減がまるごと消える。
- **現状維持（torchaudio を使い続ける）** — [ADR-0080](0080-torch-free-rvc-runtime.md) が成立しなくなる。

## Consequences

発話単位 VC の出力が数値的に変わる。阻止域は改善する方向（浅い Hann 窓 sinc → Kaiser 80dB）だが、改善であっても変化は変化なので、golden の再取得と聴感確認の対象になる。

VAD 前処理も同じ実装を通るため、VC ワーカーが torch に触る箇所が無くなる。repo のリサンプラ実装が 1 つになる。

`PolyphaseResampler` は病的な比を `ValueError` で弾く（[ADR-0075](0075-wire-sample-rate-validation.md)）。torchaudio も内部で `gcd` を取るため同種の膨張自体は元からあったが、今後は静かに巨大なフィルタを構築する代わりに明示的に失敗する。発話 VC の入力レートが 16000 と互いに素に近い値のとき、従来と挙動が変わりうる。

群遅延の扱いが変わる。一括変換のエントリは tail をフラッシュして群遅延を除去し、`round(n * dst / src)` サンプルを返す。torchaudio の出力長と一致しない可能性があり、golden の shape 比較（`out.shape == golden.shape`）はそこも見ることになる。
