# 0019. VC ノイズ対策に Silero VAD ゲートを採用し VC パスに限定する

- Status: Accepted
- Date: 2026-07-08
- Related: spec [2026-07-08-vc-vad-noise-gate-design.md](../superpowers/specs/2026-07-08-vc-vad-noise-gate-design.md); [ADR-0017](0017-rvc-input-envelope-shape-transfer.md)（マイクゲイン不変という同性質）, [ADR-0024](0024-onnx-session-single-factory.md)（onnx セッションの CPU 固定例外）; モデル pin は [ADR-0020](0020-silero-vad-v6.md) で更新

## Context

RVC は非音声入力（環境音・ブレス）からもフルスケールの人声風ノイズを生成する。HuBERT がノイズを音素的特徴として解釈し、デコーダが常にフルスケールの声を出すためである。漏れ道は 2 つ。(1) ノイズ単独チャンク — recording の門番は発話単位の dBFS 閾値（`silence_threshold`、既定 -40）なので環境音・ブレスで録音がトリガーされるとチャンク全体がノイズのまま vc に届き、[ADR-0017](0017-rvc-input-envelope-shape-transfer.md) の相対エンベロープは平均正規化のためチャンク全体がノイズだと gain≈1 に正規化され素通しする。(2) 発話内のブレス・環境音 — speaking 状態では全フレームが素通しで、録音前のプリロールも先頭に付く。両方の症状がユーザーで確認済み。同じ録音チャンクは `routes_list` で transcription と vc に並列に流れ、transcription 側は「フィラー・感嘆詞を残す」方針のため recording 側のゲート強化は採れない。

## Decision

Silero VAD（ONNX 約 2MB, MIT, CPU で 1 チャンク数 ms）による **絶対判定ゲート** を opt-in で導入し、対策を **VC パスのみ** に限定する（recording の `silence_threshold` は変更しない）。ノイズ単独チャンクは RVC 推論の前に speech 比率で判定してスキップし GPU の約 0.55s を使わない。発話内の非音声区間は 32ms 窓解像度のゲインマスクを出力サンプル軸へ線形補間して乗算する。RVC への入力はゼロ埋めせず無加工とし（ゼロ埋めは HuBERT 特徴に不連続を生みチャンク境界アーティファクトになるため、可聴性は出力側ダックで解決）、VAD セッションは CPUExecutionProvider 固定とする（[ADR-0024](0024-onnx-session-single-factory.md) の意図的例外、RVC と CUDA を取り合わない）。モデル不在・ロード失敗は起動時に fail loudly、推論中の例外はそのチャンクをゲートなしで通す（配信中に声が消える方向の故障を避ける非対称エラー処理）。初期モデルは v5.1.2 を手動取得・pin する（後に [ADR-0020](0020-silero-vad-v6.md) で v6.2.1 へ更新）。

## Alternatives rejected

- **案B: 純DSPエネルギーゲート** — 閾値がマイクゲイン依存で、[ADR-0017](0017-rvc-input-envelope-shape-transfer.md) が意図的に排除した性質の再導入。大きいブレスと静かな発話を区別できない。
- **案C: ノイズ抑制前処理（DeepFilterNet 等）** — 新規依存が重く HuBERT に入る音色が変わり変換品質に影響し得る。症状（発話と重なる定常ノイズ）ではないため過剰。
- **非音声区間を入力段でゼロ埋め** — チャンク境界アーティファクトを生む。
- **recording 側 `silence_threshold` の強化** — 並列に流れる transcription のフィラー・感嘆詞を落とす。
- **起動時も実行時も両方 fail／両方 silent pass** — 前者は配信中に声が消え、後者はノイズ復活に気づけない。

## Consequences

opt-in の絶対判定ゲートでチャンク全体ノイズの穴を塞ぐ。モデルのローカル取得・pin 運用が新たに必要になる。transcription への副作用はゼロ。
