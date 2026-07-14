# 0020. Silero VAD モデルを v6.2.1 へ更新する

- Status: Accepted
- Date: 2026-07-08
- Related: supersedes the v5.1.2 pin in [ADR-0019](0019-vc-silero-vad-gate.md); spec [2026-07-08-vc-vad-v6-migration-design.md](../superpowers/specs/2026-07-08-vc-vad-v6-migration-design.md)

## Context

VAD ノイズゲート（[ADR-0019](0019-vc-silero-vad-gate.md)）は Silero VAD v5.1.2 で実装・実機検証済みで稼働している。snakers4/silero-vad は v6 系（最新 v6.2.1, 2026-02-24）に更新されており、非音声判定の向上を期待して後継モデルへ移行したい。初期実装時の「repo master は v6、v5 の `state` 入力契約を壊すから使うな」という注記は未検証の防御的仮定であり、実アーティファクトを onnxruntime で直接検査した結果、誤りと判明した。ONNX I/O 契約は v5.1.2 と v6.2.1 で完全一致（`input`/`state [2,None,128]`/`sr` int64、出力 `output`/`stateN`）で、v5 スタイルの feed が v6 でそのまま動作する。非音声判定は実測で向上した（silence 確率 0.0120→0.0017、white-noise 0.0898→0.0105）。

## Decision

VAD モデル pin を **Silero VAD v6.2.1**（sha256 `1a153a22…`, 2327524 bytes、リリースタグのモデルが repo master と bit 一致）に更新する。推論ラッパー・ゲートアルゴリズム・既定閾値（`vad_threshold=0.5` 等）は無変更。`"state" in input_names` の検証は v5/v6 を通し v4 を弾いたまま維持する。歴史的記録である v5 spec/plan は改変しない。

## Alternatives rejected

- **v5.1.2 を維持** — ノイズ判別が v6 より劣る（実測値の通り）。
- **定量 A/B ハーネスを構築** — 過剰。検証は実機の耳確認＋env-gated real-model テストで足りる。
- **v6 移行に伴う閾値再チューニング** — v6 は判定がより決定的なため過剰 fit を避け v5 値を維持する。

## Consequences

ロールバックは退避した v5 ファイル（`silero_vad_v5.onnx`）を元パスに戻すだけ（コード変更不要）。プロジェクトメモリの「v6 do NOT use」注記を訂正する必要がある。
