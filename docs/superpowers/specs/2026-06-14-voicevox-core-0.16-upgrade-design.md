# voicevox-core 0.14.3 → 0.16.4 アップグレード 設計書

## 問題

pydantic v2 化の唯一のハードブロッカーが `voicevox-core==0.14.3`（`pydantic>=1.9.2,<2` を要求）である。この依存が残る限り pydantic を v2 へ上げられない。

## ゴール

voicevox-core を 0.16.4（pydantic 非依存）へ上げてブロッカーを除去し、後続の pydantic v2 化を可能にする。実行時資産（onnxruntime DLL・OpenJTalk 辞書・`.vvm` モデル）は設定パス指定で読み込む形にする。

## 非ゴール

- pydantic 本体の v2 化そのもの（別ブランチで実施し、本ブランチでは v1 のまま据え置く）。
- config キー `speaker_id` の改名や `acceleration_mode` の設定化。

## 受入基準

- voicevox-core が 0.16.4 で解決し、依存グラフから `pydantic<2` のピンが消えている。
- pydantic は引き続き v1 系で解決され、新たな依存競合が発生していない。
- VOICEVOX TTS が、設定パスで指定した onnxruntime DLL・OpenJTalk 辞書・`.vvm` を読み、実合成の WAV（24000Hz / mono / INT16）を出力できる。
- onnxruntime-gpu と `voicevox_onnxruntime` が同一環境で共存し、DLL の誤ロードが起きない。
- 既存の `speaker_id` 設定キーがそのまま機能し、設定互換が保たれている。
- 実資産を要しない通常のテスト実行がグリーンである。

---

- 決定根拠: [ADR-0001](../../adr/0001-upgrade-voicevox-core-0-16.md)
- 実装計画: [plan](../plans/2026-06-14-voicevox-core-0.16-upgrade.md)
