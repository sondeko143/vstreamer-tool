# Pydantic v1 → v2 移行 設計書

## 問題

`vspeech` / `gui` は Pydantic v1 専用 API（`BaseSettings`・`root_validator`・`parse_obj`・`.dict()`・`orm_mode`・`Field(env=)`・`json_encoders` 等）に依存している。さらに `WorkerInput` のバリデータはデコレータの重ね方の誤りで発火しておらず、不正入力を検証できず2件のテストが失敗し続けている。

## ゴール

全モデルを Pydantic v2 ネイティブ API へ全面移行する。壊れていた `WorkerInput` バリデータを正しく発火させ、不正 sound 入力・file_path 無し reload を実行時に拒否する。GUI→本体の平文 JSON による秘密値ハンドオフの round-trip を維持する。

## 非ゴール

- 互換シム `pydantic.v1` を用いた段階移行。
- `model_` 接頭辞フィールド（`RvcConfig.model_file` / `VoicevoxConfig.model_dir`）の改名。
- 設定キーやスキーマの変更。

## 受入基準

- pydantic が v2（`>=2,<3`）で解決し、`pydantic-settings` が導入されている。
- コード中に互換シム `pydantic.v1` への import が存在しない。
- 不正な sound 入力、および file_path の無い reload が実行時に拒否される。
- 従来 main でも失敗していた2件の `WorkerInput` バリデータ関連テストが pass する。
- 秘密値が平文 JSON で書き出され、読み戻せる（GUI→本体の round-trip が成立する）。
- 既存の設定キーは不変で、従来の config ファイルがそのまま読める。
- テストスイート全体がグリーンである。

---

- 決定根拠: [ADR-0002](../../adr/0002-migrate-to-pydantic-v2-native.md) , [ADR-0003](../../adr/0003-secretstr-json-via-field-serializer.md)
- 実装計画: [plan](../plans/2026-06-14-pydantic-v2-migration.md)
