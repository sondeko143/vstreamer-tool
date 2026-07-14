# 0003. SecretStr の平文 JSON シリアライズに field_serializer を使う

- Status: Accepted
- Date: 2026-06-14
- Related: [ADR-0002](0002-migrate-to-pydantic-v2-native.md) / spec [2026-06-14-pydantic-v2-migration-design](../superpowers/specs/2026-06-14-pydantic-v2-migration-design.md)

## Context

GUI は `config.model_dump_json()` で秘密値を平文 JSON へ書き出し、本体プロセスが `read_config_from_file`（`model_validate`）で読み戻す経路を持つ。この round-trip を維持しつつ [ADR-0002](0002-migrate-to-pydantic-v2-native.md) の v2 移行を成立させる必要がある。pydantic v2 では `json_encoders` は `model_dump_json()` の `SecretStr` に作用しない。

## Decision

`SecretStr` を持つ設定を `@field_serializer(..., when_used="json")` で平文化し、GUI→本体の平文 JSON ハンドオフを成立させる。v2 で機能しない `json_encoders` は使わない。

## Alternatives rejected

- **`json_encoders={SecretStr: ...}` を維持** — 移行の初期案として検討したが、v2 の `model_dump_json()` では `SecretStr` に作用せず round-trip が壊れることを実測で確認したため却下。

## Consequences

秘密値の JSON round-trip が保証され、非推奨 API への依存を避けられる。
