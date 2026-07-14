# 0002. Pydantic を v2 ネイティブへ全面移行する

- Status: Accepted
- Date: 2026-06-14
- Related: [ADR-0001](0001-upgrade-voicevox-core-0-16.md) がブロッカー除去 / [ADR-0003](0003-secretstr-json-via-field-serializer.md) / spec [2026-06-14-pydantic-v2-migration-design](../superpowers/specs/2026-06-14-pydantic-v2-migration-design.md)

## Context

`vspeech` / `gui` は Pydantic v1 専用 API（`BaseSettings`・`root_validator`・`parse_obj`・`.dict()`・`orm_mode`・`Field(env=)` 等）に依存している。[ADR-0001](0001-upgrade-voicevox-core-0-16.md) で唯一のブロッカーが外れ、`uv.lock` 上で pydantic を要求するのは本プロジェクトのみになったため v2 へ解決できる。既存の `WorkerInput` バリデータは `@classmethod` を `@root_validator` の上に重ねた誤りで発火しておらず、2件のテストが main でも失敗している。

## Decision

互換シム `pydantic.v1` を一切使わず、全モデルを v2 ネイティブ API へ移行する（`BaseSettings` は pydantic-settings、`model_validator`/`model_config`/`model_validate`/`model_dump` 等）。壊れていた `WorkerInput` バリデータを `@model_validator(mode="after")` で正しく発火させ、不正 sound 入力・file_path 無し reload を実行時に拒否する。`model_` 接頭辞のフィールド（`RvcConfig.model_file`・`VoicevoxConfig.model_dir`）は改名せず `protected_namespaces=()` で警告を抑止し、`filters` は `list[str]` に変更する。CLAUDE.md はこの時点からリポジトリ追跡下に入れ、v2 前提へ更新する。

## Alternatives rejected

- **`pydantic.v1` 互換シムで段階移行** — 中途半端な二重状態と技術的負債を残す。
- **壊れたバリデータを温存** — 実害あるバグの放置で、2件の失敗テストも残る。
- **`model_` 接頭辞フィールドを改名** — 設定互換を壊す。

## Consequences

依存 bump 後、全変換が完了するまでブランチ上でテストスイートが赤になる。不正入力を実行時に拒否する挙動変更を伴い、対応する2件のテストは pass へ転じる。
