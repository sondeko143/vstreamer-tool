# 0026. numpy 2 を採用する（>=2,<3）

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

numpy 1.26.4 には cp313 / cp314 wheel が無く、`Requires-Python` が上限なし（`>=3.9`）のためリゾルバは候補から外さず sdist ビルドを試みて失敗する。そして cp314 wheel を配る依存は numpy 1.26 でビルドできない以上、必然的に numpy 2 ビルドである。[ADR-0025](0025-target-python-314-phased.md) のフェーズ①として、危険で実機依存な変数を最初に切り出す。

## Decision

whisper / rvc extra の numpy を `>=1.23,<2` から `>=2,<3` へ緩和する（フェーズ①、`requires-python` は 3.12 のまま）。アプリケーションコードは NEP 50 安全で無変更であり、検証が本体となる。faiss / scipy / ctranslate2 は lock 再生成で numpy 2 対応版へ随伴する。

## Alternatives rejected

- **numpy `<2` に留める** — 3.14 到達を恒久的にブロックする。

## Consequences

cp314 wheel が存在する版へ解決できるようになる。numpy 1→2 を CUDA 変更から独立した単独変数として切り分けられる。
