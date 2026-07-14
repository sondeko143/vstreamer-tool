# 0031. audio extra を PyAudio から sounddevice へ移行する

- Status: Accepted
- Date: 2026-07-12
- Related: spec [2026-07-12-python-314-migration-roadmap-design.md](../superpowers/specs/2026-07-12-python-314-migration-roadmap-design.md); [ADR-0025](0025-target-python-314-phased.md)

## Context

[ADR-0025](0025-target-python-314-phased.md) のフェーズ③実装中に、pyworld（[ADR-0030](0030-pyworld-lazy-default-rmvpe.md)）に加えて第 2 の cp314 wheel ギャップが判明した——PyAudio に cp314 wheel が無い。roadmap spec は「唯一のギャップは pyworld」と断定していたが誤りであり、この決定は roadmap spec 本文には無く、フェーズ③実装で確定した。

## Decision

audio extra を PyAudio から sounddevice（py3-none, PortAudio 同梱）へ移行し、録音・再生・音声 I/O のコードを移植する。

## Alternatives rejected

- **PyAudio を維持する** — cp314 wheel が無く、3.14 で audio extra が解決しない。

## Consequences

3.14 で audio extra が解決可能になる。教訓: この種の実行時ギャップ（例: `get_event_loop()` が 3.14 で raise する不具合）はテストでは捕まらず、エントリポイントの実行で初めて露見した。
