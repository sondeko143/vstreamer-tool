# 0001. voicevox-core を 0.16.4 へ上げ pydantic v2 移行と分離する

- Status: Accepted
- Date: 2026-06-14
- Related: [ADR-0002](0002-migrate-to-pydantic-v2-native.md) のブロッカー除去 / spec [2026-06-14-voicevox-core-0.16-upgrade-design](../superpowers/specs/2026-06-14-voicevox-core-0.16-upgrade-design.md)

## Context

pydantic v2 化の唯一のハードブロッカーは `voicevox-core==0.14.3`（`pydantic>=1.9.2,<2` を要求）である。0.16.4 は Python API から pydantic 依存を完全に削除しており、上げればブロッカーが外れて後続の v2 化が可能になる。ただし voicevox 上げと v2 化を同一ブランチで一度に行うと検証が難しく、段階分割したい。

## Decision

voicevox-core を 0.14.3 から 0.16.4（pydantic 非依存）へ上げ、pydantic v2 化とは別ブランチに分離する。0.16 の分離 API（`Onnxruntime`/`OpenJtalk`/`Synthesizer`/`VoiceModelFile`）を使い、実行時資産（`voicevox_onnxruntime` DLL・OpenJTalk 辞書・`.vvm`）は wheel 非同梱として設定パス指定とする。`Onnxruntime.load_once(filename=...)` で `onnxruntime_path` を明示し、onnxruntime-gpu の DLL 誤ロードを防ぐ。config キー `speaker_id` は据え置き（内部は style_id 扱い）、`acceleration_mode` は AUTO 固定とする。

## Alternatives rejected

- **voicevox 上げと pydantic v2 化を同一ブランチで同時実施** — 検証が難しく、段階分割のほうが安全。
- **`onnxruntime_path` を `load_once` の既定解決に任せる** — PATH 上の onnxruntime-gpu の DLL を誤ってロードしうる。
- **`speaker_id` を `style_id` へ改名 / `acceleration_mode` を設定化** — 設定互換を壊す・YAGNI。

## Consequences

pydantic v2 化のブロッカーが除去され、後続ブランチで独立に進められる。一方、実行時資産は wheel に含まれないため公式ダウンローダによる資産取得・配置の運用が必要になる。
