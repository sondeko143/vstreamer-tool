# 0054. ストリーミング VC 設定を [stream_vc] として発話系 VC と分離する

- Status: Accepted
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0050](0050-streaming-vc-separate-subsystem.md), [0038](0038-worker-config-preflight-fail-loud.md), [0045](0045-gui-readiness-reuses-preflight.md), [0046](0046-gui-shared-asset-paths-explicit-propagate.md)

## Context

ストリーミング VC は発話系 VC(`[vc]`/`[rvc]`)と独立に有効化・調整したい。固有項目(block/context/crossfade 長、rolling envelope、streaming VAD、トランスポート種別/宛先、jitter など)を既存 `[vc]`/`[rvc]` に相乗りさせると、発話系 VC の設定と混在し、どの設定がどちらに効くか不明瞭になる。必須リソース検査は preflight に集約する規約がある([0038](0038-worker-config-preflight-fail-loud.md) / [0045](0045-gui-readiness-reuses-preflight.md))。

## Decision

ストリーミング VC の設定を独立した `[stream_vc]`(および関連する入力/トランスポート/再生セクション)として持ち、発話系 `[vc]`/`[rvc]` とは分離する。既定は disable。必須リソース検査は `preflight` に追記し、GUI の起動前 readiness は自動追従させる([0045](0045-gui-readiness-reuses-preflight.md))。

## Alternatives rejected

- **既存 `[vc]`/`[rvc]` に streaming 用フィールドを追加する** — 発話系 VC と streaming VC の設定が同一セクションに混在し、有効範囲が曖昧化する。独立 enable/独立調整が spec 要件なので、セクション分離が素直。

## Consequences

発話系と streaming を独立に on/off・調整でき、config を見れば効果範囲が一目で分かる。反面、両系統が共有する素材パス(HuBERT/RMVPE 等)は各系統へ明示 propagate が要る([0046](0046-gui-shared-asset-paths-explicit-propagate.md) の方針を streaming にも適用)。preflight/GUI の必須項目は `[stream_vc]` 有効時のみ課す形にする。

独立セクションにした帰結として、明示 propagate が要るのは**素材パスだけではない**。声質パラメータ、とりわけ `f0_up_key` も発話系 `[rvc]` から `[stream_vc.rvc]` へ写す必要がある。これを落とすと既定値 0 が使われ、モデルが学習した音域から外れた f0 を NSF ハーモニック源に与えることになり、「声としては出て内容も聞き取れるが、扇風機のような唸りが常時乗る」という原因特定の難しい壊れ方をする(実際に発生し、streaming 設計側の欠陥と誤認しかけた)。対策として `config.toml.example` の `[stream_vc.rvc]` は「個別にフィールドを拾う」のではなく「**動いている `[rvc]` を丸ごとコピーしてから streaming 用に調整する**」という書き方にし、`f0_up_key` に個別の注意書きを添えた。
