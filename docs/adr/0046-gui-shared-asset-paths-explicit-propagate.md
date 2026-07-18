# 0046. マシン共通の素材パスは default.toml で編集し明示 propagate する（pipeline config の自己完結を保つ）

- Status: Accepted
- Date: 2026-07-17
- Related: spec [2026-07-16-gui-run-readiness-design.md](../superpowers/specs/2026-07-16-gui-run-readiness-design.md); [ADR-0032](0032-gui-multi-pipeline-rewrite.md)（1 pipeline = 1 独立サブプロセス = 1 専用 Config）; [ADR-0045](0045-gui-readiness-reuses-preflight.md)

## Context

`rvc.model_file` / `rvc.hubert_model_file` / `rvc.rmvpe_model_file`、`voicevox.openjtalk_dir` / `model_dir` / `onnxruntime_path` は、マシンに 1 セットしか無い重い資産のパスである。しかし新規 pipeline の生成時に `default.toml` のスナップショットが各 pipeline の toml へコピー固定されるため、値はそこで凍る。

結果、資産パスを設定するには pipeline ごとに再入力が要り、既存 pipeline への一括更新もできない。さらに GUI から `default.toml` を編集する手段が無い（旧・単一 config GUI は [ADR-0032](0032-gui-multi-pipeline-rewrite.md) で削除された）。「新規 pipeline はプリセットから追加設定なしで動く」という [ADR-0032](0032-gui-multi-pipeline-rewrite.md) の狙いも、資産を要する vc / voicevox 構成では実質成立していない。

## Decision

マシン共通の素材パスを「共有（既定）」として GUI から `default.toml` 上で**一度だけ編集**できるようにし、**「全 pipeline へ反映」の明示操作**で既存の各 pipeline config へ書き込む。

- 各 pipeline の toml は全フィールドを持つ**自己完結ファイルのまま**とし（[ADR-0032](0032-gui-multi-pipeline-rewrite.md) の 1 pipeline = 1 専用 Config を維持）、起動時の合成は行わない。
- 新規 pipeline は従来どおり `default.toml` のスナップショットを継承する。
- propagate は**明示操作のみ**。暗黙の自動同期はしない。

## Alternatives rejected

- **`machine.toml` に資産パスを分離し、pipeline config から外して起動時にマージする** — pipeline の toml 単体では起動できなくなり、[ADR-0032](0032-gui-multi-pipeline-rewrite.md) の「1 pipeline = 1 専用 Config」の自己完結性を破る。GUI を介さず `python -m vspeech --config pipelines/<id>.toml` で起動する経路も壊れる。重複は消えるがその代償が高い。
- **`default.toml` を編集したら全 pipeline へ自動同期する** — pipeline ごとに意図的に別の資産（別 RVC モデル）を使う構成を黙って踏み潰す。同期は明示操作に留める。
- **現状維持（新規 pipeline の継承のみ）** — 既存 pipeline の一括更新ができず、pipeline ごとの再入力という摩擦がそのまま残る。

## Consequences

- 資産パスの入力が 1 回で済み、既存 pipeline へも一括で行き渡る。
- pipeline config は自己完結のままなので、GUI 外からの起動・可搬性・[ADR-0032](0032-gui-multi-pipeline-rewrite.md) は保たれる。
- 代償として値は依然 pipeline ごとに重複して保存される。propagate し忘れた pipeline は古い値のまま残るが、それは [ADR-0045](0045-gui-readiness-reuses-preflight.md) の readiness が ✗ として顕在化させる（黙って壊れない）。
- pipeline 個別に別の資産を使う構成は propagate で上書きされる。propagate は対象と変更内容を提示したうえで実行する必要がある。
