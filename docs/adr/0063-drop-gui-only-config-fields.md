# 0063. GUI 専用だった config フィールドを互換シムを置かずに削除する

- Status: Accepted
- Date: 2026-07-26
- Related: [ADR-0061](0061-remote-control-as-cli.md)（GUI を削除し `cli`/`vsctl` にした決定。本 ADR はその後始末）; [ADR-0038](0038-worker-config-preflight-fail-loud.md)（設定不備は起動時に fail-loud）

## Context

[0061](0061-remote-control-as-cli.md) で GUI を削除したとき、`gui/` のコードは消えたが **GUI しか読んでいなかった設定は `Config` に残った**。

- `template_texts` — GUI のテキスト送信欄に並べる定型文の一覧。
- `text_send_operations` — その定型文を流す先の routing chain（既定 `[["tts", "playback"]]`）。

いずれも `vspeech/` からは一度も参照されない。実際、`Config` の全フィールドを機械的に走査して `vspeech` / `cli` / `scripts` / `tests` の参照を突き合わせると、**参照ゼロはこの 2 つだけ**だった（`vr2.params.*` / `voicevox.params.*` は名前で個別参照されないが `type(x).model_fields` 走査と `model_dump()` で一括適用されており、生きている）。

放置すると、設定ファイルの読者に「この pipeline は定型文を送れる」と読める嘘が残り、`config.toml.example` もそれを勧め続ける。

## Decision

**2 つのフィールドを `Config` と `config.toml.example` から削除する。互換のためのシム（deprecated な no-op フィールド）は置かない。**

`Config` は `extra="forbid"`（pydantic-settings の既定）なので、この 2 キーが残った設定ファイルは**起動時に落ちる**。それでよい — 落ち方は「`template_texts`: Extra inputs are not permitted」と**キー名を名指しする** ValidationError で、[0038](0038-worker-config-preflight-fail-loud.md) の fail-loud と同じ性質になる。

## Alternatives rejected

- **deprecated な no-op フィールドとして残す** — 「消したのに残っている」状態を作る。読者にとっては削除前と区別がつかず（設定しても何も起きないことは実行しないと分からない）、次に消すときに同じ判断をやり直すことになる。移行対象が実在しない（下記）以上、払う価値がない。
- **`extra="ignore"` にして未知キーを黙って捨てる** — 古いキーは通るようになるが、**typo も同時に通る**ようになる。設定を間違えたまま静かに既定値で走るのは、この repo が [0038](0038-worker-config-preflight-fail-loud.md) で潰した失敗そのもの。
- **移行スクリプト / 起動時の自動削除を書く** — 実在する 6 つの設定ファイルを新スキーマで読み込んで確認したところ、**どれも 2 キーを持っていなかった**（GUI は生成した pipeline config にしか書かなかった。GUI 経由の pipeline config 群は [0061](0061-remote-control-as-cli.md) の時点で読まれなくなっている）。移行される対象が無いので、機構だけが残る。

## Consequences

`Config` に「実装のどこにも繋がっていないフィールド」が無くなった。フィールド走査による棚卸しでも参照ゼロは 0 件になる。

- **この 2 キーを含む設定ファイルは起動しなくなる**。手元の 6 ファイルは影響なしを確認済みだが、GUI が生成した古い pipeline config（`~/.config/vstreamer/pipelines/*.toml`）を直接 `--config` に渡すと落ちる。対処は該当行の削除だけで、エラーがキー名を名指しする。
- `RoutesList` 型と `recording.routes_list` は残る。routing chain を設定に書く仕組み自体を消したわけではない。
