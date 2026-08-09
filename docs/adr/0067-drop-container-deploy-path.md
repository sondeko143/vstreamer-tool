# 0067. コンテナ / Cloud Run 配備の経路を、Linux 側の依存解決ごと撤去する

- Status: Accepted
- Date: 2026-08-09
- Related: [spec](../superpowers/specs/2026-08-09-config-file-only-design.md), [ADR-0066](0066-config-input-file-only.md)

## Context

[ADR-0066](0066-config-input-file-only.md) で環境変数による config 注入を廃止すると、コンテナ
イメージは設定手段を失う。`Dockerfile` の ENTRYPOINT は `python -m vspeech` で `--config` を
渡しておらず、環境変数だけが設定経路だったため。

そこで配備が生きているかを確認したところ、既に機能していないことが分かった。

- `.vscode/launch.json` の "Cloud Run: Run/Debug Locally" 構成（`vspeech_*` 18 本を含む）は
  最終更新が **2023-05-28**。同ファイルは 2026-07-26 に削除された `vspeech.gui`
  （[ADR-0061](0061-remote-control-as-cli.md)）をいまだに参照している。
- `Dockerfile` の最終更新は Python 3.14 移行の機械的な FROM bump のみで、ヘッダのコメントは
  `requires-python = ">=3.12,<3.13"` のまま取り残されている。
- 環境変数経路を覆うテストは 1 件も無い。

加えて、`pyproject.toml` にはコンテナのためだけに存在する依存解決設定が 2 つある。
`[tool.uv] environments` の `sys_platform == 'linux'` と、`voicevox-core` の manylinux wheel
ピンで、いずれも CLAUDE.md に「Docker イメージのため」と明記されている。

## Decision

コンテナ配備の経路を一括で撤去する。対象は `Dockerfile`・`.dockerignore`・
`requirements-pod.txt`、poe の `requirements-pod` タスクと（それしか消さない）`clean` タスク、
`.vscode/launch.json` の Cloud Run 構成、および同ファイルで削除済みモジュールを指す `gui` 構成。

それを支えるためだけに存在する依存解決設定も同時に落とす。`[tool.uv] environments` を
`sys_platform == 'win32'` のみにし、`voicevox-core` の manylinux エントリを削除する。

結果、このプロジェクトは Windows 専用として一貫させる。

## Alternatives rejected

- **ENTRYPOINT に `--config` を足して配備経路を維持する** — 3 年動いていない経路のために、
  config を COPY / マウントする設計と Linux 側の依存解決を保守し続けることになる。動かして
  いないものは壊れていても気づけない。
- **成果物（`Dockerfile` 等）だけ消して uv の Linux 解決は残す** — uv.lock の再解決を避けられる
  が、存在理由が消えた設定が CLAUDE.md の説明つきで残る。これはドリフトの作り方そのもので、
  この後 `requirements-pod.txt` へ export されない `voicevox` extra のマーカー注意書きのような、
  検証されない知識だけが蓄積する。
- **環境変数経路を残してコンテナを生かす** — [ADR-0066](0066-config-input-file-only.md) の前提が
  崩れる。3 年動いていない配備のために、テストも文書も無い設定経路を保守し続けることになる。

## Consequences

uv.lock が Windows のみの解決になって縮み、`uv audit` の走査対象も減る。CLAUDE.md から
`requirements-pod` と per-platform ピンの注意書きが消え、説明すべき事柄が 1 つ減る。

一方で、Linux やコンテナで動かしたくなったら、`Dockerfile` と Linux 側の依存解決を起こし直す
ことになる。撤去した内容は git 履歴に残るので、復元の出発点はそこになる。

`uv lock` は `.venv` に触らないので稼働中のパイプラインに影響しないが、`uv sync` は実行中の
パイプラインがあると os error 5 で失敗する。ロックの更新と同期は分けて行う。
