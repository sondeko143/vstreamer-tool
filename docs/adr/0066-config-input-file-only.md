# 0066. config の入力を `--config` ファイル 1 本に統一する（pydantic-settings を撤去し `--config` を必須にする）

- Status: Accepted
- Date: 2026-08-09
- Related: [spec](../superpowers/specs/2026-08-09-config-file-only-design.md), [ADR-0067](0067-drop-container-deploy-path.md), [ADR-0038](0038-worker-config-preflight-fail-loud.md)

## Context

config の入力経路は `--config` ファイルと環境変数の 2 本ある。環境変数側は pydantic-settings の
`BaseSettings` に依存している。

2026-08-09 に実測したところ、`import pydantic_settings` 単体が **+13.7 MB RSS / +176 modules /
473 ms** を要していた。原因は `pydantic_settings/sources/providers/__init__.py` が AWS Secrets
Manager・Azure Key Vault・GCP Secret Manager・CLI・dotenv・JSON・pyproject・secrets・TOML・YAML
の全プロバイダを無条件に import することにある。導入済みの 2.14.2 が最新なので、更新では解消
しない。これは起動時にロードされるアプリ中核（60 MB / 約 1.0 s）の中で単一としては pydantic 本体
（5.7 MB）より大きい。

ただしこの 13.7 MB / 176 modules は pydantic だけを読み込んだ状態での単体計測であり、そのまま
パイプラインの削減量にはならない。実際の起動経路では grpc と google-cloud 系が `ssl` /
`importlib.metadata` / `argparse` / `zoneinfo` などを既に読み込んでいるため、pydantic-settings
固有だったのは実測で **32 modules / 約 1.6 MB** にとどまった（`import vspeech.main` の増分が
+685 → +653 modules、+45.1 MB → +43.5 MB）。この決定を支えているのは削減量ではなく、下に述べる
「対価のない経路」の方である。

対して、この依存から実際に使っている機能は `env_prefix="vspeech_"` と
`env_nested_delimiter="__"` の 2 つだけで、参照箇所も `vspeech/config.py` 1 ファイルに閉じている。
`.env` も `secrets_dir` も `settings_customise_sources` も使っていない。

さらに、その環境変数経路には利用実績の裏づけがない。覆うテストは 1 件も無く、README にも
`config.toml.example` にも記載が無い。実質的な唯一の利用者はコンテナ配備だったが、それは
[ADR-0067](0067-drop-container-deploy-path.md) で撤去する。

経路が 2 本あること自体にも実害がある。設定を明示せず `Config()` で組んでいるテストが約 40 箇所
あり、これらは開発者の環境に `vspeech_*` があれば拾ってしまう。

## Decision

`Config` を `BaseSettings` から `BaseModel` に移し、pydantic-settings を依存から外す。環境変数に
よる config 注入を廃止し、`--config` を必須にする。未指定は使用法エラーで非ゼロ終了させる。

`BaseSettings` から暗黙に継承していた `extra="forbid"` は `ConfigDict` で明示的に引き継ぐ。素の
`BaseModel` の既定は `extra="ignore"` であり、そのまま載せ替えると設定ファイルのタイポが黙って
無視される退行になるため。

`listen_port` の `validation_alias=AliasChoices("listen_port", "PORT")` は撤去する。`PORT` は
Cloud Run が注入する契約のためだけに存在していた。

## Alternatives rejected

- **自前の env loader を書いて環境変数経路を維持する** — prefix・ネスト区切り・複合型の JSON
  パース・大文字小文字の非区別・alias の 5 挙動を手で再現することになる。利用者もテストも文書も
  無い経路のために、保守対象のパースコードだけが増える。
- **`pydantic_settings` を遅延 import にして経路だけ残す** — `Config` が `BaseSettings` を継承
  する以上、クラス定義時点で import が必要になる。`create_model` で動的に `BaseSettings` 派生を
  組めば回避できるが、config スキーマの定義が二重化して読めなくなる。
- **現状維持** — テストも文書も利用者も無い経路を保守し続けることになる。常駐プロセスへの
  負荷そのものは実測 32 modules / 約 1.6 MB で、これ単独では動機にならない。
- **`--config` 未指定を全デフォルト起動のまま残す** — 全 worker 無効で receiver/sender しか
  起動しない無意味なプロセスが黙って立ち上がる。[ADR-0038](0038-worker-config-preflight-fail-loud.md)
  の fail-loud 方針に反する。

## Consequences

設定の入口が 1 つになり、`Config()` が開発者環境に依存しない hermetic な既定値になる。常駐
プロセスは実測で 32 modules / 約 1.6 MB 軽くなり、依存が 1 つ（推移的に python-dotenv も）減る。
`typing-inspection` は pydantic 本体の直接依存なので残る。起動コストの削減は当初の見込み
（13.7 MB / 176 modules）を大きく下回った。

一方で、環境変数だけで設定を差し込む配備は不可能になる。将来必要になったら、config ファイルを
生成して `--config` で渡す形に寄せる。

`extra="forbid"` は `BaseSettings` 由来の暗黙の挙動だったため、明示化したことを回帰テストで
固定する。CLI 契約が変わる（`--config` 必須）が、`vsctl` は `Config` に依存しないので
[ADR-0061](0061-remote-control-as-cli.md) の境界には影響しない。
