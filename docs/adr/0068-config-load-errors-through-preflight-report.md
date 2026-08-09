# 0068. 設定ファイルの読み込み失敗も preflight と同じ per-problem レポートで出す（0038 を refine）

- Status: Accepted
- Date: 2026-08-09
- Related: [ADR-0038](0038-worker-config-preflight-fail-loud.md), [ADR-0066](0066-config-input-file-only.md)

## Context

[ADR-0038](0038-worker-config-preflight-fail-loud.md) は「設定不備は起動時 preflight で fail-loud
に集約し、問題ごとに読める形で出す」と決めた。しかし preflight は `Config` オブジェクトを受け取る
ので、**その `Config` を作る過程で失敗した場合は preflight まで到達しない**。

具体的には `Config.read_config_from_file` が投げる 3 系統が素通りしていた。設定ファイルのタイポに
よる pydantic の `ValidationError`、TOML / JSON のパースエラー、UTF-8 でないファイルの
`UnicodeDecodeError` である。これらはエントリポイントで捕まえられず、利用者には生のスタック
トレースが出ていた。ライブラリ内部のフレームが並ぶだけで、どのキーが悪いのかは末尾を読まないと
分からない。

[ADR-0066](0066-config-input-file-only.md) で `--config` が唯一の入力経路になり、設定ファイルの
記述ミスが起動失敗の主要因になったため、この経路が塞がっていることの実害が大きくなった。

報告の順序にも制約がある。ログの整形は `configure_logger(config)` が行うが、それには `Config` が
要る。つまり読み込み失敗の報告は「まだ `Config` が無い」時点で行う必要がある。`configure_logger`
は同時に stdout/stderr を UTF-8 へ張り替えており（非コンソール出力で日本語が
`UnicodeEncodeError` になるのを防ぐため。ADR-0038 の目的そのもの）、その処理も前倒しが要る。

## Decision

設定ファイルの読み込み失敗を `ConfigError` に変換し、preflight の失敗とまったく同じ per-problem
レポートで出す。変換は `vspeech/preflight.py` に置く。ADR-0038 が「設定問題が表に出る唯一の場所」
と定めたのはこのモジュールであり、読み込み時に見つかった問題も同じ経路で届いて初めてその記述が
真になるため。

`ValidationError` は設定項目ごとに 1 エントリを持つので `ConfigProblem` へ 1 対 1 で写す。デコード
エラーはファイル全体に対する 1 件の問題として扱い、`field` は付けない（名指しできる設定項目が
無いため）。

stdout/stderr の UTF-8 化を `configure_logger` から `force_utf8_streams` として切り出し、エント
リポイントが `Config` を持つ前に呼べるようにする。

## Alternatives rejected

- **`Config.read_config_from_file` 自身に `ConfigError` を投げさせる** — 例外の変換が
  `config.py` に入り、スキーマ定義のモジュールが起動時の報告形式を知ることになる。この
  classmethod はテストからも直接呼ばれており、そこでは pydantic の `ValidationError` が
  そのまま見えたほうが検証しやすい。
- **エントリポイントで捕まえて `click.echo` で出す** — レポートの整形が preflight 側と
  エントリポイント側の 2 か所に分かれ、片方だけ直る。ADR-0038 が避けたかった状態そのもの。
- **`configure_logger` を設定読み込みより前に呼ぶ** — 不可能。ログの出力先もレベルも
  `Config` から決まる。
- **生のトレースバックのまま放置する** — 失敗は fail-loud で出てはいるが、ADR-0038 の要求は
  「落ちること」ではなく「読んで直せること」。スタックトレースはそれを満たさない。

## Consequences

設定ファイルのタイポは、preflight の検出結果と同じ `起動中止: 設定不備 N 件` + 1 行 1 問題の形で
出るようになり、悪いキーが最初の数行で分かる。終了コードは 1 のまま変わらない。

`vspeech/preflight.py` が `Config` を作る責務を持つようになった。このモジュールは「何が必要か」の
単一の権威であり、読み込みもその判断の一部と見なせる範囲に収まっている。

`force_utf8_streams` が公開関数として増えた。`configure_logger` も内部で呼ぶので、既存の呼び出し
側から見た挙動は変わらない。
