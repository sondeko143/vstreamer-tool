# 0061. 走っている pipeline の操作を GUI ではなく CLI で提供する（gui extra と依存 3 つを削除）

- Status: Accepted
- Date: 2026-07-26
- Related: spec [2026-07-26-pipeline-remote-control-design.md](../superpowers/specs/2026-07-26-pipeline-remote-control-design.md); [ADR-0060](0060-gui-remote-control-panel.md)（本 ADR が supersede する。操作の中身 — 4 イベント・`EventAddress.to_pb()` 経由・deadline 必須・reload パスは対象マシン側 — はそのまま引き継ぐ）; [ADR-0038](0038-worker-config-preflight-fail-loud.md)

## Context

[0060](0060-gui-remote-control-panel.md) で GUI をマネージャから「ping / pause / resume / reload を送るだけ」の操作パネルへ縮小した。縮小しきった結果、GUI であることが担っていたのは**宛先の一覧・入力欄・ボタン**だけになった。中身は「宛先を 1 つ決めて操作を 1 回送り、結果を 1 行見る」で、これは 1 プロセス 1 呼び出しの形そのものである。

その形のために `gui` extra は `ttkbootstrap` / `pillow` / `platformdirs` の 3 つを抱えていた。とくに `pillow` はネイティブ拡張で CVE 追随の対象になり続ける（直近も 12.2→12.3 のバンプで 9 件を処理している）。押しボタン 4 つの対価としては重い。

一方、CLI にすると GUI では出せなかったものが出る。**終了コードが操作の成否になる**ので、配信ソフトのコマンド実行・ホットキー・シェルスクリプトから合成できる。追加依存もゼロで済む — `click` も `grpcio` も既にコア依存で、`vspeech` からは `EventType` と `EventAddress` を借りるだけ。

## Decision

**`gui/` を削除し、`cli/` パッケージ + `vsctl` コマンドにする。** `gui` extra とその 3 依存も削除する。

- 入口は `[project.scripts]` の **`vsctl`**（`cli.main:main`）と `python -m cli`。サブコマンドは操作名そのまま **`ping` / `pause` / `resume` / `reload`**。
- **宛先は保存しない**。`--to HOST:PORT` を各サブコマンドに置き、既定値は環境変数 **`VSPEECH_TARGET`**（click の `envvar`）。
- `reload` は `--config-path` 必須。**対象マシン上のパス**という性質は 0060 のまま — 受け側が自分で `open` するので、こちらでは解決も存在検査もしない。
- **終了コードは操作の成否そのもの**（0 = 相手が受け取った / 1 = 失敗）。使い方の誤りは click の 2。失敗行は stderr、成功行は stdout。
- **宛先は送る前に `host:port` として検証する**。port を落とした `--to 192.0.2.1` を素通しすると gRPC が名前解決に失敗するまで待ち、「deadline まで無反応」になるため。IPv6 は角括弧を要求する（無いと `::1` が host `::` / port `1` として通る）。
- **entry point で stdout/stderr を UTF-8 へ差し替える**（`errors="backslashreplace"`）。help もエラーも日本語なので、Windows の既定 cp932/cp1252 のままだと `vsctl --help` が `UnicodeEncodeError` で落ちる。`vspeech.logger` と同じ手当てで、同じ罠。
- 送信本体（`cli/client.py`）は 0060 のものをそのまま移した。`EventAddress.to_pb()` 経由で組み、deadline を必ず付ける。

## Alternatives rejected

- **GUI を残す（CLI と併存させる）** — 依存 3 つを押しボタン 4 つのために保持し続けることになる。押す操作は端末で打つのと手数がほぼ変わらず、CLI 側だけがスクリプト・ホットキーから呼べる。
- **宛先を toml に保存する（0060 の `targets.toml` 相当を CLI でも持つ）** — 保存場所を決める必要が出て、`platformdirs` か決め打ちパスが要る（消したばかりの依存が戻る）。シェルの履歴・alias・環境変数で足り、`VSPEECH_TARGET` を既定値にすれば繰り返し操作の手数は保存方式と同じになる。
- **`vspeech` のサブコマンドにする（`vspeech ctl pause`）** — `vspeech/main.py` の変更が要る。今回の「vspeech 側は触らない」制約に反するうえ、起動コマンドと制御コマンドが同じ入口に並ぶと `--config`（起動する設定）と `--config-path`（読み直させる設定）の意味が紛らわしくなる。
- **`--to` をグループ側のオプションにする（`vsctl --to X pause`）** — 操作名を先に打てる方が短く、環境変数で省略したときにオプションの位置が揺れない。
- **対話シェル / REPL** — 状態を持たない 4 操作に対して過剰。
- **help やエラーを ASCII に寄せて encoding 問題を回避する** — 既存のドキュメント・ログ・コメントが日本語である以上そこだけ英語にするのは不整合で、しかも他人の環境の encoding 問題を根本では消せない。

## Consequences

`vsctl ping --to host:port` が base install だけで動く（extras 不要）。終了コードで判定できるので、配信ソフトのコマンド実行やホットキーから直接呼べる。

- **`uv sync --extra gui` は無くなる**。extras 一覧から `gui` が消え、`ttkbootstrap` / `pillow` / `platformdirs` が lock から落ちる（92 → 89 パッケージ）。
- `cli` がビルド対象に入る（`[tool.uv.build-backend] module-name = ["vspeech", "cli"]`）。外すと `vsctl` の entry point が解決できない。
- 宛先を覚えないので、複数マシンを頻繁に叩くなら `VSPEECH_TARGET` か shell alias を各自で用意する。GUI が出していた「一覧に直近の疎通結果を並べる」表示も無くなる。
- `bandit` の走査対象が `vspeech gui` から `vspeech cli` に変わる。
- **`vspeech/logger.py` のコメントが「GUI サブプロセスの pipe」に言及したまま残っている**。指しているコード（stdout/stderr の UTF-8 差し替え）は今も必要で、`vsctl` が同じ手当てを自前で持っている。`vspeech` 側は今回触らない方針なので据え置いた — 次に触る人が文言だけ直すとよい。
