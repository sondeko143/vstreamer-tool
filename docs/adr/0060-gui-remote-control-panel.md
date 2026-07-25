# 0060. GUI を走っている pipeline へのリモート操作パネルへ縮小する（起動・設定編集を GUI から外す）

- Status: Superseded by [ADR-0061](0061-remote-control-as-cli.md)
- Date: 2026-07-26
- Related: spec [2026-07-26-pipeline-remote-control-design.md](../superpowers/specs/2026-07-26-pipeline-remote-control-design.md); [ADR-0032](0032-gui-multi-pipeline-rewrite.md) / [ADR-0033](0033-gui-manifest-versioning.md) / [ADR-0045](0045-gui-readiness-reuses-preflight.md) / [ADR-0046](0046-gui-shared-asset-paths-explicit-propagate.md)（本 ADR が supersede する）; [ADR-0034](0034-gui-corrupt-file-resilience.md)（壊れた入力でも起動する方針は維持）; [ADR-0038](0038-worker-config-preflight-fail-loud.md)（起動時 preflight は vspeech 側にそのまま残る）

## Context

[0032](0032-gui-multi-pipeline-rewrite.md) の GUI は「1 pipeline = 1 サブプロセス = 1 config = 1 port」を**このマシンで**管理する道具として作られ、config フォーム・レシピ・起動前 readiness（[0045](0045-gui-readiness-reuses-preflight.md)）・共有素材パスの伝播（[0046](0046-gui-shared-asset-paths-explicit-propagate.md)）・profile と migration（[0033](0033-gui-manifest-versioning.md)）まで抱えた（gui/ で約 2300 行）。

実運用の形はそうならなかった。pipeline は録音マシン・GPU マシン・再生マシンに分かれて**すでに走っている**（[pipeline のトポロジ](0035-bound-sender-reconnect-backoff.md) が前提にしている構成）。この状態で欲しい操作は「そこへ届いているか」「止める / 再開する」「config を読み直させる」の 3 つで、いずれも既存 GUI には無い。逆に GUI が持っている起動・編集の層は、**自分のマシンで自分が起動した pipeline にしか効かない**うえ、config スキーマに結び付いているので設定を足すたび追随コストだけを生んでいた。

一方で、その 3 操作を運ぶ経路は既にある。`Operation` の `PAUSE` / `RESUME` / `RELOAD` / `PING` は wire contract（vstreamer-protos）に定義済みで、受け側は `receiver` → `process_command` で同期的に処理する（データイベントと違い worker queue を経由しない）。足りていなかったのは、それを押せる場所だけだった。

## Decision

**GUI は「走っている pipeline へ制御イベントを 1 本送る」だけの操作パネルにする。** pipeline の起動・停止・config 編集・readiness 表示は GUI の役目から外し、該当コードを削除する。

- 送れるのは **ping（疎通確認）/ pause / resume / reload** の 4 つだけ。データイベントも `set_filters` / `forward` も送らない。
- GUI が保持するのは**宛先だけ**（名前・ホスト・ポート・reload 用の config パス）。`~/.config/vstreamer/targets.toml` に永続化する。Config そのものは一切持たない — したがって GUI は config スキーマから独立する。
- **疎通確認は gRPC の PING を実際に送って判定する**。TCP 接続の成否ではなく、受け側が Command を解釈して `Response` を返すところまでを到達の証拠とする。往復時間も出す。
- **RPC には必ず deadline を付け（既定 3s）、送信は別スレッドで行う**。応答しないホストを選んでも UI は固まらない。結果は `after(0, ...)` で UI スレッドへ戻す。
- **reload の config パスは「対象マシン上のパス」としてテキストで持つ**。受け側が自分で `open` するため、こちら側では解決も存在検査もしない。空のまま reload を送るのは GUI 側で拒否する（受け側の `WorkerInput` validation に落ちるだけで、理由が分かりにくいため）。
- **Command の組み立ては `EventAddress.to_pb()` を通す**。GUI から protobuf の `PAUSE` などを直に触らない — `EventType` ↔ `Operation` の対応が 2 箇所に分かれると片方だけずれる。
- 壊れた `targets.toml` は退避してから空の一覧で起動する（[0034](0034-gui-corrupt-file-resilience.md) の方針を踏襲）。

## Alternatives rejected

- **マネージャを残したまま操作タブを足す** — 起動・編集の層が死荷重として残り、config スキーマへの追随コストを払い続ける。さらに「このマシンで起動した pipeline」と「宛先」という二重の対象概念が同じ画面に並び、どちらを操作しているのか分からなくなる。
- **profile（`pipelines.toml`）を宛先一覧として流用する** — port は持つがホストを持たない（全て自マシン前提）ので、別マシンで走っている pipeline を指せない。かつエントリごとに Config 一式を抱えるため、GUI がまた config スキーマに縛られる。
- **GUI ではなく CLI サブコマンドにする**（`vspeech ctl pause --to host:port`）— 実装は確かに小さい。だが実際の使い方は「配信中に開きっぱなしにして必要な瞬間に押す」で、都度コマンドを組むより押せる方が合う。排他ではないので、必要になれば同じ `gui/client.py` を使って後から足せる。
- **reload のパスをファイル選択ダイアログで選ばせる** — 選べるのは自分のマシンのパスであって、対象マシンのパスではない。別マシン相手には嘘の補助になるので、テキスト欄 + 「対象マシン上のパス」の明示に留める。
- **疎通確認を TCP connect で済ませる** — ポートが開いているかしか分からない。receiver は生きているが pipeline が処理不能という状態を「OK」と表示してしまう。
- **`sync_process_command` を使う** — 受け側は両 RPC とも同じ `process_command` を通して `Response(result=True)` を返すだけで差が無い。既存 GUI が使っていた `process_command` に揃える。
- **pause 中かどうかを GUI に表示する** — 状態を問い合わせる Operation が wire contract に無い。追加するには vstreamer-protos 側の変更（別リポジトリ・別リリース）が要り、この縮小の範囲を超える。

## Consequences

GUI は `vspeech.config.EventType` と `vspeech.shared_context.EventAddress` にしか依存しなくなり、config を増やしても追随が要らない。gui/ は約 2300 行から約 400 行になった。

- **GUI から pipeline を起動できなくなる**。起動は `uv run python -m vspeech --config <file>` で行う。config の編集も直接ファイルを触る。
- [0045](0045-gui-readiness-reuses-preflight.md) の起動前 readiness 表示は無くなる。**`vspeech/preflight.py` 自体は起動時の fail-loud（[0038](0038-worker-config-preflight-fail-loud.md)）としてそのまま残る**ので、設定不備は起動時に全件まとめて出る。「必須項目リストを GUI に複製するな」という禁は、複製できる場所が消えたことで自動的に守られる。
- [0046](0046-gui-shared-asset-paths-explicit-propagate.md) の共有素材パス伝播も無くなる。マシン共通のパスは各 pipeline の config を直接編集する。
- 既存の `~/.config/vstreamer/{default.toml, pipelines.toml, pipelines/*.toml}` は読み書きしなくなる。**削除もしない** — GUI を介さず `--config` で起動する材料としてそのまま使えるため。
- 認証・暗号化は無い（`insecure_channel`）。LAN 内前提という既存の transport の前提を変えていない。
- pause 中かどうかは GUI からは分からない。ping が返るのは「receiver が生きている」ことだけで、`running` ゲートの状態は取れない。
