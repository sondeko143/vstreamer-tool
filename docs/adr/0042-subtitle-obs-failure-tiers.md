# 0042. OBS 接続の失敗を「観測できたものだけ即死」で階層化する（0038 を refine）

- Status: Proposed
- Date: 2026-07-16
- Related: spec [2026-07-16-subtitle-obs-websocket-backend-design.md](../superpowers/specs/2026-07-16-subtitle-obs-websocket-backend-design.md); [ADR-0038](0038-worker-config-preflight-fail-loud.md)（本 ADR が refine）; [ADR-0037](0037-transcription-vad-skip-gate.md)（fail-open + fail-loud の非対称）; [ADR-0019](0019-vc-silero-vad-gate.md)（エラー非対称の初出）; [ADR-0035](0035-bound-sender-reconnect-backoff.md)（再接続バックオフの前例）

## Context

[ADR-0038](0038-worker-config-preflight-fail-loud.md) は「設定不備は起動時 preflight で fail-loud に集約する」を全 worker へ一般化した。しかし OBS バックエンドは、これまでのどの worker とも違う失敗の形を持つ。**別プロセス（OBS）が、vspeech とは独立のライフサイクルで生き死にする**。

これは preflight の検査対象（ファイル存在・デバイス発見・依存の有無）に収まらない。かつ失敗の種類によって正しい対応が正反対になる:

- OBS を vspeech より後に起動する、配信中に OBS を再起動する、というのはごく普通の操作である。これで vc / playback ごと落ちるのは字幕（音声の従属物）としては釣り合わない。
- 一方、パスワードの typo やソース名の typo は、何度リトライしても直らない。無限に警告を流し続けるのは「字幕が出ないがログに警告が出続けるだけ」という気づきにくい失敗になり、ADR-0019 / 0037 / 0038 で積み上げた fail-loud 文化に逆行する。

決定的な制約が 1 つある: **OBS に接続できていない間は、この 2 種類を区別できない。** ソース名が typo なのか、OBS がまだ起動していないだけなのかは、繋がってみるまで分からない。

## Decision

失敗を「**観測できたものだけ即死させる**」で階層化する。

| 事象 | 層 | 挙動 |
|---|---|---|
| 接続先 url 未設定 / ソース名が空 | preflight（[ADR-0038] Layer A） | `ConfigProblem` を集約して exit 1 |
| 接続拒否（OBS 未起動） | 実行時 | **fail-open**: warn once + バックオフ再接続 |
| 接続中の切断 | 実行時 | **fail-open**: warn once + バックオフ再接続 → 復帰時にスタイルと現在テキストを再 push |
| 認証失敗 | 接続確立後（[ADR-0038] Layer B） | **fail-loud**: `WorkerStartupError` |
| 指定したソースが存在しない | 接続確立後（[ADR-0038] Layer B） | **fail-loud**: `WorkerStartupError` |

preflight の `_check_subtitle` は `subtitle.enable` かつ `worker_type == OBS` のときだけ効く（TK 構成に新しい失敗を持ち込まない）。

fail-loud の判定は「接続できて、観測できて、それでも駄目だったとき」に行う。したがって `WorkerStartupError` は OBS が起動するまで（数分後かもしれない時点で）送出されうる。これは矛盾ではなく、**このワーカーの起動が OBS の起動まで遅延しているだけ**と解釈する。

## Alternatives rejected

- **全て fail-open（何が起きても warn + リトライ）** — パイプラインは絶対に死なないが、`text_source = "vspeech-txt"` のような typo が「字幕が出ない + ログに警告が流れ続ける」だけになり、気づきにくい。ADR-0019 / 0037 / 0038 が一貫して避けてきた失敗の形そのもの。
- **全て fail-loud（OBS が無ければ起動しない）** — 一貫していて実装も単純だが、「OBS を vspeech より後に起動する」「配信中に OBS を再起動する」という普通の操作で vc / playback ごと落ちる。字幕は音声の従属物であり、表示先の都合で音声を殺すのは衡平を欠く。
- **接続もソース検査も preflight（Layer A）で行う** — 設定不備を起動前に集約できて ADR-0038 の形に最も忠実だが、preflight は TaskGroup を開く前に同期的に走るため、「OBS がまだ起動していない」を「設定不備」と誤判定して exit 1 する。上の全 fail-loud と同じ問題を、より早い段階で起こすだけ。
- **接続不能を一定回数リトライした後に fail-loud へ昇格する** — 「OBS を後から起動する」を許しつつ typo も検出できるように見えるが、閾値（何秒待てば「後から起動」ではなく「設定ミス」なのか）に非自明な根拠が無い。OBS を起動するのが 10 分後の運用は普通にありうる。区別できない情報を時間で推測するより、区別できるようになるまで待つ方が正しい。
- **ソース不在を fail-open（作られるまで待つ）** — `CreateInput` を却下し構造をユーザー所有とした（[ADR-0041](0041-subtitle-obs-config-authority.md)）以上、ソースが現れるのは人が手で作ったときのみ。それは設定と OBS の不一致であり、待っても直らない典型。

## Consequences

- OBS の起動順に依存しない。vspeech を先に上げても、配信中に OBS を再起動しても、音声パイプラインは無傷で字幕だけが自動復帰する。
- typo は繋がった瞬間に落ちて理由が出る。
- OBS が永久に起動しない環境では、認証・ソース名の誤りは検出されないまま warn が流れ続ける。これは「区別できないものを推測しない」ことの代償として受け入れる。
- `WorkerStartupError`（[ADR-0038] Layer B）が、プロセス起動から数分後に送出されうる。`main.py` の `except* WorkerStartupError` は起動時のみを想定した命名だが、機構としてはそのまま機能する（集約ログ → 終了）。名前と実態の乖離は、OBS の起動を待つという本 ADR の帰結として意図的に許容する。
- **この worker では [ADR-0038] の `worker_startup` コンテキストマネージャを使わない。** あれは `except Exception` ですべてを `WorkerStartupError` に変えるので、identify 中のタイムアウト（＝リトライすれば直る）まで fail-loud に化け、本 ADR の階層をその場で壊す。代わりに、観測済みかつ回復不能と型で分かる 2 つ（`ObsIdentifyError` / `ObsResourceNotFoundError`）だけを捕まえて `WorkerStartupError` を直接送出する。層B の意図（起動時のリソース取得失敗を `WorkerStartupError` にする）は満たしつつ、包む対象を広げない。他の worker は失敗が「取れるか取れないか」の二値なので `worker_startup` の全捕捉で正しいが、この worker だけは同じ接続の上で回復可能な失敗と不能な失敗が混ざる。
- 上記の帰結として、`lib/obs_ws.py` は**自分の失敗をすべて自前の型に包む義務がある**。素の `TimeoutError` や `KeyError` を漏らすと、fail-open の `except` 節（`OSError` / `WebSocketException` / `ObsProtocolError`）を素通りして worker を貫通し、TaskGroup ごとプロセスを落とす — 字幕の都合で音声を殺すという、本 ADR がまさに避けようとした事象になる。この穴は Task 2 のレビューで実際に見つかり、塞いだ。
- 再接続のバックオフは [ADR-0035](0035-bound-sender-reconnect-backoff.md) が sender で有界化したのと同じ動機（冷起動からの復帰を遅らせない）を持つが、実装は共有しない（gRPC チャネルの再接続とは別機構）。
