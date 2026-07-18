# 0048. GCP のトークン更新を retry 付き session に載せるため認証チャネルを自前で組む

- Status: Accepted
- Date: 2026-07-18
- Related: `vspeech/lib/gcp.py`, `vspeech/worker/translation.py` の `translate_request`（transient retry）

## Context

翻訳が不定期に失敗していた:

```
ConnectionResetError(10054, '既存の接続はリモート ホストに強制的に切断されました。')
  -> google.auth.exceptions.TransportError
  -> 503 Getting metadata from plugin failed
WARN translation transient error, retrying (1/5)
```

トークン更新は概ね有効期限（約 1 時間）ごとにしか起きないので、その間プールに
残った oauth2.googleapis.com への TLS 接続は Google 側にとっくに閉じられている。

urllib3 は接続を貸し出す前に `is_connection_dropped()` で死活を見るが、それで
拾えるのは「貸す前に既に死んでいた」接続だけである。本番の traceback は
`getresponse()` -> `_read_status()` -> `recv_into` で落ちており、**POST を書いた
後に RST が来ている** = チェックと書き込みの間に死ぬ窓であって、死活チェックでは
塞げない。実験でもこの区別を確認した:

- 接続を「貸す前に」閉じる再現 → 素の `requests.Session` でも成功してしまう
  （urllib3 が検出して張り直す）。
- リクエストを受け切ってから RST する再現 → 素の Session が本番と同一の
  `ConnectionResetError(10054, ...)` で失敗、retry 付き session は新しい接続で成功。

`google.auth.transport.requests.Request()` は session を渡さなければ素の
`requests.Session()`（retry 無し）を作る。そしてその `Request()` を作っているのは
`google.api_core.grpc_helpers._create_composite_credentials` で、**session を注入する
引数が無い**。`TranslationServiceAsyncClient(credentials=...)` に任せる限り、
更新は retry 無しの session で走り続ける。

## Decision

認証チャネルを自前で組み、`AuthMetadataPlugin` に retry 付き session を載せた
`Request` を渡す（`vspeech/lib/gcp.py` の `build_auth_session` /
`create_auth_metadata_plugin` / `create_auth_channel`）。そのチャネルを
`TranslationServiceGrpcAsyncIOTransport(channel=...)` に渡す — 実体のチャネルを
渡すとライブラリは credentials を無視する（`_ignore_credentials`）ので、認証経路は
完全にこちらの持ち物になる。

retry は接続層だけに効かせる（`status=0` かつ `respect_retry_after_header=False`）。
HTTP ステータスの解釈は google.auth 自身の役目で、そこに手を出すと
`invalid_grant` のような恒久エラーまで retry してしまう。**`status=0` だけでは
足りない**: urllib3 の `is_retry()` は `respect_retry_after_header`（既定 True）が
有効で `Retry-After` ヘッダが付いていると `status_forcelist` を一切見ずに
retry すべきと判断し、そこへ `status=0` が重なると即 `is_exhausted` になって、
429/503 が「retry もされず応答本文も失われた `RetryError`」に化ける（実測）。
それは google.auth に委ねるどころか、google.auth が持っていた retry を奪う。

`allowed_methods` には **POST を明示的に含める**: urllib3 の既定は非冪等な POST を
retry 対象から外すので、既定のままではトークン更新は 1 度も retry されない。

1 試行あたりの時間も頭打ちにする（`_BoundedTimeoutAdapter`、20 秒）。google.auth は
トークン更新の POST に timeout を渡さないので既定の 120 秒が効き、retry を足すと
最悪時間が試行回数ぶん伸びる。gRPC は認証 plugin を呼び出しごとの別スレッドで走らせ、
RPC の deadline を過ぎてもそのスレッドは生き残るため、応答を返さないエンドポイント
相手では詰まったスレッドが積み上がる（実測: 変更前 120 秒 → retry 追加で約 481 秒）。
20 秒で切れば最悪でも約 81.5 秒と、retry を入れた後でも**修正前より短い**。

## Alternatives rejected

- **grpc の ERROR traceback を抑止してログを静かにするだけ** — 一度実装したが、
  ユーザーの指示で巻き戻した。症状（二重に出るログ）だけを消して原因を残す対処で、
  翻訳が 1 回遅延する事実も、失敗の頻度が見えなくなる副作用も残る。
- **`credentials.with_non_blocking_refresh()`** — STALE 窓（`REFRESH_THRESHOLD`
  = 3 分 45 秒）の間はバックグラウンドで更新し、失敗しても RPC は有効な
  トークンで通る。しかしバックグラウンド更新は**タイマーではなく RPC が来て
  初めて起動する**。この配信パイプラインは無音が長く（実測で 27 分・1 時間 42 分の
  間隔）、その間 RPC が無いままトークンが INVALID になり、次の発話が blocking
  refresh に落ちる — まさに観測された経路なので、この負荷パターンでは効かない。
- **`translate_request` の retry に任せる（現状維持）** — 実際に回復はしている。
  ただし回復するのは「1 回失敗してから」で、その発話の翻訳が遅延する。
  原因側を直せるなら直す方がよい。
- **`google.auth.transport.requests.Request` を monkeypatch する** — 変更は最小だが、
  プロセス全体の第三者シンボルを差し替えることになり、無関係な GCP クライアントの
  挙動まで変える。
- **`allowed_methods=False`（全メソッド retry）** — urllib3 は受け付けるが型スタブ上
  `Collection[str] | None` に無く ty が弾く。`frozenset({"GET", "POST"})` と
  明示した方が「POST を意図して含めた」と読める。

## Consequences

- トークン更新が死んだプール接続を掴んでも、urllib3 が新しい接続で張り直すので
  翻訳は失敗しない。`translate_request` の retry は防御層として残る。
- **api_core がやっていることを手で再現した以上、そこに追随する責任を負う。**
  特に `with_scopes_if_required`（省くとサービスアカウントがスコープ無しトークンで
  拒否される）と `default_host`（省くとサービスアカウントの self-signed JWT 経路が
  無効になる）は明示的に合わせてある。前者は
  `tests/test_gcp_auth_retry.py::test_auth_plugin_applies_the_service_scopes` が
  守る — 単体テストは `AnonymousCredentials`（Scoped ではない）を使うので、
  scopes を落としても他のテストは全部 GREEN のままになるため。
- api_core との等価性は実際の GCP に対して確認済み: ADC（`google.oauth2.credentials`
  = 本番の traceback と同じ credential 型）で `create_translation_client` を組み、
  TranslateText が正常に応答することを実測した。単体テストだけでは
  scopes/`default_host`/TLS の等価性は証明できない。
- api_core は `ssl_credentials` 未指定時に `grpc.compute_engine_channel_credentials`
  を使い GCE の DirectPath を残すが、こちらは
  `composite_channel_credentials(ssl_channel_credentials(), ...)` を使う。
  非 GCE ホスト（このプロジェクトの実行環境）では等価で、GCE 上で動かす場合のみ
  DirectPath を得られない。
- **チャネルを渡すと GAPIC transport の `options` 分岐ごと飛ばされる。**
  `create_auth_channel` の呼び出し側が
  `grpc.max_send/receive_message_length = -1` を渡し直さないと、受信上限が
  gRPC 既定の 4 MiB に戻る（実測: 5 MiB の応答が `RESOURCE_EXHAUSTED`）。
  この翻訳の応答サイズでは実害が出ないが、「api_core が黙ってやっていたことを
  落とす」という本 ADR がまさに警戒している型の欠落なので明示的に渡す。
- **影響範囲は翻訳クライアントのみ。同じ窓が残っている場所が 2 つある**:
  `worker/transcription.py` の `SpeechAsyncClient(credentials=...)` と、
  `worker/sender.py` の ID トークン更新用 `Request()`。どちらも同じ理由で
  壊れうるが、今回の修正は実際の Translation API に対してしか検証できておらず、
  Speech は scopes/host/channel options が別なので、未報告の障害を先回りする
  ために最も重要なワーカーを壊す危険の方が大きいと判断して見送った。各所に
  理由つきのコメントを置いてある。
- 失われたものが 3 つある（いずれもこのプロジェクトでは未使用）:
  `client_options.api_endpoint` / `GOOGLE_API_USE_MTLS_ENDPOINT` /
  `GOOGLE_CLOUD_UNIVERSE_DOMAIN` によるエンドポイント上書き、`quota_project_id`、
  および 401/403/404 時に credential 種別を付記する
  `_add_cred_info_for_auth_errors`（`transport._credentials` が `None` になるため）。
- サービスアカウントの self-signed JWT 経路が保たれる理由は「`default_host` を
  合わせたから」だけではない。GAPIC の `always_use_jwt_access` は迂回されるが、
  `with_scopes_if_required` が `_scopes` を `None` のままにし `default_host` が
  audience を与えるので、`service_account.py` の
  `elif not self._scopes and audience:` の分岐で両経路とも同じ audience ベースの
  JWT に落ち着く（実 RSA 鍵で両経路を実行して確認）。等価性はこの 1 分岐に
  依存している。
