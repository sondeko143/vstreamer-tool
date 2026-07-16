# 0043. obs-websocket クライアントを websockets 上に自前実装する（simpleobsws を却下）

- Status: Proposed
- Date: 2026-07-16
- Related: spec [2026-07-16-subtitle-obs-websocket-backend-design.md](../superpowers/specs/2026-07-16-subtitle-obs-websocket-backend-design.md); [ADR-0042](0042-subtitle-obs-failure-tiers.md)（再接続ポリシー）

## Context

OBS バックエンドには WebSocket クライアントが要る。既存依存には無い（`httpx` は WS 非対応、`grpcio` は別物）ので、何かは足さざるを得ない。

必要なのは obs-websocket 5.x プロトコルのごく一部である: Hello(op 0) → Identify(op 1) → Identified(op 2) のハンドシェイクと、Request(op 6) / RequestResponse(op 7) の往復のみ。認証は `base64(sha256(base64(sha256(password + salt)) + challenge))`。イベント購読もバッチリクエストも msgpack シリアライザも使わない。

候補は 2 つ。実データを取った（py3.14 で両方解決することを確認済み）:

- `websockets` 16.1 — 依存ゼロ。
- `simpleobsws` 1.4.3（2025-06-20 リリース、約 13 ヶ月前）— `websockets>=14.0` + `msgpack` に依存。**単一ファイル 336 行**。

`simpleobsws` の実装を読んだところ、`reconnect` / `backoff` / `retry` の実装が **0 ヒット**だった（公開 API は `call` / `call_batch` / `connect` / `disconnect` / `emit` / `emit_batch` / `is_identified` / `wait_until_identified` / `register_event_callback` / `deregister_event_callback`）。

## Decision

`websockets` を直接の依存として追加し、obs-websocket 5.x クライアント（ハンドシェイク・認証・requestId 相関）を `lib/obs_ws.py` に自前実装する（約 100 行）。

`ObsClient` として自前の狭いインターフェースを切り、OBS バックエンドはそれにのみ依存する。テストはこのインターフェースをフェイクし、ネットワークにも OBS にも触れない。

## Alternatives rejected

- **`simpleobsws` を使う** — プロトコル実装を書かずに済み、obs-websocket 本家（IRLToolkit）がメンテしているのは実質的な利点。しかし:
  1. **肝心の再接続を貰えない。** `reconnect` / `backoff` / `retry` を実装しておらず、[ADR-0042](0042-subtitle-obs-failure-tiers.md) の fail-open ポリシー（バックオフ再接続 + warn once）は結局こちらが書く。一番面倒な部分が節約できない。
  2. **使うのは 1 割。** 必要なのは handshake + auth + `SetInputSettings` だけで、events / batch / msgpack は不要。`msgpack` は `install_requires` なので断れない。
  3. **依存の増分が 3 個 vs 1 個。** `simpleobsws` は `websockets` を節約してくれない（依存しているので）、その上に 2 個積む形になる。uv.lock は universal で `uv audit` は全体を走査するため、extra に逃がしても監査面からは隠せない。
  4. **前例。** `lib/ami.py` は AmiVoice の SDK を使わず、pydantic でワイヤ型をモデリングして httpx で叩いている。薄いプロトコルは自前で書く流儀が既にある。

  なお当初「テスト容易性」と「13 ヶ月の停滞」も却下理由に挙げたが、いずれも**取り下げた**。テストはどちらを選んでも自前の狭いインターフェースをフェイクするので差が無く（むしろ生 WS の方がハンドシェイクごと偽装が要る分だけ面倒）、obs-websocket 5.x は 2022 年から安定しているため薄いラッパーにリリースが無いのは停滞ではなく完成の証拠でもありうる。却下は上の 4 点のみを根拠とする。
- **`aiohttp` を入れて WS クライアントに使う** — WS クライアントとサーバの両方が手に入り、将来 SSE + ブラウザソース案に転ぶ場合の布石になる。しかし現時点でサーバは不要で、`websockets`（依存ゼロ）より重い。要らない選択肢のために依存を太らせている。
- **依存を増やさない（テキストファイル経由 + OBS の「ファイルから読む」）** — 依存ゼロ・ポートゼロで実装も最小。しかし `Text (GDI+)` のファイル読み込みは mtime を **1 秒間隔でポーリング**する（`obs-text.cpp` の `Tick()`、間隔は 1.0f 固定）。e2e 2.5〜3.0 秒のパイプラインに最大 1 秒を上乗せする。加えてスタイルを push できないため [ADR-0041](0041-subtitle-obs-config-authority.md)（config が権威）が成立しない。

## Consequences

- 追加依存は `websockets` 1 個（依存ゼロ、py3.14 で確認済み）。`uv audit` の面が最小になる。
- 認証・ハンドシェイク・requestId 相関の正しさは我々の責任になる。認証はネスト順（`base64(sha256(base64(sha256(password + salt)) + challenge))`）を間違えやすいので、仕様の既知ベクタでテストする。
- `websockets` の API 変更（過去に `websockets.client.connect` → `websockets.asyncio.client.connect` の移行があった）への追従は自分で被る。`lib/obs_ws.py` に閉じ込めることで影響範囲を 1 ファイルに限る。
- `ObsClient` インターフェースの裏に実装が隠れるため、将来 `simpleobsws` へ乗り換える判断をしても差し替えは局所で済む（その場合は本 ADR を supersede する）。
