# 0038. 設定不備は起動時 preflight で fail-loud に集約する（全 worker へ一般化）

- Status: Accepted
- Date: 2026-07-16
- Related: spec [2026-07-16-worker-config-preflight-design.md](../superpowers/specs/2026-07-16-worker-config-preflight-design.md); [ADR-0019](0019-vc-silero-vad-gate.md) / [ADR-0037](0037-transcription-vad-skip-gate.md)（VAD モデル不在の起動時 fail-loud、本 ADR がその原則を全設定へ一般化）; [ADR-0031](0031-audio-pyaudio-to-sounddevice.md)（デバイス解決の基盤）

## Context

enable した worker の設定不備が現状 3 通りにばらつく。(A) worker セットアップで例外が出ると TaskGroup が全 worker を巻き込んで崩壊し、生の traceback が最上位に出るだけで「どの worker のどの設定が原因か」が読めない。(B) ACP の認証フィールドが空・参照先デバイスが不在でも worker は起動し、発話ごとに一時的な外部エラーに見えるログを吐き続けるため、設定ミスと気づけず「動いているのに無出力」になる。(C) worker_type の Enum など実際には起きえない不備への到達不能な防御分岐が散在する。

[ADR-0019](0019-vc-silero-vad-gate.md) / [ADR-0037](0037-transcription-vad-skip-gate.md) は既に「VAD モデル不在・ロード失敗は起動時に fail loudly」を選び、reload 時にゲート有効化＋モデル不在でパイプライン全体が落ちる挙動も「設定ミスは全体を落として明示的に気づかせる方が望ましい」として現状維持を決めている。この原則は VAD に限らず全 worker の設定不備に等しく当てはまるが、機構が VAD だけに個別実装されており、他の設定不備は上記 (A)(B)(C) のまま放置されている。

## Decision

worker の設定不備を **FATAL / DEGRADE の 2 バケツ**に分類し、[ADR-0019](0019-vc-silero-vad-gate.md) の起動時 fail-loud を全設定へ一般化する。

- **FATAL（worker がその設定で機能しない）→ プロセス全体を abort**。worker 単位の隔離はしない。1 worker の致命的不備で他 worker だけ生かすと「動いてるのに無出力」を招き、[ADR-0019](0019-vc-silero-vad-gate.md) の「全体を落として気づかせる」原則にも反するため。
- **DEGRADE（worker は有用な仕事を続けられる）→ WARNING を出して feature 単位で縮退**（worker 本体は継続）。

機構は **2 層**にする。

- **層A: 集中 preflight（安価な検査、TaskGroup を開く前）**。enable 済み全 worker の設定値・参照ファイル/ディレクトリの存在・デバイス発見可否・依存の有無だけを検査し、**検出できた全問題を 1 つのエラーに集約**して送出。タスク未 spawn なので ExceptionGroup ノイズが無く、原因の worker と設定値を名指しする整形ログを traceback 無しで出せる。実 HW・ネットワーク・GPU に触れないので自動テストで固定できる。
- **層B: worker startup での深層失敗の属性付け**。モデル実ロード・ONNX セッション構築・GPU 確保・ストリーム open など「試すまで分からない」失敗は、worker 起動時に worker 名と原因を持つ専用例外へ変換し、整形ログを出して停止する。既存の良メッセージ（`check_cuda_provider` / `create_vad_session`）はこの形へ合流。

個別の確定事項:

- **ACP は appkey / engine_uri / engine_name / service_id の 4 つを必須**とし、いずれか空なら発話を待たず起動時に停止する（現状 (B) の主眼）。将来別エンジンで一部を空にしたくなったら本 ADR を supersede する。
- **デバイス名→index の解決を単一の共有 resolver に一本化**し、preflight と worker（reload での再構築経路を含む）が同一経路を通る。これにより worker 側の独自「デバイス未発見」ガードを削除する（reload で config が name のみで読み直されるため worker は再解決の口が要る＝共有 resolver がその口を兼ねる）。
- **DEGRADE の代表例**: 録音ログ保存先が書込不可（ログのみ無効）、VAD 推論中の例外（そのチャンクを素通し=[ADR-0019](0019-vc-silero-vad-gate.md) のエラー非対称を踏襲）、warmup 失敗、録音 overflow、per-request の一時的外部エラー。

## Alternatives rejected

- **worker 単位の隔離（致命的 worker だけ skip し他は生かす）** — essential worker（transcription 等）が黙って skip されると recording が無出力先へ流し続け「動いてるのに何も起きない」になりゴール（明確な失敗）と衝突。TaskGroup に隔離機構を足す必要もある。[ADR-0019](0019-vc-silero-vad-gate.md) の「設定ミスは全体を落として気づかせる」と不一致。
- **pydantic の `model_validator` だけで完結させる** — ファイル存在・デバイス発見・依存の有無・実ロードは config スキーマの範囲外の環境依存事項。スキーマ検証に環境 I/O を混ぜると config ロード自体が環境依存になり、GUI→main のハンドオフなど config を読むだけの経路でも走ってしまう。値・フィールド間検査（範囲、必須の組）は pydantic 側に残すが、環境検査は preflight が担う。
- **worker 内変換のみ（集中 preflight なし）** — 失敗が TaskGroup 経由で 1 件ずつしか surface せず集約できない。ExceptionGroup プラミングが残り、HW 非依存テストが worker 起動まで到達を要する。
- **深層まで preflight で完全事前検証（GPU モデルを起動前に実ロード）** — preflight と worker でモデルを二重ロードするか、解決済みリソースを worker へ受け渡す大きな restructure が要る。試すまで分からない失敗は「起動時取得の失敗」として層B で扱えば十分で、preflight は安価な検査に限る。

## Consequences

- パイプラインが起動しない/機能しない理由が、原因 worker と設定値を名指しする明確なログとして一括で出る。複数の設定不備を 1 回の起動で全部把握できる。
- 「起動できた＝その設定で機能する」が前提にでき、機能しない設定値を runtime で凌ぐ到達不能な防御コードを削減できる（ただし per-request の一時エラー処理と inbound 実データ検証は設定防御ではないので残す）。防御コード削減の実収穫は限定的で、preflight 導入が主・削減は副。
- preflight は worker が必要とする設定を worker とは別の場所でも知るため、新しい必須リソースを足すときは 2 箇所（preflight と worker）を更新する保守コストが生じる。デバイス解決は共有 resolver に寄せてこの重複を最小化する。
- [ADR-0019](0019-vc-silero-vad-gate.md) / [ADR-0037](0037-transcription-vad-skip-gate.md) の VAD 起動時 fail-loud は本 ADR の一般化に包含される（機構決定はそのまま有効）。
