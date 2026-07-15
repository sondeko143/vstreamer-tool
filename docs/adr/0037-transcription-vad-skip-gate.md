# 0037. Silero VAD スキップゲートを transcription パスへ拡張する（独立ワーカーを却下）

- Status: Proposed
- Date: 2026-07-16
- Related: spec [2026-07-16-transcription-vad-skip-gate-design.md](../superpowers/specs/2026-07-16-transcription-vad-skip-gate-design.md); [ADR-0019](0019-vc-silero-vad-gate.md)（VC ゲート、本 ADR がスコープ拡張）; [ADR-0020](0020-silero-vad-v6.md)（モデル pin）; [ADR-0024](0024-onnx-session-single-factory.md)（VAD の CPU 固定例外）; [ADR-0036](0036-whisper-resample-via-pyav.md)（再利用する PyAV リサンプル経路）

## Context

[ADR-0019](0019-vc-silero-vad-gate.md) は Silero VAD ゲートを VC パスに限定した。理由の一つは、同じ録音チャンクが `routes_list` で transcription にも並列に流れ、recording 側でゲートを強化すると transcription が残したいフィラー・感嘆詞まで落ちるため、対策を VC 側へ寄せたことにある。

しかし transcription 自身も無音・ノイズ単独チャンクを受け取る。recording の門番は発話単位の dBFS 閾値なので、環境音・ブレス・発話後の余韻でトリガーされたチャンクがそのまま音声認識に届く。無音・ノイズ入力に対して whisper は実在しないテキスト（カタカナ等）を幻聴し、無駄な GPU 推論も走る。transcription パス「側」に recording とは独立の絶対判定ゲートを opt-in で置けば、フィラーを落とさずに無音チャンクだけを弾ける——これは 0019 が recording 側強化を避けた理由と整合する。

当初は VAD を独立のルーティング可能ワーカーとして切り出す案を主軸に検討したが、実測でコストに見合わないと判明したため方針を変えた。

## Decision

Silero VAD の絶対判定ゲートを、**独立ワーカーではなく transcription ワーカー内のスキップ判定**として追加し、0019 のゲートを **VC + transcription の両パス**へ拡張する（opt-in、既定 OFF）。

- transcription は音声を出力しないため **スキップ判定のみ**（音声比率が閾値未満のチャンクを認識前に落とす）。VC 側の出力ダック（窓ごとゲインマスク）は transcription に持ち込まない。
- 入力→16kHz 変換は transcription 既存の PyAV 経路（[ADR-0036]）を再利用し、torch を持ち込まない。
- ACP / GCP / WHISPER 全バックエンドで効かせる。
- 設定は **各ワーカー独立**（vc と transcription が別々に model/threshold 等を持つ）。vc は無変更。
- エラー非対称（0019 踏襲）: モデル不在・ロード失敗は起動時 fail loudly、推論中の例外はそのチャンクをゲートせず通す。
- VAD セッションは CPUExecutionProvider 固定（0019 / [ADR-0024] の意図的例外）。

## Alternatives rejected

- **VAD を独立のルーティング可能ワーカーに切り出す（+ wire 契約に VAD operation 追加）** — 主動機は「vc と transcription で同一感度・二重計算回避」。だが実測（v6.2.1, CPU）で二重計算の単位＝`speech_probs` 1 回は約 8.5–9 ms/秒（0.5s→4.3ms, 5s→44ms）、numpy 側の判定・マスク生成は約 0.02ms で無視可能。二経路は別スレッド（`to_thread`）で並列に走り推論は GIL を解放するため、クリティカルパスに遅延を足さず、増えるのは数 ms の CPU 総量のみ（e2e 約 3s の 1% 未満）。さらに出力ダックは RVC 出力専用で vc に残るため、ワーカーが skip 判定だけ転送しても vc は結局 probs を再計算する。二重計算を真に消すには窓ごとゲイン配列を protobuf 経由で渡す新フィールドまで要る。得られる削減 << 新 EventType + wire 拡張 + 両経路にワーカー 1 ホップ、で不成立。
- **共有トップレベル `[vad]` config セクション（感度を 1 箇所で管理）** — vc の shipped 設定（`vc.vad_*`）を破壊的に移行する必要があり、稼働中の config・0019/0020・example・test を更新するコストに対し、config は稀にしか編集しないため利得が小さい。各ワーカー独立フィールドなら移行ゼロで、将来 vc と transcription を別感度にする余地も残る。
- **recording 側 `silence_threshold` の強化で両パスまとめて弾く** — 0019 と同じ理由で却下。transcription が残したいフィラー・感嘆詞まで落ちる。
- **transcription 出力側でのダック/ゲイン** — transcription は音声を出力しないため無意味。

## Consequences

- 無音・ノイズチャンクに対する whisper の幻聴と無駄な GPU 推論を、フィラーを保ったまま opt-in で抑えられる。
- VAD 判定ロジックは `lib/vad.py` の共有ライブラリのままで、vc と transcription が各々呼ぶ（ワーカーは 1 つも増えない）。同じチャンクに対し VAD が二重計算されうるが、並列実行で無害（実測 数〜数十 ms の CPU）。
- 感度設定が vc と transcription の 2 箇所に分かれるため、「同一感度」にしたい場合は手動で揃える必要がある（ドキュメントで明示）。
- 0019 の「VC パスのみ」スコープは本 ADR で緩和されるが、0019 の機構決定（CPU 固定・入力ゼロ埋め回避・エラー非対称・opt-in）はそのまま有効。
