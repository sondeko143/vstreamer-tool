# 0040. 字幕の OBS 出力を新 EventType ではなく subtitle.worker_type のバックエンドとして足す

- Status: Accepted
- Date: 2026-07-16
- Related: spec [2026-07-16-subtitle-obs-websocket-backend-design.md](../superpowers/specs/2026-07-16-subtitle-obs-websocket-backend-design.md); [ADR-0041](0041-subtitle-obs-config-authority.md)（スタイル権威と境界）; [ADR-0042](0042-subtitle-obs-failure-tiers.md)（失敗の階層化）; [ADR-0043](0043-obs-websocket-client-in-house.md)（依存）

## Context

subtitle は vspeech で唯一 tkinter に依存する worker で、ディスプレイの無い環境では有効にできない。OBS へ字幕を出す経路を tk ウィンドウのキャプチャ以外に用意したい。

これを「新しいワーカー」として素直に足そうとすると、契約に触れる。ワーカーは `SharedContext.add_worker` に **`EventType` をキーに**登録されるため、真に別のワーカーには新しい `EventType` が要る。ところが `EventType` は `event_to_operation()` で protobuf の `Operation` と 1:1 対応しており、対応の無い EventType を remote 付きの route に書くと `EventToOperationConvertError` が送出される。これは `Exception` ではなく **`BaseException` 派生**（`exceptions.py`）なので、sender の `except Exception` を素通りして sender タスクごとプロセスを落とす。

`to_pb()` はリモート宛のときしか呼ばれない（`sender.py` の `_dispatch_output`）ため「ローカル専用 EventType」は技術的には成立するが、既存の EventType はすべて `Operation` 対応済みで、ローカル専用イベントという概念はこのコードベースに存在しない。かつ playback で既にクロスホスト運用（`//host:port/...`）をしており、字幕も同じ形で飛ばせる余地を潰したくない。

一方このリポジトリには、同一イベントに複数の実装を持たせる既存の作法がある。transcription（ACP / GCP / WHISPER）と tts（VR2 / VOICEVOX）が `worker_type` でディスパッチしており、CLAUDE.md も「新しいバックエンドはそこに足す」と明記している。

## Decision

`EventType.subtitle` を変えず、`SubtitleConfig.worker_type`（`TK` | `OBS`、既定 **`TK`**）で tts / transcription と同じ形にディスパッチする。OBS 出力は「新しい EventType のワーカー」ではなく「subtitle ワーカーの新しいバックエンド」として実装する。

ヘッドレス目的を達成するため、tkinter の import はバックエンド選択より後になければならない。現状 `worker/subtitle.py` は module レベルで tkinter を import し、`main.py` がそこから `create_subtitle_task` を import するため、そのままではディスパッチャ経由でも tkinter が読まれる。よって以下に分割する:

- `lib/subtitle_state.py` — 純粋な状態機械（`Text` / `Texts` / `ingest_text` / `update_display_sec` / `how_many_should_we_pop`）。tkinter 非依存。両バックエンドが共有する。
- `worker/subtitle.py` — ディスパッチャと `create_subtitle_task` のみ。tkinter を import しない。
- `worker/subtitle_tk.py` — 既存の tk バックエンド（`Tk` / `Canvas` / `Font` / `wrap_text_to_width` / `draw_text_with_outline` / `redraw_panel` / `set_bg_color`）を逐語移動。`Tk()` の生成もタスク生成時からこのバックエンド内へ移す。
- `worker/subtitle_obs.py` — OBS バックエンド。

## Alternatives rejected

- **新 EventType `subtitle_obs`（ローカル専用、protos 無変更）** — `event_to_operation` に対応が無いため、`//obshost:8080/subtitle_obs` と書いた瞬間 `EventToOperationConvertError`（BaseException）が sender を貫通してプロセスが落ちる。playback で実運用中のクロスホスト配線を字幕から奪うことになる。加えて「ローカル専用イベント」という新概念を持ち込む（既存 12 イベントはすべて `Operation` 対応）。得られるのは「tk と OBS を同一プロセスで同時に動かせる」ことだけで、需要が無い。
- **新 EventType + vstreamer-protos に `SUBTITLE_OBS` を追加** — 上記の穴は塞がるが、別リポジトリで .proto を変更 → python/rust/go スタブを各々再生成 → CI の draft prerelease を publish → こちらの wheel URL pin を bump、という多リポジトリ作業になり、Rust bot 等の他コンポーネントにも波及する。得られる差分（tk と OBS の同時稼働）に対して桁違いに重い。
- **`main.py` で `worker_type` を見て分岐する** — `config.subtitle.enable` のゲートに worker_type の分岐を足せば tkinter の遅延 import は達成できるが、transcription / tts が確立した「ワーカー内で `worker_type` にディスパッチする」作法から外れ、バックエンド選択ロジックが起動シーケンスに漏れる。
- **既存 `subtitle.py` に OBS バックエンドを同居させる（分割しない）** — module レベルの tkinter import が残るため、OBS 構成でも tkinter が読まれヘッドレス目的が達成できない。tts が両バックエンドを 1 ファイルに置く前例はあるが、tts のバックエンドは遅延 import 可能な通常ライブラリで、事情が異なる。

## Consequences

- 既定が `TK` なので、既存の設定ファイルは無改変でこれまで通り動く。routes（`sub` / `subtitle`）もクロスホスト配線も一切変わらない。wire 契約は不変で、vstreamer-protos に触れない。
- OBS 構成のプロセスは tkinter を import しない。subtitle が Linux / Docker / サービス実行で有効化できるようになる。
- tk と OBS を同一プロセスで同時に動かすことはできない（1 イベント 1 バックエンド）。必要になったら本 ADR を supersede する。
- 既存 tk コードの機械的な移動が発生し、`test_subtitle_wrap.py` / `test_subtitle_redraw.py` / `test_subtitle_ingest.py` の import 先が変わる。挙動は不変。副次的に、これらのテストは tkinter 非依存になる。
- `wrap_text_to_width` は tk バックエンド専用のまま残る。OBS 側は `extents_wrap` が行分割するため、実測で確認した「折り返し発火時に 1 回の再描画がイベントループを 26〜87ms 止める」崖は OBS 構成には構造的に存在しない。tk 側の崖は本 ADR のスコープ外（非ゴール）。
