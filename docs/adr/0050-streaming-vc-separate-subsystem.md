# 0050. ストリーミング VC を Command/routing の外の専用サブシステムとして分離する

- Status: Accepted
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0004](0004-per-destination-sender-transport.md), [0051](0051-stream-transport-swappable-tiered.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0054](0054-stream-vc-config-section.md)

## Context

既存のイベント/ルーティングモデル(`Command`→`WorkerInput`→`WorkerOutput`)は自己完結な離散リクエストで、ストリーム識別子・順序番号・ブロック間状態を持たない。低遅延の連続 VC は、固定ブロックを途切れなく処理し、ブロック間で文脈/クロスフェード状態を保持し、順序・ジッタ・欠落を扱う必要がある。この要件を既存 routing に載せると、発話単位前提の routing/sender/receiver を streaming 都合で歪める。一方、発話単位の字幕/翻訳(録音→文字起こし→翻訳→字幕)は現状の routing のまま維持したい。

## Decision

ストリーミング VC を既存の Command/routing モデルの**外**に置く専用サブシステムとして実装する。発話系は既存 routing のまま無改変で並走させる。両系統の接点はマイク入力のみとする。

## Alternatives rejected

- **既存 routing に載せる(案A: 入力ストリーム worker → vc worker → 既存 unary sender で流用)** — 発話単位の離散モデルに seq/ブロック間状態/ジッタを後付けすることになる。`_send` の例外握り潰し(`sender.py`)による無音穴や単一 sender の順序依存など、streaming に不利な性質をそのまま抱え、低遅延を実測で追い込みにくい。routing を streaming 都合で歪めるコストが、transport/routing 再利用で得る節約を上回る。

## Consequences

発話系のコード/routing を無傷で残せるため「発話単位の字幕/翻訳を維持」という要件に最も安全（これは routing/コードレベルの分離であって、runtime のプロセスレベルの分離ではない — 後述）。streaming 側は連番・バックプレッシャ・ジッタ吸収を自前で設計できる。反面、sender/receiver/routing の再利用は限定的になり新規コードが増える。設定も別系統になる([0054](0054-stream-vc-config-section.md))。マシン間トランスポートは差し替え層として別途決める([0051](0051-stream-transport-swappable-tiered.md))。

サブシステムは Command **routing** の外にあるが、グローバルな **pause/resume ゲート `context.running` は尊重する**([lib/command.py](../../vspeech/lib/command.py) の `pause`/`resume` が発話系と共有で clear/set する Event)。当初は `pause` が字幕/TTS を止めてもストリーミング VC はマイクを出力デバイスへ変換し続けていた。現在は `vc_loop` が `block = await in_queue.get()` の直後・変換の前で 1 点だけ `context.running` を見る:paused の間は消費/変換を止め、capture は回り続けて `drop_oldest_put` が backlog を捨てるので paused 音声は溜まらない。resume 遷移(not-set→set)では実時間が飛んでいるので `StreamingVc._reset_context()` と VAD ゲートをリセットし、最初の post-resume ブロックを pre-pause の尾ではなく無音から fade-in させる。**`reload` は `[stream_vc]` にはまだ効かない**:サブシステムは `sv_config` を起動時に 1 度だけ捕獲し、`reload` イベントで `[stream_vc]` を読み直さない(将来対応)。

**structural な分離(routing/コード)と runtime の障害挙動は別物である。** streaming は Command routing の外にあり発話系のコードを一切触らない(上記)ので、その意味では発話系は無傷で残る。しかしこれは **プロセスレベルの分離ではない**。ストリーミングは opt-in で有効化する機能であり、そこで **回復不能(unrecoverable)な障害** が起きたときは、握り潰さずプロセスごと落とす(fail-loud)のが **意図した挙動** である — daemon/プロセスマネージャ前提の運用で、VC を静かに失って半死したプロセスを生かし続けるより、クリーンに落として再起動させるほうがオペレーション上望ましい、という利用者の判断による。したがって capture/vc/playback ループの回復不能な例外は内側 TaskGroup を abort させ、main の外側 TaskGroup が全 worker(文字起こし・翻訳・字幕・TTS 含む)を cancel して `exit(1)` する — これは是正すべきバグではなく望んだ挙動である。落ちる前にその原因(真の例外)が集約 `BaseExceptionGroup` のノイズに埋もれないよう、`_stream_vc_subsystem` は原因を一度だけ明示ログしてから再送出する(swallow も restart もしない)。

**回復可能(recoverable)な障害だけがサブシステム内で自己回復する。** steady-state の `(OSError, sd.PortAudioError)`(マイク抜け・フォーマット変更・出力先消失)は各ループ内で捕え、close→指数バックオフ(0.5s→5s)→再 open で**サブシステム自身のデバイスだけを**再接続する(発話系 `vspeech/worker/recording.py` の retry パターンを踏襲。共通処理は `vspeech/stream_vc/retry.py`)。vc ループの `process_block` の transient GPU error(CUDA error / OOM = `RuntimeError`)も tear down せず 1 ブロック drop して継続する。**ただし回復可能の範囲を超えたら fail-loud**:連続失敗が閾値を超えた `process_block`、ORT ネイティブ例外(`onnxruntime` の `Fail`/`RuntimeException` 等 — `RuntimeError` 派生ではないので drop されず伝播する)、再 open が成功しないデバイス、そして**起動時のリソース取得失敗**(モデル/デバイスの初回 open は `worker_startup("stream_vc")` 内で行い `WorkerStartupError` として伝播、設定不備を無限 retry で隠さない)は、いずれも上記のとおりプロセスを落とす。cancellation は捕えず必ず propagate する。
