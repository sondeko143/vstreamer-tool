# 0055. ストリーミング VC を producer/consumer の role で 2 マシンに分割する

- Status: Accepted
- Date: 2026-07-24
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0050](0050-streaming-vc-separate-subsystem.md), [0051](0051-stream-transport-swappable-tiered.md), [0052](0052-dual-independent-mic-capture.md), [0054](0054-stream-vc-config-section.md), [0045](0045-gui-readiness-reuses-preflight.md)

## Context

M2(単一マシン内ストリーミング)は `capture + vc + playback` の 3 ループを 1 つの内側 TaskGroup で回す([0050](0050-streaming-vc-separate-subsystem.md))。M3 の 2 マシン分割では、録音+VC マシン(=producer)と再生マシン(=consumer)で走らせるループが違う。同じサブシステムがどのループ集合を組むかを、この instance の役割から選べる必要がある。

加えて、マシン間を渡るのは変換音声のみ(spec / [0050](0050-streaming-vc-separate-subsystem.md))なので、再生マシンは変換済み PCM を鳴らすだけで torch/RVC/GPU を一切要らない。「この instance が producer / consumer / それとも単一マシン(local)か」をどう表現し、再生側を重依存無しで動かすかを決める必要がある。

## Decision

`[stream_vc]` に role 列挙 `local | producer | consumer` を足す。

- **`local`(既定)** = M2 と同一。`capture + vc + playback` を 1 プロセスで回し `InProcessTransport` を使う。既定値なので既存挙動は不変。
- **`producer`**(GPU 機 `.149`) = `capture + vc + 網 send`。
- **`consumer`**(再生専任 `.150`) = `網 recv + playback`。**torch/RVC/GPU を import せず動く**(重い import はこのロールでは引かない)。

`transport_type`([0051](0051-stream-transport-swappable-tiered.md))は transport 実装の選択として残す(`udp` を追加)。`role = local` は `transport_type` に関わらず `InProcessTransport` を強制する(topology と transport を分離したまま両者を素直に組み合わせる)。宛先/待受は既存のデバイス設定と同じ flat フィールド:producer は `peer_host`/`peer_port`(送信先)、consumer は `bind_host`/`bind_port`(待受)。必須フィールドの検査は preflight にのみ足し、role ごとに要る項目を見る。GUI の起動前 readiness は preflight を単一の権威として自動追従する([0045](0045-gui-readiness-reuses-preflight.md))。

## Alternatives rejected

- **アドレスの有無から role を推論する**(`peer_host` があれば producer、`bind_host` があれば consumer、無ければ local) — フィールドが少なくて済む反面、両方設定・typo・片方欠落が曖昧で、preflight が「この instance は何をするつもりか」を明快に判定・説明できない。明示 role の方が誤設定を起動時に早く弾ける([0038](0038-worker-config-preflight-fail-loud.md) の fail-loud 方針に沿う)。
- **role と transport を単一 `mode` 列挙に畳む**(`local | udp_producer | udp_consumer`) — 今は 1 ノブで単純だが、topology(誰が capture/vc/playback するか)と transport 実装(UDP か)を結合する。将来 TCP/bidi を足すとき「他ロジックを変えず transport だけ差し替える」([0051](0051-stream-transport-swappable-tiered.md))が崩れ、role×transport の全組合せを 1 列挙に展開する羽目になる。

## Consequences

- role 既定 `local` なので発話系無改変・M2 の byte-identity は維持される([0050](0050-streaming-vc-separate-subsystem.md))。
- consumer が GPU-less で動くため再生機に torch/RVC 依存を置かずに済む。サブシステムの lazy import 境界を、consumer ロールでは `runner.py`/`lib/stream_vc.py`(torch を引く)に触れさせないよう配線する必要がある。
- サブシステムに「role で組むループ集合を変える」分岐が 1 つ増え、preflight は role 別検査になる。
- 単一ストリーミングセッション前提(spec 非ゴール)は据置。複数話者・複数同時セッションは対象外。
- Status は `Accepted`(実装 + 各 task レビュー + 実機 2 マシン run が裏づけ)。
