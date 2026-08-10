# 0075. ワイヤ形式の sample_rate を範囲 + 25Hz 格子で検証し、外れた datagram を drop する

- Status: Proposed
- Date: 2026-08-11
- Related: [0051](0051-stream-transport-swappable-tiered.md)（このワイヤ形式を定義した ADR）, [0070](0070-device-boundary-inhouse-polyphase-resampler.md)（この検査が必要になった原因）, [0050](0050-streaming-vc-separate-subsystem.md), [0055](0055-stream-vc-producer-consumer-role-split.md), [0056](0056-stream-vc-consumer-jitter-buffer.md), [0062](0062-log-throttle-time-based-episodes.md)

## Context

[0070](0070-device-boundary-inhouse-polyphase-resampler.md) により、ストリーミング VC の出口は `packet.sample_rate` とデバイスレートから有理比ポリフェーズ FIR を**構築する**ようになった。この `sample_rate` は認証のない LAN の UDP ソケットから届く unsigned 32bit で、`decode_packet` は magic と version しか検査していない（[0051](0051-stream-transport-swappable-tiered.md) が定めたヘッダ）。

0070 の前は、この値は `sd.RawOutputStream(samplerate=...)` に渡るだけだった。異常値は `PortAudioError` になり、デバイス障害の backoff 経路が拾って**次の正常パケットで自己回復**していた。0070 の後は同じ値がリサンプラ構築に直行する。実測（この機体、`vspeech/lib/resample.py` の算術をそのまま辿ったもの）:

| ヘッダが運びうる値 | 結果 |
|---|---|
| `0` | `ValueError: rates must be positive`。デバイス例外ではないので `(OSError, PortAudioError)` の retry 層を素通りし、**subsystem ごとプロセスが落ちる** |
| `2**32-1` | up=3200 / down=286331153 → **2.9e10 タップ = float64 で 233GB**。落ちるより悪い（OOM / wedge） |
| `44101`（範囲内・デバイスと互いに素） | up=48000 → 4.8M タップ、**peak 563MB / 構築 1.4s**、以後 1 ブロック 27ms |
| `191999`（範囲内・同上） | up=192000 → 19.6M タップ、**peak 2.3GB / 構築 6.5s** |

ここで効いている性質が 1 つある。**構築コストを決めるのはレートの大きさではなく `device_rate // gcd(rate, device_rate)`（位相数）である。** パケットレートがデバイスレートと互いに素なら、そのレートが 1 でも 44101 でも位相数はデバイスレートそのものになり、フィルタは同じだけ巨大になる。つまり**範囲だけを検査しても病理は残る**。

一方、受信側には既に「壊れた datagram は `WireError` にして drop + `stream_vc_malformed_drop` + 間引き warning」という設計された経路がある（`udp.py` の `_RecvProtocol.datagram_received`、[0062](0062-log-throttle-time-based-episodes.md)）。0070 は結果的にこの経路を迂回する新しい致命経路を作ってしまっていた。

## Decision

`sample_rate` を **`encode_packet` と `decode_packet` の両方**で検証し、規則を外れた値は `WireError` にする。

- **範囲 8000〜192000 Hz。** 標準オーディオレート族の外縁（電話帯域〜ハイレゾ上限）。この経路が実際に運ぶ RVC モデルのレート（32000 / 40000 / 48000）はその内側にある。
- **25Hz の倍数であること。** 標準レート族（8000, 11025, 16000, 22050, 24000, 32000, 40000, 44100, 48000, 88200, 96000, 176400, 192000）の **gcd がちょうど 25** で、これが全部を残せる最大の刻みである（11025 があるので 50 は取れない）。実在するデバイスレートも同じ格子上にあるので、格子を要求すると `gcd >= 25` が保証され、位相数が `device_rate/25` 以下に抑えられる。実測で最悪ケースは同じ範囲内で **785k タップ / peak 91MB / 構築 0.24s** まで落ちる。
- decode 側の `WireError` は既存の malformed-drop 経路に乗る。**1 個の壊れた datagram で consumer は落ちも wedge もせず、drop されて計上され、間引き warning に 1 行出る。**
- encode 側も同じ規則で弾く。自分のデコーダが拒否するものを送出しない。規則外のレートのモデルを積んだ producer は最初のブロックで fail loud する（遠隔で全 drop され続けて無音になるより、送信側で即座に理由が出るほうが診断できる）。

例外メッセージは日本語（[0064](0064-code-comments-in-english.md)）。受信側の warning に `%r` で載って、操作者がなぜ無音なのかを知る唯一の手掛かりになるため。

## Alternatives rejected

- **範囲だけを検査する（格子を要求しない）** — 上表のとおり `up` は大きさではなく gcd で決まるので、範囲内の互いに素なレート 1 個で 563MB / 1.4s、上限付近なら 2.3GB / 6.5s を作れてしまう。再生専任機（[0055](0055-stream-vc-producer-consumer-role-split.md) の consumer は小型機を想定）では OOM になる。範囲は `2**32-1` のような極端値しか止められない。
- **標準レートのホワイトリスト** — さらに強いが、32k/40k/48k 以外で学習された将来の RVC モデルを理由なく拒否する。格子は「実在しうるレートは通す / 病的な比だけ落とす」に近く、拒否の理由も「25 の倍数でない」と説明できる。
- **`PolyphaseResampler` 側で `up` に上限を設けて拒否する** — 層としてはこちらが本質的で、入口側の同じ病理（デバイスが 44099 を報告する場合、[0071](0071-device-native-rate-resolution.md) が実測で警告している）にも効く。だが「信頼できない入力は境界で落とす」の原則からは wire 層が先であり、リサンプラ側の上限は本 ADR と両立する（将来足してよい）。
- **何もせず [0050](0050-streaming-vc-separate-subsystem.md) の fail loud に委ねる** — 0050 の fail loud は「明示的に有効化した機能の**回復不能な**障害」、すなわち設定・機器・モデルの問題を指している。ネットワークから来た 1 個の壊れたバイト列はそれに当たらず、受信側は既に drop 経路を持っている。1 datagram でプロセスが落ちるのは [0056](0056-stream-vc-consumer-jitter-buffer.md) の「損失は穴埋めして観測する」という consumer の設計とも逆行する。
- **`sample_rate` をヘッダから外し、セッション確立時に 1 度だけ交渉する** — 検証すべき面は減るが、UDP は接続を持たないので新しいハンドシェイク層が要る。[0051](0051-stream-transport-swappable-tiered.md) の「1 datagram = 1 ブロックで自己完結」を壊す変更で、得るものに対して大きすぎる。

## Consequences

- ワイヤ形式の値域が狭まる。**契約の縮小**なので後方非互換になりうるが、今日の producer が送る値（RVC モデルの `samplingRate` = 32000/40000/48000）は全て通るため実運用の互換性は保たれる。
- NTSC プルダウン（47952 / 44056）のような非標準レートのモデルは UDP 経路で使えなくなる。`role=local` は wire を通らないので同じ構成でも動く — この非対称は意図的で、信頼できない入力があるのは network 経路だけである。
- 検証が両側にあるので、送信側の構成ミスは自分のマシンで、伝送の破損は受信側で、それぞれ最も診断しやすい場所に出る。
- 数値（8000 / 192000 / 25）は実測とレート族の gcd に基づくが、**デバイスレート自体が 25 の倍数でない環境**（プルダウンレートのプロ機材）では格子による位相数の保証が効かない。その場合も範囲の保証と `2**32-1` の遮断は残る。上限が必要になったらリサンプラ側（却下案 3）で受ける。
- `wire.py` が純粋なコーデックではなくなり、1 フィールドだけ値の意味を知るようになった。他のフィールドに同じ検査を広げたくなったら、その都度この ADR の理由（受信側が値をそのまま資源確保に使うか）で判断する。
