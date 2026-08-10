# 0070. ストリーミング VC のデバイス latency を入出力別の設定値にする

- Status: Proposed
- Date: 2026-08-10
- Related: [spec](../superpowers/specs/2026-08-10-stream-vc-latency-config-design.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0054](0054-stream-vc-config-section.md), [0055](0055-stream-vc-producer-consumer-role-split.md), [0068](0068-config-load-errors-through-preflight-report.md)

## Context

`stream_vc` が PortAudio に要求する latency は、入力(マイク capture)・出力(playback / consumer)とも `"low"` にハードコードされている。

`"low"` は絶対値ではなくデバイス側が申告する既定値で、ホスト API によって桁が変わる(Windows の MME と WASAPI で顕著)。低すぎるデバイスでは overflow / underflow が定常的に出て、間引かれた警告と telemetry だけが積み上がる。高すぎるデバイスでは、`block_ms` を詰めても取り返せない遅延が乗る。どちらもコードを書き換える以外に手当てできない。

これは他の遅延パラメータと同じ予算の一項目である。ストリーミング VC の遅延は「固定 per-inference floor(実測 ~40ms/fcpe) + `block_ms` + デバイス latency」で決まり([0053](0053-streaming-vc-fixed-block-crossfade.md))、前二者は設定で詰められるのに最後だけが露出していない。

さらに入力と出力は別デバイスであり、role を分ければ**別マシン**にある([0055](0055-stream-vc-producer-consumer-role-split.md))。producer 機のマイクと consumer 機のスピーカーで事情が食い違うのが普通の状態で、片側だけ緩めるという当たり前の調整ができない。

## Decision

`[stream_vc]` に `input_latency` と `output_latency` を追加し、それぞれ capture の `RawInputStream` と playback / consumer が共有する `RawOutputStream` へ渡す。

型は `Literal["low", "high"] | float`(float は `gt=0`)、既定は両方 `"low"`。値は単位変換せず sounddevice へ素通しする — float は**秒**、PortAudio の `suggestedLatency` の単位そのままである。

あわせて、ストリームを開いたログに「要求値」と「PortAudio が実際に返した値(`stream.latency`)」の両方を出す。

## Alternatives rejected

- **単一 `latency` フィールドを入出力で共有する** — フィールドは 1 本で済むが、入力と出力は別デバイス(role 分割では別マシン)であり、片側の underflow を直すために両側を高遅延にする羽目になる。この設定を足す動機そのものを潰す。
- **ms 単位の float(`input_latency_ms` 等)にする** — `block_ms` / `context_ms` など既存項目と単位は揃うが、「デバイス既定の high」を表現できず、PortAudio が持つ 2 つの既定値の片方を捨てることになる。加えて ms→秒 の変換を挟むぶん、素通しなら存在しない単位取り違えのバグ面が増える。単位不統一は `_ms` を付けない命名で示す。
- **`"low"` / `"high"` の 2 択に限る** — 設定ミスの余地は最小になるが、「low だと underflow、high だと遅すぎる」中間を狙えない。PortAudio 自体が任意の秒数を受けるので、ここで 2 択に狭める理由がない。
- **ハードコードのまま、必要になったらコードを書き換える** — 現状。デバイスを替えるたびにコード変更とデプロイが要り、[0054](0054-stream-vc-config-section.md) が `[stream_vc]` を独立させた理由(マシンごとに独立して調整する)に反する。
- **preflight に専用の検査を足す** — 不正値は pydantic の型で config ロード時に落ち、[0068](0068-config-load-errors-through-preflight-report.md) によって preflight と同じ per-problem レポートで報告される。preflight 側に同じ判定を書くと二重管理になる。
- **発話系 `[recording]` / `[playback]` にも同時に足す** — こちらは latency を要求しておらず PortAudio 既定に任せている。同じ項目を足すかは別の判断であり、ストリーミング側の必要に引きずって決めない。

## Consequences

デバイス起因の overflow / underflow をコード修正なしに調整でき、producer / consumer で片側ずつ詰められる。既定が `"low"` = 現在のハードコード値なので、既存 config の挙動は変わらない。

実際に得られた latency をログに出すため、`block_ms` を詰める前に「そのデバイスで `"low"` が何秒だったか」が読める。ただし PortAudio は要求値を保証しない — 明示秒数を書いてもデバイスがそれを上回る / 下回る値を返しうる。ログはその差を見えるようにするためのもので、要求が通った証拠ではない。

`[stream_vc]` に秒単位のフィールドが 2 本混ざる。`_ms` を付けない命名で単位の取り違えは防ぐが、このセクションを読む人にとって単位が 2 種類になるのは事実として残る。

発話系 `[recording]` / `[playback]` は据え置きなので、「ストリーミング側だけ latency を設定できる」非対称が残る。発話系で同じ問題が出たら、そのときに改めて判断する。
