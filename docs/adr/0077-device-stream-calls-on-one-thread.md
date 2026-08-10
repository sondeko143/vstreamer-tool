# 0077. デバイスストリームの blocking 呼び出しと close を 1 本のスレッドに直列化する

- Status: Accepted
- Date: 2026-08-11
- Related: [0031](0031-audio-pyaudio-to-sounddevice.md)（sounddevice への移行 = この blocking API を採用した ADR）, [0038](0038-worker-config-preflight-fail-loud.md)（設定不備は `exit 1` で落ちるという約束）, [0050](0050-stream-vc-device-retry.md)（同じ close パターンを持つ再接続ループ）, [0074](0074-device-native-rate-resolution.md), [0076](0076-preflight-device-rate-open-probe.md)

## Context

録音ワーカー (`vspeech/worker/recording.py`) は、blocking な `stream.read()` をイベントループから外すため `await to_thread(stream.read, n)` で読み、ループを抜けたら `finally: stream.close()` で閉じる。この形は sounddevice へ移行した [0031](0031-audio-pyaudio-to-sounddevice.md) 以来のもので、`stream_vc` の capture/playback と発話系 playback も同じ形をしている。

問題は **キャンセルされたときの重なり**である。`await to_thread(...)` はタスクがキャンセルされた瞬間に制御を返すが、`to_thread` に渡したスレッドは止まらない — `concurrent.futures` は走り出したジョブをキャンセルできないので、スレッドは `Pa_ReadStream` の中に居続ける。その直後に `finally` が `stream.close()` を呼ぶと、`Pa_CloseStream` がストリーム本体とホストバッファを解放し、**読み手がその中にいるまま解放される**。PortAudio はこの 2 つを一切同期しない。読み手は解放済みメモリを辿り、Windows はそれをアクセス違反 (`0xC0000005`、終了コード `-1073741819`) として報告する — Python の例外ではないので捕捉できず、プロセスがその場で死ぬ。

この経路は例外的な状況ではなく、**起動失敗のたびに通る**。gRPC ポートが既に埋まっている等で `receiver` が `RuntimeError` を投げると、TaskGroup が録音ワーカーを read の最中にキャンセルするからである。つまり「設定不備は `exit 1` で落とす」という [0038](0038-worker-config-preflight-fail-loud.md) の約束が、条件によってアクセス違反にすり替わっていた。

実機で測った発火率（同一 config を 2 プロセス起動し、2 本目のポート衝突で起動を失敗させる。VAC Line 2 / MME）:

| コード | 試行 | アクセス違反 |
| --- | --- | --- |
| `main` (この branch の前) | 70 | 1 |
| この branch (HEAD) | 30 | 2 |

**この branch が持ち込んだ回帰ではない**（`main` でも同じスタックで落ちる: `sounddevice.py` の `_raw_read` にいる `asyncio_0` スレッドでのアクセス違反）。解放されたブロックが再利用されるまでに読み手が触るかどうかで決まる競合なので、発火は確率的で、1 回ずつの比較では「branch だけが落ちる」ように見える。branch 側の発火率が高く見えるのは、デバイスをネイティブレート (48000Hz) で開くようになった分ホストバッファが 3 倍になり、解放ブロックが再利用されやすいためと考えられるが、この試行数では有意差とは言えない。

## Decision

**1 つのストリームに対する blocking なネイティブ呼び出しと `close()` を、そのストリーム専用の 1 スレッドに直列化する。** `vspeech/lib/audio.py` に `DeviceStreamThread`（`max_workers=1` の `ThreadPoolExecutor` を 1 本持つ）を置き、録音ワーカーは `to_thread` の代わりにこれを通して読む。`close()` は次のように振る舞う:

- **呼び出しが実行中なら**、`close` を同じ executor に **queue する**。1 スレッドなので、走っている `read` が返るまで `close` は始まりようがない。待つのはそのスレッドであり、イベントループでも呼び出し側でもない。
- **実行中でなければ**（初回の read 前 / read が例外で終わった直後 / `aclose()` で抜けた直後）、**その場で同期的に閉じる**。デバイス障害での再オープンや reload の経路はこれまでどおり「戻ってきた時点で閉じ終わっている」ままになる。

「実行中か」は `submit()` が返す `concurrent.futures.Future` の `done()` で判定する。`done()` が True になるのはワーカースレッドが `fn` から戻った後なので、True ならネイティブ呼び出しの外にいることが保証される。判定直後に True へ変わった場合は queue 側に倒れるだけで、どちらに転んでも重ならない。

プロセス終了時も取りこぼさない: `shutdown(wait=False)` は queue 済みの作業をキャンセルしないし、インタプリタ終了時の `concurrent.futures.thread._python_exit` は各スレッドに番兵 `None` を **queue の末尾に** 積んでから join するので、その手前に積んだ `close` は必ず実行されてからスレッドが終わる。

## Alternatives rejected

- **`finally` で実行中の read を待ってから閉じる**（`asyncio.wrap_future` した concurrent future を `finally` で await し直す） — イベントループを止めずに済むが、キャンセル済みタスクの `finally` での await はキャンセルの届き方に依存する。2 度目のキャンセル（TaskGroup の再アボート、ループの終了）が来ればその await は即座に飛ばされ、`close()` だけが残ってクラッシュが戻る。直列化はキャンセルの意味論に一切依存しない。
- **`threading.Lock` で read と close を排他する** — 正しいが、ロックを取るのがイベントループのスレッドになる。デバイスが死んで `Pa_ReadStream` が返ってこない場合、イベントループごと止まる（今回のクラッシュは「デバイスが他プロセスに握られている」状況で起きているので、まさにその状況で最悪化する）。
- **`close()` の前に `stop()` / `abort()` を呼ぶ** — `Pa_StopStream` / `Pa_AbortStream` も、読み手が中にいるストリームに対する非同期なネイティブ呼び出しであることに変わりはない。PortAudio の blocking I/O はこれらとの同時呼び出しを保証していない。
- **`close()` を諦めてハンドルを漏らす / `os._exit()` で終了処理ごと飛ばす / faulthandler を切る** — どれも症状を隠すだけで、[0076](0076-preflight-device-rate-open-probe.md) の Consequences で塞いだばかりのハンドルリークを作り直すか、クラッシュを見えなくするだけである。
- **`lib/audio.open_device_stream` が返すストリームをラッパで包み、4 境界すべてを一度に安全にする** — 最も網羅的だが、`open_device_stream` は呼び出し側が作ったストリーム型 `StreamT` をそのまま返すジェネリックであり、4 箇所の `sd.RawInputStream` / `sd.RawOutputStream` という注釈とそれに紐づくテストを全部書き換えることになる。今回再現・検証できたのは録音経路だけなので、検証できない 3 箇所を巻き込んで作り替えるのは割に合わない。
- **何もしない（`main` から在る既知の競合として放置する）** — 発火率は低いが、発火するのは「設定不備で起動に失敗したとき」= 一番出やすい失敗であり、`exit 1` と明確なエラーメッセージで終わるはずのところがアクセス違反になる。[0038](0038-worker-config-preflight-fail-loud.md) の約束と正面から矛盾する。

## Consequences

- 録音の read は共有のデフォルト executor (`asyncio.to_thread`) を使わなくなり、ストリームを開くたびに専用スレッドが 1 本立って `close` とともに退役する。ストリームの再オープン（デバイス障害からの復帰）ごとに 1 本作り直す。
- キャンセル直後の `close` は最大で 1 ブロック分（既定 `chunk=1024` / 16000Hz で 64ms、実運用の `chunk=1600` で 100ms）遅れて実行される。プロセス終了時はその分だけ終了が遅くなるが、`_python_exit` の join がそれを待つので終了コードは意図どおり `exit 1` になる。
- **同じパターンが `stream_vc/capture.py`（read）、`stream_vc/playback.py`（write）、`worker/playback.py`（write）にも残っている。** いずれも `main` から在るもので、今回のセッションでは実機検証できないため、各 close 地点にこの ADR を指すコメントを残して次に触る人へ渡す（このリポジトリの「先送りした指摘はコードのその場に書く」規約に従う）。録音経路と違い、これらは steady state のデバイス障害でも `close` を呼ぶが、その経路では呼び出しが例外で終わっているので重ならない — 危ないのはキャンセル（購読終了・Ctrl-C からの TaskGroup 巻き戻し）だけである。
- 通常の停止経路（`vsctl pause` / `reload`、正常な発話の切れ目）は元から安全だった: そこでは非同期ジェネレータが `yield` で中断しており、read は既に返っている。危ないのは「read の最中にキャンセルが届く」場合だけである。
