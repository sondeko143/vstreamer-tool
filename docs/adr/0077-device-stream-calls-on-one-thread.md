# 0077. デバイスストリームの blocking 呼び出しと close を 1 本のスレッドに直列化する

- Status: Accepted
- Date: 2026-08-11
- Related: [0031](0031-audio-pyaudio-to-sounddevice.md)（sounddevice への移行 = この blocking API を採用した ADR）, [0038](0038-worker-config-preflight-fail-loud.md)（設定不備は `exit 1` で落ちるという約束）, [0050](0050-streaming-vc-separate-subsystem.md)（デバイス障害のたびに close を呼ぶ再接続ループ）, [0052](0052-dual-independent-mic-capture.md), [0074](0074-device-native-rate-resolution.md), [0076](0076-preflight-device-rate-open-probe.md)（同じくストリームを閉じ損ねる穴を塞いだ ADR）

## Context

デバイス境界は 4 つある（発話系の録音 / 発話系の再生 / streaming VC の入力 / streaming VC の出力）。sounddevice へ移行した [0031](0031-audio-pyaudio-to-sounddevice.md) 以来、4 つとも同じ形をしている: blocking な `stream.read()` / `stream.write()` を `await to_thread(...)` でイベントループの外に出し、ループを抜けたら `finally` で `close()` する。

問題は **キャンセルされたときの重なり**である。`await to_thread(...)` はタスクがキャンセルされた瞬間に制御を返すが、`concurrent.futures` は走り出したジョブをキャンセルできないので、スレッドは `Pa_ReadStream` / `Pa_WriteStream` の中に居続ける。その直後に `close()` が呼ばれると、`Pa_CloseStream` がストリーム本体とホストバッファを解放し、**呼び出し側がその中にいるまま解放される**。PortAudio はこの 2 つを一切同期しない。読み手/書き手は解放済みメモリを辿り、Windows はそれをアクセス違反 (`0xC0000005`、終了コード `-1073741819`) として報告する — Python の例外ではないので捕捉できず、プロセスがその場で死ぬ。

発火は確率的である。解放されたブロックが、読み手が触る前に再利用されたかどうかで決まる（採取したクラッシュの 1 つは、メインスレッドが ExceptionGroup の巨大なトレースバックを整形している最中 = 大量のヒープ確保の最中に起きている）。実機で測った発火率（同一 config を 2 プロセス起動し、2 本目のポート衝突で起動を失敗させる。VAC Line 2 / MME）:

| コード | 試行 | アクセス違反 |
| --- | --- | --- |
| `main`（この branch の前） | 70 | 1 |
| この branch（修正前） | 70 | 4 |

**この branch が持ち込んだ回帰ではない**（`main` でも同じスタックで落ちる: `sounddevice.py` の `_raw_read` にいる `asyncio_0` スレッドでのアクセス違反）。1 回ずつの比較では「branch だけが落ちる」ように見えるが、そうではない。branch 側が高く見えるのは、デバイスをネイティブレート (48000Hz) で開くようになった分ホストバッファが 3 倍になり、解放ブロックが再利用されやすくなったためと考えられるが、この試行数では有意差とは言えない。

引き金の出やすさは境界によって違う:

- **発話系の録音**: 引き金は「起動失敗」= gRPC ポートの衝突などで `receiver` が落ち、TaskGroup が録音ワーカーを read の最中にキャンセルする経路。つまり **[0038](0038-worker-config-preflight-fail-loud.md) が約束している「設定不備は `exit 1` で落とす」経路が、条件次第でアクセス違反にすり替わる**。
- **streaming VC の入力/出力**: `close_quietly` は **デバイス障害からの再オープンのたび**に走る。[0050](0050-streaming-vc-separate-subsystem.md) はデバイス障害が現場で起きる前提で書かれているので、露出はこちらの方が大きい。
- **発話系の再生**: 1 発話が 1 回の write なので、重なりうる窓は再生中の音声の長さそのもの。

## Decision

**1 つのストリームに対する blocking なネイティブ呼び出しと `close()` を、そのストリーム専用の 1 スレッドに直列化する。** `vspeech/lib/audio.py` に `DeviceStreamThread`（`max_workers=1` の `ThreadPoolExecutor` を 1 本持つ）を置き、**4 境界すべて**がこれを通して読み書きする。`close()` は次のように振る舞う:

- **呼び出しが実行中なら**、close を同じ executor に **queue する**。1 スレッドなので、走っている read/write が返るまで close は始まりようがない。待つのはそのスレッドであり、イベントループでも呼び出し側でもない。
- **実行中でなければ**（初回の呼び出し前 / 呼び出しが例外で終わった直後 / `aclose()` で抜けた直後 / 発話の切れ目）、**その場で同期的に閉じる**。デバイス障害での再オープン・reload・発話ごとの開き直しはこれまでどおり「戻ってきた時点で閉じ終わっている」ままで、close が投げる例外も従来どおり呼び出し側に届く（`close_quietly` が握りつぶせる）。

「実行中か」は `submit()` が返す `concurrent.futures.Future` の `done()` で判定する。`done()` が True になるのはワーカースレッドが `fn` から戻った後なので、True ならネイティブ呼び出しの外にいることが保証される。判定直後に True へ変わった場合は queue 側に倒れるだけで、どちらに転んでも重ならない。**判定は未完了の呼び出し全部に対して行い、最新の 1 つだけを見ることはしない** — 呼び出し側が 2 つ居ると、後発の呼び出しは queue に積まれたまま await がキャンセルされうる。queue 済み未実行の future の `cancel()` は成功する（`_WorkItem.run` の `set_running_or_notify_cancel` が False を返して `fn` を呼ばない）ので `done()` は True になり、「最新が done なら閉じてよい」だと**先発の呼び出しが PortAudio の中に居るままインライン close する** = 直そうとしている use-after-free そのものになる。現時点では 4 境界とも 1 タスクからしか駆動していないが、それはコード上強制されていないので、判定をその前提に乗せない。

インライン close は例外を投げうる（壊れた/既に閉じたデバイスの close が `DEVICE_ERRORS` を投げるのは、まさに `close_quietly` が存在する理由 = streaming VC の再接続ループの live path）。その例外は従来どおり呼び出し側へ伝播させるが、executor の `shutdown(wait=False)` は `finally` で必ず実行する — でないとワーカースレッドが queue で待ったまま残り、このオブジェクトが GC されるまで畳まれない。決定的な後始末を目的にした仕組みがスレッドの寿命を refcount に委ねるのは筋が通らない。

`close()` はストリームそのものではなく **閉じ方（`Callable[[], None]`）を受け取る**。境界ごとに閉じ方が違う（素の `stream.close()` か、`close_quietly` 越しか）ためで、渡すのはそのデバイスを閉じるだけの呼び出しに限る（実行中の場合それは呼び出し側ではなく所有スレッドで走る）。

**保持のしかたは境界ごとに、ストリームを所有するオブジェクトへ持たせる**: 録音ワーカーはジェネレータのローカルとして直接、streaming VC 入力は新設の `InputTap`（stream + 開いたレート + スレッド）、streaming VC 出力は既存の `OutputSink`、発話系の再生は既存の `OutputStream`。3 つとも `close()` を持つオブジェクトなので、`stream_vc/retry.py`（`run_with_device_retry` / `close_quietly`）は **docstring 以外は無変更** — `_Closable` 境界のまま、閉じられる対象が「生のストリーム」から「そのストリームを所有する物」に変わっただけで、`run` のシグネチャも呼び出し側も retry のテストも動いていない（`close_quietly` の docstring には、渡されるのが所有オブジェクトであることと、遅延した close の例外はもうここでは捕まらないことを追記した）。`InputTap` は同時に、開いたレートを capture_loop の `nonlocal` closure で読み出し口へ運んでいた仕掛けも不要にした。

プロセス終了時も取りこぼさない: `shutdown(wait=False)` は queue 済みの作業をキャンセルしないし、インタプリタ終了時の `concurrent.futures.thread._python_exit` は各スレッドに番兵 `None` を **queue の末尾に** 積んでから join するので、その手前に積んだ `close` は必ず実行されてからスレッドが終わる。

## Alternatives rejected

- **`finally` で実行中の呼び出しを待ってから閉じる**（`asyncio.wrap_future` した concurrent future を `finally` で await し直す） — イベントループを止めずに済むが、キャンセル済みタスクの `finally` での await はキャンセルの届き方に依存する。2 度目のキャンセル（TaskGroup の再アボート、ループの終了）が来ればその await は即座に飛ばされ、`close()` だけが残ってクラッシュが戻る。直列化はキャンセルの意味論に一切依存しない。
- **`threading.Lock` で呼び出しと close を排他する** — 正しいが、ロックを取るのがイベントループのスレッドになる。デバイスが死んで `Pa_ReadStream` が返ってこない場合、イベントループごと止まる（今回のクラッシュは「デバイスが他プロセスに握られている」状況で起きているので、まさにその状況で最悪化する）。
- **`close()` の前に `stop()` / `abort()` を呼ぶ** — `Pa_StopStream` / `Pa_AbortStream` も、呼び出し側が中にいるストリームに対する非同期なネイティブ呼び出しであることに変わりはない。PortAudio の blocking I/O はこれらとの同時呼び出しを保証していない。
- **`close()` を諦めてハンドルを漏らす / `os._exit()` で終了処理ごと飛ばす / faulthandler を切る** — どれも症状を隠すだけで、[0076](0076-preflight-device-rate-open-probe.md) の Consequences で塞いだばかりのハンドルリークを作り直すか、クラッシュを見えなくするだけである。
- **`lib/audio.open_device_stream` が返すストリームをラッパで包み、4 境界を暗黙に安全にする** — 呼び出し側の変更が要らない代わりに、`open_device_stream` は呼び出し側が作ったストリーム型 `StreamT` をそのまま返すジェネリックなので、4 箇所の `sd.RawInputStream` / `sd.RawOutputStream` という注釈とそれに紐づくテストを全部書き換えることになる。所有オブジェクト側に持たせる形なら、境界ごとの「閉じ方」の違いもそのまま残せる。
- **`run_with_device_retry` にスレッドを持たせ、`run` コールバックへ渡す**（streaming VC 用の別案） — `run` のシグネチャと 2 つの呼び出し側・retry のテストが動く。`InputTap` / `OutputSink` に持たせれば retry.py は無変更で済み、しかも入力側と出力側の形が揃う。
- **録音経路だけ直し、残り 3 箇所は `[Open, deferred]` コメントで残す**（最初の実装） — 同じ欠陥クラスに一貫性が無く、しかも露出が大きいのは残す方（`stream_vc` は再接続のたびに close する）だった。仕組みとテストが既にある以上、4 境界すべてに適用するのが筋である。
- **何もしない（`main` から在る既知の競合として放置する）** — 発火率は低いが、発火するのは「設定不備で起動に失敗したとき」や「デバイス障害から復帰するとき」= 一番出やすい失敗であり、`exit 1` と明確なエラーメッセージで終わるはずのところがアクセス違反になる。[0038](0038-worker-config-preflight-fail-loud.md) / [0050](0050-streaming-vc-separate-subsystem.md) の狙いと正面から矛盾する。

## Consequences

- 4 境界とも共有のデフォルト executor (`asyncio.to_thread`) を使わなくなり、ストリームを開くたびに専用スレッドが 1 本立って `close` とともに退役する。再オープン（デバイス障害からの復帰、発話ごとのフォーマット変更）ごとに 1 本作り直す。
- **リアルタイム経路の遅延は増えない。** 1 回の呼び出しのコストは executor への submit 1 回とウェイクアップ 1 回で、`asyncio.to_thread` と同じ（実測: no-op の往復が p50 0.164ms → 0.142ms、1ms の呼び出しでは 1.277ms → 1.271ms と差が無く、デフォルトプールが他の仕事で埋まっている状況でも同等）。各境界の呼び出しは元からその境界自身の `await` で直列化されているので、専用の 1 ワーカー executor が新たに何かを待たせることもない。むしろアイドル時のテール（no-op の max 10.9ms → 1.2ms）は共有プールを離れた分だけ改善する。
- キャンセル直後の `close` は最大で 1 ブロック分（録音は既定 `chunk=1024` / 16000Hz で 64ms、実運用の `chunk=1600` で 100ms、streaming VC は `block_ms` 既定 160ms、発話系の再生は再生中の発話の残り）だけ遅れて実行される。プロセス終了時はその分だけ終了が遅くなるが、`_python_exit` の join がそれを待つので終了コードは意図どおりになる。
- `close()` が実行中の呼び出しの後ろに queue された場合、その close が投げた例外を受け取る相手はもういない（`close_quietly` の try はとうに抜けている）。所有スレッド側で warning としてログに出す形にした。
- 不変条件（close は実行中の呼び出しと重ならない / それでも必ず閉じる）は 4 境界それぞれについて `tests/test_device_stream_close_race.py` で固定した。ネイティブクラッシュ自体は確率的なので、クラッシュの有無ではなく順序を検定する形にしてある。
