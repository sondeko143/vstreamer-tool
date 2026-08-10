# 0076. preflight はデバイスレートを実際に開いて検証し、比が分からない境界は標準レート族に対して確認する

- Status: Accepted
- Date: 2026-08-11
- Related: [0038](0038-worker-config-preflight-fail-loud.md)（preflight のレイヤ分割を定めた ADR）, [0073](0073-device-boundary-inhouse-polyphase-resampler.md), [0074](0074-device-native-rate-resolution.md)（この検査対象を作った ADR）, [0075](0075-wire-sample-rate-validation.md)（`MAX_PROTOTYPE_TAPS` と標準レート族の gcd 解析）

## Context

[0074](0074-device-native-rate-resolution.md) により、4 つのデバイス境界（`recording` 入力 / `playback` 出力 / `stream_vc` 入力・出力）はそれぞれ `resolve_device_rate` でレートを解決してから開くようになった。解決できない場合は `DeviceRateUnresolvedError`、解決したレートを PortAudio が拒否する場合は `sd.PortAudioError` になるが、いずれも**worker が実際に起動するまで**分からない。preflight（[0038](0038-worker-config-preflight-fail-loud.md) の layer A）は「実資源を確保せずに安く判定できるもの」だけを扱う設計で、この2つはこれまで range 外だった。

さらに [0075](0075-wire-sample-rate-validation.md) が `PolyphaseResampler` に `MAX_PROTOTYPE_TAPS` の上限を入れたことで、病的な変換比は資源確保の前に `ValueError` で拒否されるようになったが、拒否されるタイミングは境界ごとに違う。発話系の再生（`worker/playback.py`）は発話ごとに `ValueError` を warning で握りつぶして次へ進む(送信側から来る `sound.rate` は設定の問題とは限らないため)。一方 `recording` / `stream_vc` 入力・出力の3箇所は、デバイスレートとパイプライン側のレートの組がどちらも実質固定（前者は `config.recording.rate` / `CAPTURE_RATE` という設定定数、後者は一度選んだ RVC モデルが有効な間ずっと同じ `target_sample_rate` を吐き続ける）なので、**設定を直さない限り同じ変換比が毎回失敗する**。この3つを worker 起動時の失敗のままにしておくのは、[0038](0038-worker-config-preflight-fail-loud.md) が fail-loud をレイヤ A に寄せた狙いと合わない。

`recording` / `stream_vc` 入力の2つは、比較先のパイプラインレートが設定定数として preflight の時点で厳密に分かる（`config.recording.rate` / `stream_vc/capture.py` の `CAPTURE_RATE`）。しかし `stream_vc` 出力の比較先（RVC モデルの `target_sample_rate`）は ONNX モデルのメタデータからしか読めず、それを preflight で読むには GPU セッションを構築する必要がある — まさに layer A が避けている「実資源の確保」そのもので、しかも起動のたびに毎回コストを払うことになる。

## Decision

**(1) レート解決とデバイス開通は、4 境界すべてで preflight が検証する。** `resolve_device_rate` を呼び、`DeviceRateUnresolvedError` を worker と同じ `field`（`*.input_device_rate` / `*.output_device_rate`）を持つ `ConfigProblem` に変換する。続けて、解決したレートで実際に `sd.RawInputStream` / `sd.RawOutputStream` を**開いて即座に閉じる**(読み書きはしない)。開けなければ `ConfigProblem` に、そのレートの数値をメッセージへ埋め込んで報告する。**これは preflight が実資源を確保する唯一の検査になる** — 解決したレートを PortAudio が受け付けるかどうかは、実際に開いてみる以外に知りようがないため。出力側の2境界(`playback` / `stream_vc` 出力)は、[0073](0073-device-boundary-inhouse-polyphase-resampler.md)/[0074](0074-device-native-rate-resolution.md) がデバイスをネイティブレート固定で開くようにする前は最初のパケット/発話が来るまでレートが決まらなかったので、起動時に検証できるのは今回が初めて。`Pa_OpenStream`(構築)ではなく `Pa_StartStream` (`.start()`) まで踏み込むのは意図的なトレードオフである: `paInvalidSampleRate` は `Pa_OpenStream` の時点で出るので `.start()` を呼ぶこと自体は受入基準が要求する以上に強い検査だが、その分 preflight を実行するたびに実際にエンドポイントを起動することになり(アンプの起動音や、環境によっては他アプリのオーディオダッキングといった副作用がありうる)、引き換えに open は通るが start だけ失敗する個体差(排他デバイスの取り合いなど)まで拾える。

**(2) 病的な変換比は `recording` / `stream_vc` 入力・出力の3境界で preflight が拒否する。** `make_resampler(device_rate, target)` を呼び、`ValueError`(= [0075](0075-wire-sample-rate-validation.md) の `MAX_PROTOTYPE_TAPS` 超過)を、その `device_rate` と `target` を埋め込んだ `ConfigProblem` に変換する。`recording` は `target = config.recording.rate` 1つ、`stream_vc` 入力は `target = CAPTURE_RATE` 1つ — どちらも設定定数なので、その1つを厳密に検査すれば足りる。デバイス開通確認(1)より先に行い、既に病的と分かった device_rate に対してはハードウェアを開かない(同じ `field` に2つ目の無関係な問題が乗るのも避けられる)。

**(3) `stream_vc` 出力だけは、厳密な `target_sample_rate` の代わりに標準レート族に対して検査する。** モデルを preflight で読み込まずに済ませるため、[0075](0075-wire-sample-rate-validation.md) が実測で「4つの境界が正当に出会える中で最悪の対」と結論した標準オーディオレート族(8000/11025/16000/22050/24000/32000/40000/44100/48000/88200/96000/176400/192000、25Hz 格子の代表点)の**全メンバーに対して**検査する。1つでも爆発すれば `device_rate` 自体が病的と判断して `ConfigProblem` にする。実在の RVC モデルが吐くレート(32000/40000/48000)はこの族の内側にあり、[0075](0075-wire-sample-rate-validation.md) の実測(最悪でも 261k タップ・84ms)により全対が `MAX_PROTOTYPE_TAPS` 未満に収まると分かっているので、`device_rate` 自身が標準レート族に乗っている限り誤検知しない。**ただし拒否の原因が族のうち 8000/11025/88200/176400 のいずれかだけである場合、この境界にとっては構造的に偽陽性になりうる** — これら4点は実在する RVC モデルのレート(32000/40000/48000)ではないので、レビューで実測した具体例(`device_rate=49152Hz` は `11025` との対で 1,671,169 タップとなり拒否されるが、実際に対向しうる `48000` に対しては 125/128 と自明に軽い)のように、族の周辺値だけが原因の拒否は実運用では起こらない組み合わせを弾いている可能性がある。

発話系の再生(`worker/playback.py`)には (2)(3) のどちらも追加しない — 現状どおり発話ごとの warning に委ねる(Context 参照)。

## Alternatives rejected

- **`stream_vc` 出力の比較検査もしない(レート解決とデバイス開通だけ検証する)** — [0075](0075-wire-sample-rate-validation.md) が実測したとおり、病的な比は worker 起動後に subsystem ごと落とす致命傷であり、3境界のうち2つだけ起動時に潰しても「起動後に必ず落ちる設定」を1つ残すことになる。ADR-0038 の狙い(fail-loud を preflight に寄せる)に反する。
- **`stream_vc` 出力用に、preflight で ONNX モデルを実際に読み込んで正確な `target_sample_rate` を得る** — 正確だが、GPU セッションの構築は layer A が避けている「実資源の確保」そのもので、しかも `[vc]` の WHISPER バックエンドと同様に preflight を「起動のたびに数秒余計に待たされる」ものに変える。プロキシとしての標準レート族は、実在するモデルのレートを全て内側に含み、かつ [0075](0075-wire-sample-rate-validation.md) の実測で誤検知しないと分かっている範囲内で、この精度の犠牲は許容できる。
- **標準レート族の代わりに、実在する RVC モデルレート(32000/40000/48000)だけに絞る** — より狭く速いが、他の族メンバー(16000/24000/96000/192000 など)で学習された将来のモデルを preflight だけが理由なく拒否することになる。[0075](0075-wire-sample-rate-validation.md) が同じ判断([0075]の "Alternatives rejected" の "標準レートのホワイトリスト" 節)を既に下しており、preflight だけ狭い基準を持つのは一貫しない。
- **preflight でのデバイス開通確認をしない(worker 起動時の `PortAudioError` に任せる)** — brief の受入基準そのもの。出力側は [0073](0073-device-boundary-inhouse-polyphase-resampler.md)/[0074](0074-device-native-rate-resolution.md) 以前は実行時にしか分からなかったが、今はネイティブレート固定なので起動時に検証できる。この機体で確認した移行破壊(WDM-KS 64個 + 疑似デバイス4個がレート解決不能)を、起動後のクラッシュのままにせず起動時の明確なメッセージに変えるのが本 task の主目的の半分である。
- **開通確認を、実際の worker と全く同じ blocksize/channels/dtype/latency で行う** — `playback`(発話系)は実際の format/channels が config に無く実行時の発話ごとに決まるため、そもそも worker と同一の形状を preflight から再現できない。4境界とも channels=1・dtype=int16(recording のみ config の値)の最小限のプローブで統一し、検証するのは「このレートで開けるか」に絞る。`latency` もこの意図的な形状差の一つ: `stream_vc` の入出力は `latency="low"` で開く(`stream_vc/capture.py`, `stream_vc/playback.py`)が、preflight のプローブは既定 latency で開く。低レイテンシ要求だけを拒否するデバイスは preflight を通過し、worker の open で初めて落ちる — 実在確率は低いと見て、この差も他の形状差と同じ扱いのまま残す。
- **preflight の開通確認を `lib/audio.open_device_stream`(resolve→ログ→open→乖離 warning の共通手順)にそのまま委譲する** — レビューで「preflight は resolve→open→start を独自に複製しており、worker と同じ経路を通っていない」という指摘を受けたが、委譲するには preflight 側で先に確認した `rate` を `override=rate` として渡す必要があり(でないと `resolve_device_rate` が二重に呼ばれる)、そのログ行が実際には自動解決だったレートに対して「明示」と表記してしまい不正確になる。代わりに preflight 独自の最小プローブは維持しつつ、そこで失われていた「PortAudio が要求と違うレートを返したときの warning」(`open_device_stream` の step 4)だけを複製して埋めた — worker との検査の強さの差を実質的になくしつつ、ログの正確さも保つ。

## Consequences

- preflight は初めて実資源(音声デバイス)を開く。開いて即閉じるだけで読み書きはしないが、排他モードのデバイスや他プロセスが専有中のデバイスでは preflight 自体が新たな失敗点になりうる — これは意図した挙動で、worker 起動時に同じ理由で失敗していたものを早期化しただけである。
- `stream_vc` 出力の起動時チェックは、`device_rate` が標準レート族の外(WASAPI が奇妙な値を返す壊れたデバイスなど)にある場合のみ発火する。実在するモデルのレート(32000/40000/48000)を含め、実在するデバイスレート同士の組み合わせは全て通る([0075](0075-wire-sample-rate-validation.md) の実測に基づく)。
- `tests/test_preflight.py` は `sounddevice` をスタブする必要が生じた(以前はデバイス解決を `vspeech.lib.audio` の高レベル関数のモックだけで済ませていたが、レート解決とデバイス開通は `sd.query_hostapis` / `sd.query_devices` / `sd.RawInputStream` / `sd.RawOutputStream` まで届く)。テスト自体が実機のマイク/スピーカーを開かないよう、これらをスタブする autouse フィクスチャを追加した。
- `Pa_OpenStream` は成功したが `Pa_StartStream` (`.start()`) が失敗するケース(排他デバイスの取り合いなど)は、sounddevice のストリームに `__del__` が無く `close()` だけが `Pa_CloseStream` を呼ぶため、`close()` を呼ばずに例外を伝播させるとネイティブハンドルを漏らす。preflight 自身のプローブと、4境界すべてが通る共有ヘルパ `lib/audio.open_device_stream` の両方にこの穴があった(レビューで指摘、実装時は見落としていた)。両方とも `try: ... except: close(); raise` / `try/except/else/finally` で必ず閉じるように直し、`start()` が失敗しても `close()` が呼ばれることを直接表明するテストを追加した(`tests/test_device_rate.py::test_a_start_failure_still_closes_the_stream`、`tests/test_preflight.py` の `*_start_failure_still_closes_the_stream` 2件)。`open_device_stream` 側は成功時には閉じない(呼び出し側が開いたまま使うため)ことも合わせてテストで固定した。
