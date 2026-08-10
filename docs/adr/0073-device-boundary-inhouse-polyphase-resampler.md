# 0073. デバイス境界のサンプルレート変換を OS から自前 numpy ポリフェーズへ移す（0036 を refine）

- Status: Accepted
- Date: 2026-08-10
- Related: [spec](../superpowers/specs/2026-08-10-device-sample-rate-in-process-design.md), [0036](0036-whisper-resample-via-pyav.md)（本 ADR が refine する住み分け）, [0038](0038-worker-config-preflight-fail-loud.md), [0050](0050-streaming-vc-separate-subsystem.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0055](0055-stream-vc-producer-consumer-role-split.md), [0069](0069-torch-213-and-terminal-torchaudio.md), [0074](0074-device-native-rate-resolution.md)

## Context

オーディオデバイスを開くとき、要求したレートがデバイスのネイティブレートと違えば、変換は PortAudio より下の OS 側で暗黙に行われる。本プロジェクトは 4 箇所すべてでそうなっている — ストリーミング VC の入口(16kHz 固定)、同出口(RVC モデルのレート)、発話系の録音(`recording.rate`)、発話系の再生(音声ソースのレート)。

この変換の素性（アンチエイリアス特性・遅延）はホスト API とドライバに依存し、こちらから決められない。自動テストで検証できず、ログにも現れない。加えて実測で、WASAPI 共有モードはミックスフォーマット以外のレートを拒否する（`Invalid sample rate [PaErrorCode -9997]`）。つまりネイティブレート以外で開く現状の実装では WASAPI をそもそも選べず、MME/DirectSound に固定されている。

実装を選ぶ上で効いている制約は 3 つある。

- **consumer ロールは torch を引かない**（[0055](0055-stream-vc-producer-consumer-role-split.md)）。出口側のリサンプラは torch-free でなければならない。
- **numpy は全インストールに必ず存在する**。base 依存の `ctranslate2` が numpy を要求するため、`audio` extra だけの再生専任機にも入る。
- **ストリーミング経路は固定ブロックに再ブロック化する**（[0053](0053-streaming-vc-fixed-block-crossfade.md)）。滞留 d>0 のリサンプラをここに素で挟むと、d が何 ms であろうと配信が恒常的に 1 ブロック遅れ、実測で **+160ms** になる（インパルスを実パイプライン模擬に通して確認）。

[0036](0036-whisper-resample-via-pyav.md) は既に「whisper 入力 = PyAV / VC 内部 = torchaudio」という住み分けを定めている。本 ADR はそこに「デバイス境界」という 3 つめの境界を足して refine する。

## Decision

デバイス境界（ストリーミング VC の入口・出口、発話系の録音・再生の 4 箇所）のサンプルレート変換は、**numpy のみで書いた有理比ポリフェーズ FIR**（Kaiser 窓 sinc、阻止域 -90dB 級、半長は `scipy.signal.resample_poly` と同じ設計則）でプロセス内で行う。デバイスは既定でネイティブレートで開き、OS の変換段を no-op にする。

- リサンプラは **ストリーミング用（状態保持）** と **ワンショット用（末尾までフラッシュ）** の 2 モードを持つ。前者は連続ストリーム（capture・ストリーミング再生・発話系録音）、後者は 1 発話ごとに独立した buffer（発話系再生）に使う。
- **固定ブロックへ再ブロック化する入口でも、事前充填は要らない。** 因果ポリフェーズは出力本数が `ceil(L*n/M)` で欠けないため、1 device tick あたり 1 ブロックがそのまま出る(実測で配信遅れ min=max=0)。事前充填が要るのは soxr のように滞留を内部に抱える実装で、そこでは滞留が丸ごと 1 hop の遅延に量子化される(実測 +160ms)。この差が自前実装を選んだ理由そのものなので、Alternatives rejected の soxr 項と合わせて読むこと。
- デバイス境界での float32 → 整数 PCM 変換は**必ず飽和クリップ**で行う。リサンプルは Gibbs 現象で元のピークを超えうるため、ラップアラウンドさせるとクリックになる。
- リサンプラの住み分けを次のとおり明文化する。**デバイス境界 = 自前ポリフェーズ / whisper 入力 = PyAV（[0036](0036-whisper-resample-via-pyav.md) のまま）/ RVC 内部 = torchaudio**。後ろ 2 つは本 ADR では変更しない。

## Alternatives rejected

- **soxr（python-soxr）を依存に足す** — cp314 で動作し `ResampleStream` も揃っていることは確認した。だが python-soxr は出力バッファを入力長から算出するため内部の滞留を取り切れず、滞留量が **HQ で 6〜37ms、VHQ で 9〜76ms の幅で振動する**（投入チャンクサイズを 5ms〜160ms で振っても変わらない = soxr 内部の固定バッファ）。振動する以上、上記の hop 量子化を避ける事前充填は最大値に合わせるしかなく、入口+出口の追加遅延が **LQ 26ms / HQ 63ms / VHQ 102ms** になる。ブロック 160ms・推論 floor 40ms のこの経路では割に合わない。自前実装は毎回出せる分を全部出すので滞留がフィルタ半長で固定され、振動しない。品質・速度は soxr が上だが、-90dB 級の阻止域は int16 かつ RVC を通るこの経路では既に十分。
- **PyAV (libswresample)** — [0036](0036-whisper-resample-via-pyav.md) の実績があり品質も十分だが、`av` は faster-whisper 経由で `whisper` extra にしか入っていない。`audio` extra へ足すと ffmpeg バイナリ約 35MB が再生専任機（[0055](0055-stream-vc-producer-consumer-role-split.md) の consumer）にも乗る。
- **`vspeech/lib/rvc.py` の `get_resampler`（torchaudio）を再利用する** — VC 側には既にあるが、consumer ロールは torch を引かないので出口では使えず、入口と出口で実装が 2 本に割れる。[0069](0069-torch-213-and-terminal-torchaudio.md) が torchaudio の撤去を将来課題に挙げている方向とも逆行する。
- **`scipy.signal.resample_poly`** — scipy は `rvc` extra にしかなく、再生専任機に 120MB を足すことになる。[0036](0036-whisper-resample-via-pyav.md) が同じ理由で却下済み。
- **リサンプルは OS 任せのまま、ホスト API だけ WASAPI へ寄せる** — WASAPI 共有はネイティブレート以外を拒否するので、レート変換を自前でやらない限りそもそも開けない。順序が逆で、単独では成立しない。
- **soxr を採ったうえで入口で事前充填せず、滞留を許容する** — 実測 +160ms。ブロック長まるごとの遅延なので、この経路では受け入れられない。

## Consequences

- **依存追加ゼロ**。numpy は `ctranslate2` 経由で base に必ず入るため、producer / consumer / 発話系のすべてで同一実装が動く。
- 変換特性（阻止域・通過帯平坦性・ブロック分割一致・群遅延）を pytest で数値としてアサートできる。「OS 任せで確かめられない」が構造的に消える。
- デバイスをネイティブレートで開くようになるので、WASAPI / 排他 / WDM-KS が**選べるように**なる。ただし既定のホスト API 選択は変えない（それは別の決定であり、本 ADR のスコープ外）。
- 発話系の再生は出力ストリームのレートが静的になり、**音声ソースのレート変化でデバイスを開き直さなくなる**（TTS 24kHz → VC 40kHz の交互再生でリオープンが消える）。副次的に、出力デバイスを preflight で検証できるようになる（今はレートが実行時にしか決まらず `check_output_settings` を掛けられない）。
- **既存ユーザーの音が変わる**（OS のリサンプラ → 自前）。実機の耳確認が要る。デバイスレートと境界を流れる音声のレートが一致する場合は素通しでビット不変。
- フィルタ設計を自前で持つ責任が生まれる。テストで固定するので、係数や窓を変えるときは阻止域・通過帯のアサートを通す必要がある。
- リサンプラ実装が 3 つ併存する（自前 / PyAV / torchaudio）。境界ごとの使い分けを本 ADR で明文化したので、新しい変換を足すときは「どの境界か」で選ぶ。RVC 内部を差し替えると seeded golden が壊れることは [0069](0069-torch-213-and-terminal-torchaudio.md) が実証済みで、そこへ手を出す理由は本 ADR には無い。
