# 0036. whisper のリサンプルに PyAV(libswresample) を使い torchaudio/scipy を却下する

- Status: Accepted
- Date: 2026-07-16
- Related: `fix/whisper-transcription-resample` (f023b09), [0031](0031-audio-pyaudio-to-sounddevice.md)

## Context

`transcription.py` の `pcm_to_waveform` は faster-whisper に float32 ndarray を直接渡す（ディスク往復を廃した 72a9af4 以降）。だが faster-whisper が自動リサンプルするのは file/bytes 入力だけで、**ndarray は 16kHz 前提でそのまま消費**する。録音レートが 16kHz でないと音声が誤った速度でモデルに入り「謎のカタカナ」になる（実機で発生）。よって transcription 側でリサンプルが要る。

制約：

- whisper extra は軽量・cross-platform を保ちたい。`torch` は whisper extra では `sys_platform=='win32'` 限定で **Linux には入らない**。`torchaudio`・`scipy` は whisper extra に**無い**（rvc extra 側）。
- リサンプルはダウンサンプル（例 48k→16k）を含むのでアンチエイリアスが要る。
- 発生頻度は低い（非16kのときだけ）が、無ければ文字起こしが静かに壊れる。

3手法を同一機で実測（48k→16k, 5秒モノ, 呼び出しごと＝フィルタ構築込み）：

| 手法 | 速度 | 通過帯域1k | 10kエイリアス抑圧 | 追加サイズ |
|---|---:|---:|---:|---:|
| av (libswresample) | 19.4 ms | 完璧 | −50.3 dB | 0（faster-whisper 同梱） |
| scipy resample_poly | 5.4 ms | 完璧 | −53.2 dB | +120 MB |
| torchaudio resample | 5.4 ms | 完璧 | −44.9 dB | +torch 2.65 GB |

品質は3手法とも ASR には十分（≥45dB 抑圧・通過帯域完璧）。速度差（14ms/回）は whisper 本体 ~400ms の前かつ非16kのときだけなので実用上無視できる。決め手は**依存の相性**。

## Decision

whisper パスのリサンプルは **PyAV の `AudioResampler`(libswresample)** で行う。`av` は faster-whisper の直接依存なので**追加依存ゼロ・win/Linux 両対応**。`pcm_to_waveform` は format 対応デコード → mono 化 → `sound.rate != 16000` のとき 16kHz へリサンプル、を常に 16kHz float32 ndarray で返す。

コードベースの住み分けを明文化する：**VC/RVC パスは torchaudio（torch 前提・GPU 可）、whisper パスは av（torch-free・同梱）**。

## Alternatives rejected

- **torchaudio.functional.resample** — torch(2.65GB)+torchaudio が前提。whisper extra の torch は win32 限定で **Linux の whisper インストールでは使えない**。torch が既に居る VC/RVC パスでは最適だが、whisper パスの携帯性を壊すため却下。
- **scipy.signal.resample_poly** — 最速かつ最良の抑圧(−53dB)だが、稀にしか通らないコードパスのために **120MB の cross-platform 依存を whisper extra へ足す**のは割に合わず却下。
- **numpy 線形補間** — アンチエイリアス無しでダウンサンプル時にエイリアスを生む。品質不足で却下。
- **BytesIO WAV を faster-whisper に渡して自動リサンプルさせる** — 動くが WAV 再エンコードが要り、`wave` モジュールは float32 を扱えない（int32 と誤解）。ndarray 高速パスの方が素直なので却下。

## Consequences

- 追加依存なし・cross-platform。既定の 16kHz/INT16 パスは従来どおり ndarray 直渡しで**ビット不変**（リサンプルは非16kのときだけ走る）。
- av は per-call のリサンプラ構築＋frame marshaling で scipy/torchaudio の約3.6倍遅い（19 vs 5.4ms）が、非16k時のみ・whisper 本体の前では誤差。
- av の例外面（空フレームで `av.error.MemoryError` 等）は worker の `ValueError` catch より広い。空入力は PyAV に触れる前に早期 return でガード済み（f023b09）。
- 将来 whisper 側で高品質・高速リサンプルが必要になったら scipy 追加を再検討する余地がある（本 ADR を supersede）。
