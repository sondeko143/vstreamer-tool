# VStreamer Tool

AmiVoice Cloud Platform API, VOICEROID2, Google translation API v3 などと連携してそういう配信をするときに使う  
以下の機能を実装

1. 録音
2. 文字起こし
    - [AmiVoice Cloud Platform API](https://acp.amivoice.com/)
    - [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text?hl=ja)
    - [Whisper](https://github.com/SYSTRAN/faster-whisper) (faster-whisper)
3. 音声合成
    - VOICEROID2
    - [VOICEVOX](https://github.com/VOICEVOX/voicevox_core)
4. 翻訳 (Google translation API v3)
5. 字幕
6. ボイスチェンジャー
    - RVC モデル
    - [VC Client](https://github.com/w-okada/voice-changer) を参考にさせていただいております

## 設定

Python 3.14 が必要です (`>=3.14,<3.15`)。uv で依存パッケージをインストールします。

```sh
# 全部入り。迷ったらこれ
uv sync --all-extras

# 一部だけ入れる場合は、必要な extra を 1 つのコマンドに並べる
uv sync --extra audio --extra whisper
```

**`uv sync` は指定した extras の集合に環境を合わせます。**あとから
`uv sync --extra rvc` と単独で叩くと、それ以外の extras (`voicevox`, `whisper`,
`audio` …) は**アンインストールされます**。機能を「足す」つもりで実行すると壊れます。

**`rvc` extra は NVIDIA の CUDA ランタイム wheel を 6 個取ってきます。ダウンロード
合計 1,027MB (約 1.0GiB)、展開後の `site-packages/nvidia/` は 1,481MB です**
(前者は `uv.lock` に記録された wheel サイズの合計、後者は実測)。`--all-extras` はこれを
含みます。`onnxruntime-gpu` の wheel は CUDA を同梱しないので、これが唯一の供給元です
([ADR-0083](docs/adr/0083-cuda-runtime-from-nvidia-wheels.md))。従量課金や低速な回線では
先に把握しておいてください。RVC を使わないなら `rvc` を外した extra 指定で回避できます。

| extra | 内容 |
| --- | --- |
| (なし) | 文字起こし / 翻訳 / 字幕 |
| `audio` | 録音・再生 (sounddevice。PortAudio は wheel に同梱されるので別途インストールは不要) |
| `whisper` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) による文字起こし |
| `vroid2` | VOICEROID2 音声合成 |
| `voicevox` | VOICEVOX 音声合成 |
| `rvc` | RVC ボイスチェンジャー |
| `mozc` | AmiVoice の結果をかな漢字変換する (`transcription.transliterate_with_mozc`) |

設定項目は `config.toml.example` や `vspeech/config.py` を参照してください。ごめんなさい。

VOICEROID2 を使う場合は 64bit 版エディターがダウンロードされている必要があります。

VOICEVOX を使う場合、`voicevox-core` の wheel には **ONNX Runtime・Open JTalk 辞書・
音声モデル (`.vvm`) が含まれていません**。[VOICEVOX のダウンローダ](https://github.com/VOICEVOX/voicevox_core/releases/tag/0.16.4)
で取得し、`voicevox.openjtalk_dir` / `model_dir` / `onnxruntime_path` を指してください。
既定値は `./voicevox/dict/open_jtalk_dic_utf_8-1.11` と `./voicevox/models/vvms` です。

```sh
curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe -o download-voicevox.exe
./download-voicevox.exe -o ./voicevox --exclude c-api --devices cuda
```

(テストスイート用の資産は `uv run poe voicevox-assets` が `tests/assets/voicevox` へ取得します。)

VOICEVOX は whisper / RVC が使う `onnxruntime-gpu` とは**別ビルド**の
`voicevox_onnxruntime` を読みます。正しい方が読まれるよう `onnxruntime_path` は明示してください。

GPU を使う場合、CUDA 13 ランタイム (cuBLAS / cuDNN / cuFFT / cudart) は `rvc` extra が
名指しで pin している `nvidia-*` wheel が venv 内に供給します。`onnxruntime-gpu` の wheel は
CUDA を同梱しないので、これが唯一の供給元です
([ADR-0083](docs/adr/0083-cuda-runtime-from-nvidia-wheels.md))。したがって
RVC / ボイスチェンジャーだけのホストは **NVIDIA ドライバ R580 以降**があれば足ります
([ADR-0028](docs/adr/0028-migrate-to-cuda-13.md))。

whisper を GPU で回すホストは、それに加えて **CUDA 12 ツールキット (cuBLAS + cuDNN 9)** が要ります。
faster-whisper が使う ctranslate2 は CUDA 12 専用ビルドしかなく `cublas64_12.dll` を要求しますが、
venv に入るのは CUDA 13 世代の wheel だけだからです
([ADR-0039](docs/adr/0039-whisper-hosts-need-cuda12-toolkit.md))。

## 実行

```sh
uv run python -m vspeech --config ./config.toml
```

走っている pipeline の操作（**すでに走っている** pipeline へ疎通確認 / pause /
resume / reload を送る。起動・設定編集はしない。extra 不要 — 追加依存はゼロ。
[ADR-0061](docs/adr/0061-remote-control-as-cli.md)）

```sh
uv run vsctl ping   --to 192.0.2.10:8080     # 疎通確認 (exit 0 = 届いた)
uv run vsctl pause  --to 192.0.2.10:8080
uv run vsctl resume --to 192.0.2.10:8080
uv run vsctl reload --to 192.0.2.10:8080 --config-path D:/vstreamer/config.toml

# 毎回打ちたくなければ環境変数へ (--to の既定値)
export VSPEECH_TARGET=192.0.2.10:8080
uv run vsctl pause

uv run python -m cli --help                  # vsctl と同じもの
```

宛先は操作対象 pipeline の `listen_address:listen_port`。終了コードは操作の成否
そのもの（0 = 相手が受け取った / 1 = 失敗 / 2 = 引数の誤り）なので、スクリプトや
配信ソフトのコマンド実行から繋げられる。

`reload` の `--config-path` は**対象マシン上のパス**で、reload を受けた側が自分で
開く。
